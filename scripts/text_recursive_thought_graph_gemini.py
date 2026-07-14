from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - the repo environment already has it.
    load_dotenv = None

try:
    from google import genai
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: install google-genai to run this script.") from exc


DEFAULT_MODEL = "gemini-2.5-flash-lite"

STOPWORDS = {
    "a",
    "aby",
    "ale",
    "by",
    "byl",
    "byla",
    "bylo",
    "co",
    "do",
    "je",
    "jen",
    "jako",
    "jak",
    "kde",
    "kdy",
    "ktery",
    "ktera",
    "ma",
    "musi",
    "na",
    "nebo",
    "nejpozdeji",
    "neni",
    "po",
    "podle",
    "pokud",
    "pred",
    "proc",
    "se",
    "si",
    "tak",
    "tedy",
    "ten",
    "to",
    "u",
    "uz",
    "v",
    "ve",
    "z",
    "za",
    "ze",
}


@dataclass
class Question:
    id: str
    text: str
    aliases: list[str] = field(default_factory=list)
    required_terms: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    id: str
    task_type: str
    context: str
    questions: list[Question]


def load_env() -> None:
    env_path = Path(".env")
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path)
        return
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = "".join(
        char for char in unicodedata.normalize("NFD", value) if unicodedata.category(char) != "Mn"
    )
    value = re.sub(r"[^a-z0-9: ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def contains_term(normalized_answer: str, term: str) -> bool:
    normalized_term = normalize_text(term)
    if not normalized_term:
        return False
    if normalized_term in {"ano", "ne"}:
        return normalized_term in set(normalized_answer.split())
    if re.fullmatch(r"\d+(?::\d{2})?", normalized_term):
        return normalized_term in set(normalized_answer.split())
    if normalized_term in normalized_answer:
        return True
    if any(len(token) == 1 for token in normalized_term.split()):
        return False
    term_tokens = token_set(normalized_term)
    answer_tokens = token_set(normalized_answer)
    return bool(term_tokens) and term_tokens.issubset(answer_tokens)


def answer_matches(answer: str, question: Question) -> bool:
    normalized_answer = normalize_text(answer)
    alias_ok = True
    if question.aliases:
        alias_ok = any(contains_term(normalized_answer, alias) for alias in question.aliases)
    required_ok = all(contains_term(normalized_answer, term) for term in question.required_terms)
    forbidden_ok = not any(contains_term(normalized_answer, term) for term in question.forbidden_terms)
    return alias_ok and required_ok and forbidden_ok


def extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        fenced = match.group(1).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            text = fenced

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
            return parsed
        except json.JSONDecodeError:
            continue

    first_obj = text.find("{")
    first_arr = text.find("[")
    starts = [idx for idx in [first_obj, first_arr] if idx >= 0]
    if not starts:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        recovered = recover_incomplete_answer_json(text)
        if recovered is not None:
            return recovered
        raise ValueError(f"Incomplete JSON response: {text[:200]}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        recovered = recover_incomplete_answer_json(text)
        if recovered is not None:
            return recovered
        raise


def recover_incomplete_answer_json(text: str) -> dict[str, Any] | None:
    answer_match = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)', text, flags=re.DOTALL)
    if not answer_match:
        return None
    answer_raw = answer_match.group(1)
    try:
        answer = json.loads(f'"{answer_raw}"')
    except json.JSONDecodeError:
        answer = answer_raw
    rationale_match = re.search(r'"rationale"\s*:\s*"((?:\\.|[^"\\])*)', text, flags=re.DOTALL)
    rationale = ""
    if rationale_match:
        try:
            rationale = json.loads(f'"{rationale_match.group(1)}"')
        except json.JSONDecodeError:
            rationale = rationale_match.group(1)
    return {
        "answer": answer,
        "rationale": rationale,
        "confidence": 0.0,
        "_recovered_from_incomplete_json": True,
    }


class GeminiJsonClient:
    def __init__(self, model: str, temperature: float = 0.0, max_retries: int = 5) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is missing. Put it in .env or the environment.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.calls: list[dict[str, Any]] = []

    def json_call(self, role: str, prompt: str, max_output_tokens: int = 2048) -> Any:
        config = {
            "temperature": self.temperature,
            "response_mime_type": "application/json",
            "max_output_tokens": max_output_tokens,
        }
        last_error: str | None = None
        started = time.perf_counter()
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                raw = response.text or ""
                parsed = extract_json(raw)
                self.calls.append(
                    {
                        "role": role,
                        "attempt": attempt,
                        "latency_s": time.perf_counter() - started,
                        "prompt_chars": len(prompt),
                        "response_chars": len(raw),
                    }
                )
                return parsed
            except Exception as exc:  # Gemini SDK exceptions are version-specific.
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(1.5 * attempt + random.random())
        fallback = self.fallback_response(role, last_error or "unknown error")
        if fallback is not None:
            self.calls.append(
                {
                    "role": role,
                    "attempt": self.max_retries,
                    "latency_s": time.perf_counter() - started,
                    "prompt_chars": len(prompt),
                    "response_chars": 0,
                    "failed_with_fallback": True,
                    "error": last_error,
                }
            )
            return fallback
        raise RuntimeError(f"Gemini call failed after {self.max_retries} attempts: {last_error}")

    def fallback_response(self, role: str, error: str) -> Any | None:
        if role == "question_slots":
            return {"slots": [], "_call_failed": error}
        if role == "verifier":
            return {"accepted_ids": [], "rejected": [], "_call_failed": error}
        if (
            role == "atomizer"
            or role.startswith("composer")
            or role in {"loop_expand", "task_graph"}
        ):
            return {"nodes": [], "_call_failed": error}
        if role.startswith("answer"):
            return {"answer": "", "rationale": "", "confidence": 0.0, "_call_failed": error}
        return None


def make_scenarios() -> list[Scenario]:
    scenarios = [
        Scenario(
            id="transport_policy",
            task_type="plan_constraints",
            context=(
                "Petrův autobus přijel v 9:35 místo v 9:00. Tím zmeškal vlak v 9:20. "
                "Další pravidelný vlak odjíždí v 10:50 a přijíždí do Prahy ve 12:10. "
                "Petr má být na schůzce v Praze v 11:30. Expresní autobus odjíždí v 9:55 "
                "a přijíždí v 11:25, ale lístek musí být koupen nejpozději v 9:50. "
                "Taxi by dorazilo v 11:05. Firemní pravidlo dovoluje taxi jen tehdy, "
                "když pravidelný vlak dorazí až po začátku schůzky."
            ),
            questions=[
                Question(
                    id="regular_train_on_time",
                    text="Stihne Petr schůzku dalším pravidelným vlakem?",
                    aliases=["ne", "nestihne"],
                    forbidden_terms=["ano"],
                ),
                Question(
                    id="taxi_allowed_on_time",
                    text="Je taxi podle firemního pravidla dovoleno a dorazí včas?",
                    aliases=["ano"],
                    required_terms=["včas"],
                    forbidden_terms=["ne", "není jisté", "nelze určit"],
                ),
                Question(
                    id="express_ticket_deadline",
                    text="Do kdy nejpozději by musel koupit lístek na expresní autobus?",
                    aliases=["9:50"],
                ),
            ],
        ),
        Scenario(
            id="nested_relocation",
            task_type="nested_objects",
            context=(
                "Červená složka je v obálce A. Obálka A je v krabici B. "
                "Krabice B je na polici C. Karel potom vyndal obálku A z krabice B "
                "a položil ji do zásuvky D. Prázdnou krabici B nechal na polici C. "
                "Modrá složka po celou dobu zůstala v krabici B."
            ),
            questions=[
                Question(
                    id="red_folder_final",
                    text="Kde je nakonec červená složka?",
                    aliases=["zásuvce d", "v zásuvce d", "zásuvky d", "do zásuvky d"],
                ),
                Question(
                    id="red_on_shelf",
                    text="Je červená složka stále na polici C?",
                    aliases=["ne", "není"],
                    forbidden_terms=["ano"],
                ),
                Question(
                    id="blue_folder_final",
                    text="Kde zůstala modrá složka?",
                    aliases=["krabici b", "na polici c"],
                    required_terms=["krabici b"],
                ),
            ],
        ),
        Scenario(
            id="release_schedule",
            task_type="temporal_constraints",
            context=(
                "Revize kódu začala ve 12:40 a trvá 25 minut. Build smí začít až po revizi "
                "a pevně začíná v 13:10. Testy trvají 45 minut a začínají ihned po buildu. "
                "Nasazení smí začít nejdříve 20 minut po dokončení testů. Demo zákazníkovi "
                "je ve 14:30. Nouzová cesta bez revize je zakázaná."
            ),
            questions=[
                Question(
                    id="earliest_deploy",
                    text="Kdy nejdříve může začít nasazení?",
                    aliases=["14:15"],
                ),
                Question(
                    id="before_demo",
                    text="Může nasazení začít před demem zákazníkovi?",
                    aliases=["ano"],
                    forbidden_terms=["ne"],
                ),
                Question(
                    id="skip_review_allowed",
                    text="Je povolené zrychlit plán vynecháním revize?",
                    aliases=["ne", "není"],
                    forbidden_terms=["ano"],
                ),
            ],
        ),
        Scenario(
            id="false_belief_chain",
            task_type="belief",
            context=(
                "Mapa byla ráno v šuplíku. Eva přesunula mapu do batohu, když Marek nebyl "
                "v místnosti. Jana viděla Evu mapu přesunout. Marek později řekl Janě, "
                "že podle něj mapa zůstala v šuplíku. Jana Markovi neprozradila, že přesun viděla."
            ),
            questions=[
                Question(
                    id="marek_search",
                    text="Kde bude Marek pravděpodobně hledat mapu?",
                    aliases=["v šuplíku", "šuplíku"],
                ),
                Question(
                    id="jana_knows_real",
                    text="Ví Jana, že mapa je ve skutečnosti v batohu?",
                    aliases=["ano"],
                    forbidden_terms=["ne"],
                ),
                Question(
                    id="real_location",
                    text="Kde je mapa ve skutečnosti?",
                    aliases=["v batohu", "batohu"],
                ),
            ],
        ),
        Scenario(
            id="approval_pipeline",
            task_type="plan_constraints",
            context=(
                "Lucie musí odeslat nabídku klientovi do 15:00. Nabídku musí nejdřív schválit "
                "Anna, která je dostupná jen do 14:00. Po Annině schválení následuje právní "
                "kontrola dlouhá 30 minut a potom export dlouhý 20 minut. Odeslání bez Annina "
                "schválení klient automaticky odmítne."
            ),
            questions=[
                Question(
                    id="latest_anna_approval",
                    text="Do kdy nejpozději musí Lucie získat Annino schválení?",
                    aliases=["14:00"],
                ),
                Question(
                    id="post_approval_duration",
                    text="Kolik času zabere právní kontrola a export po schválení dohromady?",
                    aliases=["50 minut", "padesát minut"],
                ),
                Question(
                    id="send_without_approval",
                    text="Je rozumné poslat nabídku bez Annina schválení?",
                    aliases=["ne", "není"],
                    forbidden_terms=["ano"],
                ),
            ],
        ),
        Scenario(
            id="warehouse_counts",
            task_type="state_updates",
            context=(
                "Sklad A měl ráno 18 červených senzorů. Sklad B měl ráno 4 červené senzory. "
                "Ze skladu A se přesunulo 7 senzorů do skladu B. Ve skladu B se poté vyřadily "
                "3 poškozené červené senzory. Nakonec přišla dodávka 5 nových červených senzorů "
                "do skladu A. Přesun mezi sklady nemění celkový počet, vyřazení ano."
            ),
            questions=[
                Question(
                    id="warehouse_a_final",
                    text="Kolik červených senzorů má nakonec sklad A?",
                    aliases=["16"],
                ),
                Question(
                    id="warehouse_b_final",
                    text="Kolik červených senzorů má nakonec sklad B?",
                    aliases=["8"],
                ),
                Question(
                    id="total_final",
                    text="Kolik červených senzorů je nakonec celkem v obou skladech?",
                    aliases=["24"],
                ),
            ],
        ),
        Scenario(
            id="access_permissions",
            task_type="logic_constraints",
            context=(
                "Nina má aktivní účet a roli viewer. Viewer smí číst dashboard, ale nesmí exportovat data. "
                "Export dat vyžaduje roli analyst a platné dvoufaktorové ověření. Nadřízený dnes schválil "
                "dočasnou roli analyst, takže role analyst je připravena k aktivaci. Ninino dvoufaktorové "
                "ověření však vypršelo včera a zatím nebylo obnoveno."
            ),
            questions=[
                Question(
                    id="can_export_now",
                    text="Může Nina právě teď exportovat data?",
                    aliases=["ne", "nemůže"],
                    forbidden_terms=["ano"],
                ),
                Question(
                    id="missing_requirement",
                    text="Která podmínka pro export stále chybí?",
                    aliases=[
                        "dvoufaktorové ověření",
                        "2fa",
                        "platné ověření",
                        "obnovení dvoufaktorového ověření",
                    ],
                ),
                Question(
                    id="can_read_dashboard",
                    text="Může Nina číst dashboard?",
                    aliases=["ano"],
                    forbidden_terms=["ne"],
                ),
            ],
        ),
        Scenario(
            id="incident_revision",
            task_type="revision_conflict",
            context=(
                "První hlášení tvrdilo, že výpadek způsobila mašina A. Pozdější kontrola zjistila, "
                "že časové značky senzoru A byly posunuté o deset minut, takže první hlášení je "
                "nespolehlivé. Záložní log ukazuje, že mašina B se přehřála v 10:12 a výpadek začal "
                "v 10:15. Mašina A se restartovala až v 10:20, tedy po začátku výpadku. Poslední "
                "oprava má přednost před prvním hlášením."
            ),
            questions=[
                Question(
                    id="actual_cause",
                    text="Která mašina podle opravených informací způsobila výpadek?",
                    aliases=["mašina b"],
                    forbidden_terms=["mašina a"],
                ),
                Question(
                    id="a_was_cause",
                    text="Byla mašina A příčinou výpadku?",
                    aliases=["ne", "nebyla"],
                    forbidden_terms=["ano"],
                ),
                Question(
                    id="why_first_unreliable",
                    text="Proč je první hlášení nespolehlivé?",
                    aliases=["časové značky", "posunuté", "senzoru a"],
                ),
            ],
        ),
        Scenario(
            id="bike_route",
            task_type="route_constraints",
            context=(
                "Jana jede na kole na úřad v 10:00. Most přes řeku je uzavřený. Tunel je otevřený "
                "jen pro auta, ne pro kola. Přívoz bere kola a odjíždí v 9:40, na druhé straně je "
                "v 9:55. Taxi s držákem na kolo by přijelo až v 10:10. Jana nechce nechat kolo "
                "bez dozoru."
            ),
            questions=[
                Question(
                    id="viable_route",
                    text="Která možnost ji může dostat na úřad včas i s kolem?",
                    aliases=["přívoz"],
                ),
                Question(
                    id="tunnel_possible",
                    text="Může použít tunel na kole?",
                    aliases=["ne", "nemůže"],
                    forbidden_terms=["ano"],
                ),
                Question(
                    id="taxi_on_time",
                    text="Stihne to taxi s držákem na kolo?",
                    aliases=["ne", "nestihne"],
                    forbidden_terms=["ano"],
                ),
            ],
        ),
        Scenario(
            id="recipe_allergy",
            task_type="constraint_satisfaction",
            context=(
                "Koláč musí být bez ořechů a bez laktózy. Původní recept používá mandlové mléko, "
                "které obsahuje ořechy. V lednici je kravské mléko s laktózou, ovesné mléko bez "
                "ořechů a bez laktózy a sójové mléko. Sójové mléko by šlo použít jen s citronem, "
                "ale citron doma není. Ovesné mléko lze v receptu použít v poměru jedna ku jedné."
            ),
            questions=[
                Question(
                    id="safe_substitute",
                    text="Které mléko je nejlepší bezpečná náhrada?",
                    aliases=["ovesné mléko", "ovesne mleko"],
                ),
                Question(
                    id="almond_allowed",
                    text="Je vhodné použít původní mandlové mléko?",
                    aliases=["ne", "není"],
                    forbidden_terms=["ano"],
                ),
                Question(
                    id="soy_possible",
                    text="Proč není sójové mléko praktická volba?",
                    aliases=["citron", "není"],
                    required_terms=["citron"],
                ),
            ],
        ),
        Scenario(
            id="library_fines",
            task_type="state_updates",
            context=(
                "Alena měla na účtu v knihovně pokutu 120 Kč. V pondělí zaplatila 50 Kč. "
                "V úterý jí přibyla nová pokuta 30 Kč za pozdní vrácení časopisu. Ve středu "
                "knihovna omylem připsala stejnou třicetikorunovou pokutu ještě jednou, ale "
                "ve čtvrtek omyl stornovala. V pátek Alena zaplatila dalších 40 Kč. Čtenář smí "
                "rezervovat nové knihy jen tehdy, když pokuta nepřesahuje 70 Kč."
            ),
            questions=[
                Question(
                    id="final_fine",
                    text="Jaká je konečná výše Aleniny pokuty?",
                    aliases=["60", "60 kč"],
                ),
                Question(
                    id="can_reserve",
                    text="Smí Alena rezervovat nové knihy?",
                    aliases=["ano"],
                    forbidden_terms=["ne"],
                ),
                Question(
                    id="duplicate_effect",
                    text="Má omylem připsaná druhá třicetikorunová pokuta vliv na konečný zůstatek?",
                    aliases=["ne", "nemá"],
                    forbidden_terms=["ano"],
                ),
            ],
        ),
        Scenario(
            id="server_failover",
            task_type="causal_chain",
            context=(
                "Primární server S1 přestal odpovídat v 02:10. Monitor po pěti minutách spouští "
                "failover, pokud server stále neodpovídá. S1 se obnovil až v 02:19. Záložní server S2 "
                "přebírá provoz dvě minuty po spuštění failoveru. Platební brána zaznamenala výpadek "
                "od 02:12 do 02:17. Audit říká, že pokud S2 převezme provoz nejpozději v 02:17, incident "
                "je pokryt automatickým failoverem."
            ),
            questions=[
                Question(
                    id="failover_start",
                    text="Kdy monitor spustil failover?",
                    aliases=["02:15", "2:15"],
                ),
                Question(
                    id="s2_takeover",
                    text="Kdy S2 převzal provoz?",
                    aliases=["02:17", "2:17"],
                ),
                Question(
                    id="covered_by_failover",
                    text="Je incident podle auditu pokryt automatickým failoverem?",
                    aliases=["ano"],
                    forbidden_terms=["ne"],
                ),
            ],
        ),
        Scenario(
            id="school_trip_budget",
            task_type="budget_constraints",
            context=(
                "Třída má rozpočet 12 000 Kč. Autobus stojí 7 500 Kč. Vstupné do muzea stojí "
                "120 Kč na žáka a jede 28 žáků. Pojištění stojí 900 Kč. Škola získala slevu "
                "1 000 Kč na autobus, ale sleva platí jen tehdy, když se zaplatí pojištění. "
                "Oběd za 2 200 Kč je volitelný a není nutný pro uskutečnění výletu."
            ),
            questions=[
                Question(
                    id="mandatory_cost",
                    text="Kolik stojí povinné položky po uplatnění slevy?",
                    aliases=["10 760", "10760"],
                ),
                Question(
                    id="with_lunch_possible",
                    text="Vejde se výlet do rozpočtu i s obědem?",
                    aliases=["ne", "nevejde"],
                    forbidden_terms=["ano"],
                ),
                Question(
                    id="without_lunch_possible",
                    text="Vejde se výlet do rozpočtu bez oběda?",
                    aliases=["ano"],
                    forbidden_terms=["ne"],
                ),
            ],
        ),
        Scenario(
            id="medical_triage",
            task_type="priority_rules",
            context=(
                "Pacient A čeká 18 minut a má střední bolest. Pacient B čeká 8 minut a má vysokou "
                "bolest. Pacient C čeká 25 minut a má nízkou bolest. Pravidlo říká, že vysoká bolest "
                "má přednost před délkou čekání. Pokud nikdo nemá vysokou bolest, vybere se pacient "
                "s čekáním nad 20 minut. Pokud je více takových pacientů, vybere se ten s vyšší bolestí."
            ),
            questions=[
                Question(
                    id="first_patient",
                    text="Který pacient má být vyšetřen jako první?",
                    aliases=["pacient b"],
                    forbidden_terms=["pacient a", "pacient c"],
                ),
                Question(
                    id="why_b",
                    text="Proč má pacient B přednost?",
                    aliases=["vysokou bolest", "vysoká bolest"],
                ),
                Question(
                    id="if_b_absent",
                    text="Kdo by měl přednost, kdyby pacient B nebyl přítomen?",
                    aliases=["pacient c"],
                    forbidden_terms=["pacient a"],
                ),
            ],
        ),
        Scenario(
            id="package_tracking",
            task_type="revision_conflict",
            context=(
                "Balík byl v 8:00 ve skladu Sever. V 9:10 byl naskenován na autě R12. "
                "V 9:40 systém omylem ukázal doručení na adresu A, ale kurýr později označil tento "
                "scan jako chybný. V 10:05 byl balík předán na výdejní místo B. Poslední platný scan "
                "má přednost před chybným údajem. Adresa A a výdejní místo B nejsou stejné místo."
            ),
            questions=[
                Question(
                    id="final_location",
                    text="Kde je balík podle posledního platného údaje?",
                    aliases=["výdejní místo b", "místě b"],
                    forbidden_terms=["adresa a"],
                ),
                Question(
                    id="delivered_a",
                    text="Byl balík platně doručen na adresu A?",
                    aliases=["ne", "nebyl"],
                    forbidden_terms=["ano"],
                ),
                Question(
                    id="why_ignore_a",
                    text="Proč se nemá brát jako platné doručení na adresu A?",
                    aliases=["chybný scan", "označil jako chybný", "chybný údaj"],
                ),
            ],
        ),
        Scenario(
            id="robot_boxes",
            task_type="nested_objects",
            context=(
                "Robot R položil čip do malé krabičky. Malou krabičku vložil do zeleného boxu. "
                "Zelený box přesunul z linky 1 na linku 2. Potom vyndal malou krabičku ze zeleného "
                "boxu a vložil ji do modrého boxu. Modrý box zůstal na lince 2. Zelený box se prázdný "
                "vrátil na linku 1."
            ),
            questions=[
                Question(
                    id="chip_final_box",
                    text="V kterém boxu je nakonec čip?",
                    aliases=["modrém boxu", "modrý box", "modre box"],
                    forbidden_terms=["zelený box"],
                ),
                Question(
                    id="chip_line",
                    text="Na které lince je nakonec čip?",
                    aliases=["lince 2", "linka 2"],
                ),
                Question(
                    id="green_contains_chip",
                    text="Obsahuje zelený box nakonec čip?",
                    aliases=["ne", "neobsahuje"],
                    forbidden_terms=["ano"],
                ),
            ],
        ),
        Scenario(
            id="contract_options",
            task_type="logic_constraints",
            context=(
                "Smlouva může být podepsána elektronicky jen tehdy, když má klient ověřenou identitu "
                "a částka je pod 500 000 Kč. Klient identitu ověřil včera. Částka je 620 000 Kč. "
                "Papírový podpis je povolen pro libovolnou částku, ale vyžaduje přítomnost notáře. "
                "Notář je dostupný zítra. Dnešní uzávěrka platí pouze pro elektronický podpis."
            ),
            questions=[
                Question(
                    id="electronic_allowed",
                    text="Je možné smlouvu podepsat elektronicky?",
                    aliases=["ne", "není"],
                    forbidden_terms=["ano"],
                ),
                Question(
                    id="blocking_condition",
                    text="Která podmínka blokuje elektronický podpis?",
                    aliases=["částka", "620 000", "nad 500 000"],
                ),
                Question(
                    id="paper_tomorrow",
                    text="Je papírový podpis zítra možný, pokud přijde notář?",
                    aliases=["ano"],
                    forbidden_terms=["ne"],
                ),
            ],
        ),
        Scenario(
            id="meeting_rooms",
            task_type="scheduling_constraints",
            context=(
                "Tým potřebuje místnost pro poradu od 13:00 do 14:00. Místnost Alfa je volná od "
                "12:30 do 13:30. Místnost Beta je volná od 13:15 do 14:30. Místnost Gama je volná "
                "od 13:00 do 14:00, ale nemá projektor. Porada vyžaduje projektor. Přenosný projektor "
                "je dostupný od 12:45 a lze ho použít v libovolné místnosti."
            ),
            questions=[
                Question(
                    id="room_without_portable",
                    text="Která místnost splňuje čas bez přenosného projektoru?",
                    aliases=["žádná", "zadna", "žádná místnost"],
                ),
                Question(
                    id="room_with_portable",
                    text="Která místnost splní podmínky s přenosným projektorem?",
                    aliases=["gama"],
                ),
                Question(
                    id="beta_time_fit",
                    text="Vyhovuje místnost Beta celému času porady?",
                    aliases=["ne", "nevyhovuje"],
                    forbidden_terms=["ano"],
                ),
            ],
        ),
        Scenario(
            id="subscription_billing",
            task_type="state_updates",
            context=(
                "Zákazník měl tarif Basic za 200 Kč měsíčně. Dne 10. dne měsíce přešel na Pro za "
                "500 Kč měsíčně. Měsíc má 30 dní a účtuje se poměrně podle počtu dní. Basic se účtuje "
                "za prvních 10 dní a Pro za zbývajících 20 dní. Zákazník měl kredit 100 Kč. Daň ani "
                "další poplatky se nepočítají."
            ),
            questions=[
                Question(
                    id="basic_part",
                    text="Kolik stojí část měsíce na tarifu Basic?",
                    aliases=["66,67", "66.67", "67"],
                ),
                Question(
                    id="pro_part",
                    text="Kolik stojí část měsíce na tarifu Pro?",
                    aliases=["333,33", "333.33", "333"],
                ),
                Question(
                    id="after_credit",
                    text="Kolik má zákazník zaplatit po odečtení kreditu?",
                    aliases=["300", "300 kč"],
                ),
            ],
        ),
        Scenario(
            id="exam_requirements",
            task_type="logic_constraints",
            context=(
                "Student splní kurz, pokud má alespoň 60 bodů celkem a zároveň alespoň 20 bodů ze "
                "závěrečné zkoušky. Domácí úkoly mu daly 38 bodů. Projekt mu dal 18 bodů. Závěrečná "
                "zkouška mu dala 19 bodů. Bonus 5 bodů se přičítá jen k celkovému součtu, ne ke "
                "zkouškovému minimu."
            ),
            questions=[
                Question(
                    id="total_points",
                    text="Kolik má student bodů celkem po bonusu?",
                    aliases=["80"],
                ),
                Question(
                    id="passes_course",
                    text="Splnil student kurz?",
                    aliases=["ne", "nesplnil"],
                    forbidden_terms=["ano"],
                ),
                Question(
                    id="failed_requirement",
                    text="Kterou podmínku student nesplnil?",
                    aliases=["20 bodů ze závěrečné zkoušky", "zkouškové minimum", "závěrečná zkouška"],
                ),
            ],
        ),
        Scenario(
            id="train_platform_change",
            task_type="revision_conflict",
            context=(
                "Ranní tabule ukázala, že vlak do Brna pojede z nástupiště 3. V 8:20 hlášení změnilo "
                "nástupiště na 5. V 8:25 aplikace stále ukazovala nástupiště 3, protože se nesynchronizovala. "
                "Stanice říká, že poslední živé hlášení má přednost před aplikací. Vlak odjíždí v 8:35."
            ),
            questions=[
                Question(
                    id="correct_platform",
                    text="Na které nástupiště má cestující jít?",
                    aliases=["5", "nástupiště 5"],
                    forbidden_terms=["nástupiště 3"],
                ),
                Question(
                    id="app_reliable",
                    text="Je údaj v aplikaci v 8:25 spolehlivý?",
                    aliases=["ne", "není"],
                    forbidden_terms=["ano"],
                ),
                Question(
                    id="why_platform_5",
                    text="Proč má přednost nástupiště 5?",
                    aliases=["poslední živé hlášení", "hlášení změnilo", "má přednost"],
                ),
            ],
        ),
        Scenario(
            id="energy_meter",
            task_type="numeric_constraints",
            context=(
                "Dům měl ráno na baterii 12 kWh. Solární panely během dne dodaly 8 kWh. "
                "Spotřeba domu byla 14 kWh. Večer se ještě prodaly 3 kWh do sítě. Baterie nesmí "
                "jít pod 2 kWh rezervy. Pokud by výpočet klesl pod rezervu, prodej do sítě se musí "
                "snížit tak, aby rezerva zůstala zachována."
            ),
            questions=[
                Question(
                    id="raw_remaining",
                    text="Kolik kWh by zůstalo po plánovaném prodeji bez ohledu na rezervu?",
                    aliases=["3"],
                ),
                Question(
                    id="reserve_ok",
                    text="Je plánovaný prodej 3 kWh slučitelný s rezervou 2 kWh?",
                    aliases=["ano"],
                    forbidden_terms=["ne"],
                ),
                Question(
                    id="final_battery",
                    text="Kolik kWh zůstane v baterii po prodeji?",
                    aliases=["3"],
                ),
            ],
        ),
        Scenario(
            id="garden_watering",
            task_type="priority_rules",
            context=(
                "Zahrada se má zalít, pokud dva dny nepršelo a vlhkost půdy je pod 35 %. "
                "Dnes ráno vlhkost byla 32 %. Včera nepršelo, ale předevčírem pršelo 4 mm. "
                "Předpověď na večer hlásí déšť 12 mm. Nouzové pravidlo říká, že pokud je předpovězen "
                "déšť alespoň 10 mm během 12 hodin, zalévání se odloží bez ohledu na ranní vlhkost."
            ),
            questions=[
                Question(
                    id="basic_condition",
                    text="Splňuje zahrada základní podmínku dvou dnů bez deště?",
                    aliases=["ne", "nesplňuje"],
                    forbidden_terms=["ano"],
                ),
                Question(
                    id="emergency_delay",
                    text="Aktivuje se nouzové pravidlo pro odložení zalévání?",
                    aliases=["ano"],
                    forbidden_terms=["ne"],
                ),
                Question(
                    id="water_today",
                    text="Má se zahrada dnes zalít?",
                    aliases=["ne", "nemá"],
                    forbidden_terms=["ano"],
                ),
            ],
        ),
    ]
    scenarios.extend(make_general_scenarios())
    return scenarios


def make_general_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []

    # These are deliberately broad, self-contained QA tasks rather than cases designed
    # around the thought graph machinery.
    for i in range(1, 11):
        start_a = 30 + i * 3
        start_b = 12 + i
        moved = 5 + (i % 4)
        damaged = 2 + (i % 3)
        added = 4 + (i % 5)
        final_a = start_a - moved + added
        final_b = start_b + moved - damaged
        scenarios.append(
            Scenario(
                id=f"general_inventory_{i:02d}",
                task_type="general_inventory",
                context=(
                    f"Ráno měl sklad A {start_a} kusů a sklad B {start_b} kusů. "
                    f"Ze skladu A se přesunulo {moved} kusů do skladu B. "
                    f"Ve skladu B se potom vyřadily {damaged} poškozené kusy. "
                    f"Do skladu A dorazila nová dodávka {added} kusů. "
                    "Přesun mezi sklady nemění celkový počet, vyřazení ano."
                ),
                questions=[
                    Question(
                        id="final_a",
                        text="Kolik kusů má nakonec sklad A?",
                        aliases=[str(final_a)],
                    ),
                    Question(
                        id="final_b",
                        text="Kolik kusů má nakonec sklad B?",
                        aliases=[str(final_b)],
                    ),
                    Question(
                        id="final_total",
                        text="Kolik kusů je nakonec celkem v obou skladech?",
                        aliases=[str(final_a + final_b)],
                    ),
                ],
            )
        )

    for i in range(1, 9):
        start_hour = 8 + (i % 3)
        start_min = 5 * i
        task1 = 20 + i
        gap = 5 + (i % 4)
        task2 = 25 + 2 * i
        deadline_hour = start_hour + 1
        deadline_min = 35 + (i % 3) * 5
        total_minutes = start_hour * 60 + start_min + task1 + gap + task2
        finish_hour, finish_min = divmod(total_minutes, 60)
        deadline_total = deadline_hour * 60 + deadline_min
        on_time = total_minutes <= deadline_total
        scenarios.append(
            Scenario(
                id=f"general_schedule_{i:02d}",
                task_type="general_schedule",
                context=(
                    f"Práce začne v {start_hour}:{start_min:02d}. První krok trvá {task1} minut. "
                    f"Po něm je povinná pauza {gap} minut. Druhý krok trvá {task2} minut. "
                    f"Výsledek musí být hotový nejpozději v {deadline_hour}:{deadline_min:02d}. "
                    "Kroky musí proběhnout v uvedeném pořadí."
                ),
                questions=[
                    Question(
                        id="finish_time",
                        text="Kdy práce skončí?",
                        aliases=[f"{finish_hour}:{finish_min:02d}"],
                    ),
                    Question(
                        id="on_time",
                        text="Stihne se termín?",
                        aliases=["ano" if on_time else "ne"],
                        forbidden_terms=["ne" if on_time else "ano"],
                    ),
                    Question(
                        id="total_duration",
                        text="Kolik minut zabere práce včetně pauzy?",
                        aliases=[str(task1 + gap + task2)],
                    ),
                ],
            )
        )

    for i in range(1, 9):
        limit = 100 + i * 10
        price = 70 + i * 12
        member = i % 2 == 0
        verified = i % 3 != 0
        allowed = price <= limit and member and verified
        missing: list[str] = []
        if price > limit:
            missing.append("cena")
        if not member:
            missing.append("členství")
        if not verified:
            missing.append("ověření")
        scenarios.append(
            Scenario(
                id=f"general_rule_{i:02d}",
                task_type="general_rule",
                context=(
                    f"Nákup je povolen jen tehdy, když cena nepřekročí {limit} Kč, "
                    "uživatel má aktivní členství a jeho účet je ověřený. "
                    f"Aktuální cena je {price} Kč. "
                    f"Členství je {'aktivní' if member else 'neaktivní'}. "
                    f"Účet je {'ověřený' if verified else 'neověřený'}."
                ),
                questions=[
                    Question(
                        id="allowed",
                        text="Je nákup povolen?",
                        aliases=["ano" if allowed else "ne"],
                        forbidden_terms=["ne" if allowed else "ano"],
                    ),
                    Question(
                        id="price_ok",
                        text="Splňuje cena limit?",
                        aliases=["ano" if price <= limit else "ne"],
                        forbidden_terms=["ne" if price <= limit else "ano"],
                    ),
                    Question(
                        id="missing_condition",
                        text="Která podmínka chybí, pokud nákup není povolen?",
                        aliases=missing or ["žádná", "nic nechybí"],
                    ),
                ],
            )
        )

    for i in range(1, 7):
        original = ["stůl", "šuplík", "police", "taška", "box", "skříň"][i - 1]
        final = ["batoh", "krabice", "zásuvka", "kufr", "sejf", "regál"][i - 1]
        actor = ["Eva", "Petr", "Jana", "Marek", "Lucie", "Tomáš"][i - 1]
        observer = ["Adam", "Nina", "Filip", "Olga", "Karel", "Irena"][i - 1]
        saw = i % 2 == 0
        scenarios.append(
            Scenario(
                id=f"general_belief_{i:02d}",
                task_type="general_belief",
                context=(
                    f"Dokument byl ráno uložen na místě {original}. "
                    f"{actor} přesunul dokument na místo {final}. "
                    f"{observer} {'viděl' if saw else 'neviděl'}, že {actor} dokument přesunul. "
                    f"{actor} ví, že dokument je nyní na místě {final}."
                ),
                questions=[
                    Question(
                        id="real_location",
                        text="Kde je dokument ve skutečnosti?",
                        aliases=[final],
                    ),
                    Question(
                        id="observer_belief",
                        text=f"Kde bude {observer} pravděpodobně hledat dokument?",
                        aliases=[final if saw else original],
                    ),
                    Question(
                        id="observer_saw_move",
                        text=f"Viděl {observer} přesun dokumentu?",
                        aliases=["ano" if saw else "ne"],
                        forbidden_terms=["ne" if saw else "ano"],
                    ),
                ],
            )
        )

    return scenarios


def make_large_general_scenarios(target_questions: int = 500) -> list[Scenario]:
    per_family = max(1, target_questions // 25)
    scenarios: list[Scenario] = []
    scenarios.extend(make_large_inventory_scenarios(per_family))
    scenarios.extend(make_large_schedule_scenarios(per_family))
    scenarios.extend(make_large_rule_scenarios(per_family))
    scenarios.extend(make_large_route_scenarios(per_family))
    scenarios.extend(make_large_belief_scenarios(per_family))
    return scenarios[: max(1, target_questions // 5)]


def make_large_inventory_scenarios(n: int) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for i in range(1, n + 1):
        a0 = 40 + 2 * i
        b0 = 25 + (3 * i) % 17
        c0 = 15 + (5 * i) % 13
        ab = 3 + i % 6
        bc = 2 + (i * 2) % 5
        damaged_c = 1 + i % 4
        add_a = 4 + i % 7
        a = a0 - ab + add_a
        b = b0 + ab - bc
        c = c0 + bc - damaged_c
        total = a + b + c
        largest_name = max([("sklad A", a), ("sklad B", b), ("sklad C", c)], key=lambda item: item[1])[0]
        threshold = total - (3 if i % 2 else -3)
        above = total > threshold
        scenarios.append(
            Scenario(
                id=f"large_inventory_{i:03d}",
                task_type="large_inventory",
                context=(
                    f"Sklad A měl ráno {a0} kusů, sklad B {b0} kusů a sklad C {c0} kusů. "
                    f"Ze skladu A se přesunulo {ab} kusů do skladu B. "
                    f"Ze skladu B se přesunulo {bc} kusů do skladu C. "
                    f"Ve skladu C se vyřadily {damaged_c} poškozené kusy. "
                    f"Do skladu A dorazila dodávka {add_a} kusů. "
                    "Přesuny mezi sklady nemění celkový počet, vyřazení ano."
                ),
                questions=[
                    Question("final_a", "Kolik kusů má nakonec sklad A?", [str(a)]),
                    Question("final_b", "Kolik kusů má nakonec sklad B?", [str(b)]),
                    Question("final_total", "Kolik kusů je nakonec celkem ve všech skladech?", [str(total)]),
                    Question("largest", "Který sklad má nakonec nejvíce kusů?", [largest_name]),
                    Question(
                        "above_threshold",
                        f"Je konečný celkový počet vyšší než {threshold}?",
                        ["ano" if above else "ne"],
                        forbidden_terms=["ne" if above else "ano"],
                    ),
                ],
            )
        )
    return scenarios


def make_large_schedule_scenarios(n: int) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for i in range(1, n + 1):
        start_h = 7 + i % 5
        start_m = (7 * i) % 50
        prep = 12 + i % 12
        review = 18 + (2 * i) % 17
        wait = 4 + i % 6
        export = 16 + (3 * i) % 19
        total_duration = prep + review + wait + export
        finish_total = start_h * 60 + start_m + total_duration
        finish_h, finish_m = divmod(finish_total, 60)
        deadline_total = finish_total + (8 if i % 3 else -6)
        deadline_h, deadline_m = divmod(deadline_total, 60)
        on_time = finish_total <= deadline_total
        longest = max([("příprava", prep), ("revize", review), ("export", export)], key=lambda item: item[1])[0]
        slack = deadline_total - finish_total
        scenarios.append(
            Scenario(
                id=f"large_schedule_{i:03d}",
                task_type="large_schedule",
                context=(
                    f"Proces začne v {start_h}:{start_m:02d}. Příprava trvá {prep} minut. "
                    f"Revize začíná po přípravě a trvá {review} minut. "
                    f"Po revizi je čekání {wait} minut. Export trvá {export} minut. "
                    f"Výsledek musí být hotový nejpozději v {deadline_h}:{deadline_m:02d}. "
                    "Kroky se nesmí překrývat."
                ),
                questions=[
                    Question("finish_time", "Kdy proces skončí?", [f"{finish_h}:{finish_m:02d}"]),
                    Question("on_time", "Stihne se deadline?", ["ano" if on_time else "ne"], forbidden_terms=["ne" if on_time else "ano"]),
                    Question("duration", "Kolik minut proces trvá celkem?", [str(total_duration)]),
                    Question("longest_step", "Který hlavní krok trvá nejdéle?", [longest]),
                    Question("slack", "Kolik minut je rezerva vůči deadlinu? Pokud je záporná, uveď zpoždění.", [str(abs(slack))]),
                ],
            )
        )
    return scenarios


def make_large_rule_scenarios(n: int) -> list[Scenario]:
    scenarios: list[Scenario] = []
    missing_cycle = ["cena", "členství", "ověření", "žádná", "žádná"]
    for i in range(1, n + 1):
        limit = 120 + 5 * i
        missing = missing_cycle[(i - 1) % len(missing_cycle)]
        price = limit + 11 if missing == "cena" else limit - 9
        member = missing != "členství"
        verified = missing != "ověření"
        allowed = missing == "žádná"
        scenarios.append(
            Scenario(
                id=f"large_rule_{i:03d}",
                task_type="large_rule",
                context=(
                    f"Žádost je schválena jen tehdy, když částka nepřekročí {limit} Kč, "
                    "žadatel má aktivní členství a účet je ověřený. "
                    f"Aktuální částka je {price} Kč. "
                    f"Členství je {'aktivní' if member else 'neaktivní'}. "
                    f"Účet je {'ověřený' if verified else 'neověřený'}."
                ),
                questions=[
                    Question("allowed", "Je žádost schválena?", ["ano" if allowed else "ne"], forbidden_terms=["ne" if allowed else "ano"]),
                    Question("price_ok", "Splňuje částka limit?", ["ano" if price <= limit else "ne"], forbidden_terms=["ne" if price <= limit else "ano"]),
                    Question("member_ok", "Je splněna podmínka členství?", ["ano" if member else "ne"], forbidden_terms=["ne" if member else "ano"]),
                    Question("verified_ok", "Je splněna podmínka ověření účtu?", ["ano" if verified else "ne"], forbidden_terms=["ne" if verified else "ano"]),
                    Question("missing_condition", "Která podmínka chybí?", [missing, "nic nechybí"] if missing == "žádná" else [missing]),
                ],
            )
        )
    return scenarios


def make_large_route_scenarios(n: int) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for i in range(1, n + 1):
        deadline_h = 10 + i % 4
        deadline_m = (5 * i) % 50
        deadline = deadline_h * 60 + deadline_m
        walk = deadline + (8 if i % 2 else -4)
        bus = deadline - (6 + i % 5)
        taxi = deadline - (12 + i % 4)
        bus_ticket = i % 3 != 0
        taxi_allowed = i % 4 != 0
        valid: list[tuple[str, int]] = []
        if walk <= deadline:
            valid.append(("pěšky", walk))
        if bus <= deadline and bus_ticket:
            valid.append(("autobus", bus))
        if taxi <= deadline and taxi_allowed:
            valid.append(("taxi", taxi))
        best = min(valid, key=lambda item: item[1])[0] if valid else "žádná"
        best_time = min(valid, key=lambda item: item[1])[1] if valid else deadline
        scenarios.append(
            Scenario(
                id=f"large_route_{i:03d}",
                task_type="large_route",
                context=(
                    f"Cíl je potřeba stihnout nejpozději v {deadline_h}:{deadline_m:02d}. "
                    f"Pěší cesta dorazí v {walk // 60}:{walk % 60:02d}. "
                    f"Autobus dorazí v {bus // 60}:{bus % 60:02d}, ale lístek je {'dostupný' if bus_ticket else 'vyprodaný'}. "
                    f"Taxi dorazí v {taxi // 60}:{taxi % 60:02d}, ale pravidla taxi {'povolují' if taxi_allowed else 'zakazují'}. "
                    "Vybraná možnost musí dorazit včas a splnit své omezení."
                ),
                questions=[
                    Question("best_option", "Která možnost je nejlepší platná volba?", [best]),
                    Question("bus_valid", "Je autobus platná možnost?", ["ano" if bus <= deadline and bus_ticket else "ne"], forbidden_terms=["ne" if bus <= deadline and bus_ticket else "ano"]),
                    Question("taxi_valid", "Je taxi platná možnost?", ["ano" if taxi <= deadline and taxi_allowed else "ne"], forbidden_terms=["ne" if taxi <= deadline and taxi_allowed else "ano"]),
                    Question("best_arrival", "Kdy dorazí nejlepší platná volba?", [f"{best_time // 60}:{best_time % 60:02d}"]),
                    Question("walk_on_time", "Dorazí pěší cesta včas?", ["ano" if walk <= deadline else "ne"], forbidden_terms=["ne" if walk <= deadline else "ano"]),
                ],
            )
        )
    return scenarios


def make_large_belief_scenarios(n: int) -> list[Scenario]:
    starts = ["stůl", "šuplík", "police", "taška", "box", "skříň", "kufr", "regál"]
    finals = ["batoh", "krabice", "zásuvka", "sejf", "obálka", "modrý box", "archiv", "sklad"]
    observers = ["Adam", "Nina", "Filip", "Olga", "Karel", "Irena", "Marta", "Pavel"]
    movers = ["Eva", "Petr", "Jana", "Marek", "Lucie", "Tomáš", "Sára", "David"]
    scenarios: list[Scenario] = []
    for i in range(1, n + 1):
        start = starts[(i - 1) % len(starts)]
        final = finals[(i - 1) % len(finals)]
        observer = observers[(i - 1) % len(observers)]
        mover = movers[(i - 1) % len(movers)]
        saw = i % 2 == 0
        belief = final if saw else start
        scenarios.append(
            Scenario(
                id=f"large_belief_{i:03d}",
                task_type="large_belief",
                context=(
                    f"Balíček byl ráno na místě {start}. {mover} přesunul balíček na místo {final}. "
                    f"{observer} {'viděl' if saw else 'neviděl'}, že {mover} balíček přesunul. "
                    f"{mover} ví, že balíček je na místě {final}. Nikdo další balíček nepřesunul."
                ),
                questions=[
                    Question("real_location", "Kde je balíček ve skutečnosti?", [final]),
                    Question("observer_belief", f"Kde bude {observer} pravděpodobně hledat balíček?", [belief]),
                    Question("observer_saw", f"Viděl {observer} přesun?", ["ano" if saw else "ne"], forbidden_terms=["ne" if saw else "ano"]),
                    Question("original_location", "Kde byl balíček ráno?", [start]),
                    Question("mover_knows", f"Ví {mover}, kde balíček je?", ["ano"], required_terms=[final], forbidden_terms=["ne"]),
                ],
            )
        )
    return scenarios


def atomizer_prompt(context: str, max_atoms: int) -> str:
    return f"""
Jsi Atomizer pro textový graf myšlenek. Vytvoř základní uzly pouze z kontextu.
Neznáš žádnou otázku a nesmíš skrytě odpovídat na konkrétní dotaz.

Cíl:
- zachovej přesné časy, počty, negace, výjimky, podmínky a opravy informací;
- rozlož složené věty na samostatné atomické myšlenky;
- explicitně označ zdroj jako "věta N";
- nepřidávej závěry, které vyžadují kombinaci více atomů.

Vrať maximálně {max_atoms} uzlů a pouze JSON:
{{
  "nodes": [
    {{
      "id": "T1",
      "level": 1,
      "thought": "...",
      "parents": [],
      "source": ["věta 1"],
      "operation": "atom",
      "support": "entailed",
      "utility": 0.7
    }}
  ]
}}

Kontext:
{context}
""".strip()


def composer_prompt(nodes: list[dict[str, Any]], level: int, max_new_nodes: int) -> str:
    return f"""
Jsi Composer pro rekurzivní textový graf myšlenek. Dostáváš pouze existující uzly,
ne původní kontext a ne otázku.

Vytvoř maximálně {max_new_nodes} nové uzly úrovně {level}.
Nový uzel přijmi jen pokud:
- má 1 až 4 existující rodiče;
- kombinuje rodiče do nového užitečného významu, který v žádném rodiči samostatně není;
- je dohledatelný z rodičů a neobsahuje nový fakt zvenku;
- pomáhá pro časové pořadí, stav po změnách, příčinu, omezení, plán, opravu konfliktu,
  přesvědčení postav, početní součet nebo vyloučení možnosti;
- není duplicitou existujícího uzlu.

U časů a počtů proveď výpočet explicitně v poli thought. U konfliktů respektuj pozdější
opravy a pravidla přednosti. U false-belief odděl realitu od přesvědčení postavy.

Vrať pouze JSON:
{{
  "nodes": [
    {{
      "id": "H{level}_1",
      "level": {level},
      "thought": "...",
      "parents": ["T1", "T2"],
      "source": [],
      "operation": "constraint_combination",
      "support": "entailed",
      "utility": 0.85
    }}
  ]
}}

Existující uzly:
{json.dumps(nodes, ensure_ascii=False, indent=2)}
""".strip()


def verifier_prompt(
    existing_nodes: list[dict[str, Any]], candidates: list[dict[str, Any]], max_accept: int
) -> str:
    return f"""
Jsi Verifier textových myšlenkových uzlů. Zkontroluj kandidáty proti existujícím uzlům.

Přijmi kandidáta pouze pokud:
- všichni rodiče existují;
- thought skutečně vyplývá z rodičů;
- není duplicitní;
- nepřidává nedoložený fakt;
- u časů a počtů je výpočet správný;
- přidává význam užitečný pro budoucí otázky.

Vrať maximálně {max_accept} nejlepších kandidátů a pouze JSON:
{{
  "accepted_ids": ["H2_1"],
  "rejected": [
    {{"id": "H2_2", "reason": "duplikát nebo nedoložený závěr"}}
  ]
}}

Existující uzly:
{json.dumps(existing_nodes, ensure_ascii=False, indent=2)}

Kandidáti:
{json.dumps(candidates, ensure_ascii=False, indent=2)}
""".strip()


def summary_prompt(context: str) -> str:
    return f"""
Vytvoř stručné věcné shrnutí kontextu bez znalosti budoucí otázky.
Zachovej klíčové časy, počty, negace, podmínky a opravy informací.
Vrať pouze JSON: {{"summary": "..."}}

Kontext:
{context}
""".strip()


def direct_answer_prompt(context: str, question: str, reasoning: bool) -> str:
    instruction = (
        "Odpověz a přidej jednu stručnou větu s důvodem."
        if reasoning
        else "Odpověz co nejkratší věcnou odpovědí."
    )
    return f"""
{instruction}
U ano/ne otázky začni slovem Ano nebo Ne.
Vrať pouze JSON:
{{"answer": "...", "rationale": "...", "confidence": 0.0}}

Kontext:
{context}

Otázka:
{question}
""".strip()


def graph_answer_prompt(question: str, nodes: list[dict[str, Any]], variant: str) -> str:
    return f"""
Jsi Answerer. Nedostáváš původní kontext, pouze vybrané myšlenkové uzly.
Odpověz jen z těchto uzlů a z jejich kombinace. Nepoužívej nedoložené domněnky.
U ano/ne otázky začni slovem Ano nebo Ne. Pokud detail v uzlech chybí, řekni to.

Varianta: {variant}

Vrať pouze JSON:
{{"answer": "...", "rationale": "...", "used_nodes": ["T1"], "confidence": 0.0}}

Otázka:
{question}

Myšlenkové uzly:
{json.dumps(nodes, ensure_ascii=False, indent=2)}
""".strip()


def loop_answer_prompt(question: str, nodes: list[dict[str, Any]]) -> str:
    return f"""
Jsi Answerer. Nedostáváš původní kontext, pouze omezený výběr myšlenkového grafu.
Buď odpověz, nebo řekni, jaké cílené odvozené myšlenky z existujících uzlů chybí.
Nepožaduj původní text. Požaduj pouze kombinace, srovnání nebo výpočty nad uzly.

Vrať pouze JSON jedním z tvarů:
{{"status": "answered", "answer": "...", "rationale": "...", "used_nodes": ["H1"], "confidence": 0.0}}
{{"status": "needs_more_thoughts", "missing": ["..."], "rationale": "..."}}

Otázka:
{question}

Myšlenkové uzly:
{json.dumps(nodes, ensure_ascii=False, indent=2)}
""".strip()


def targeted_expand_prompt(nodes: list[dict[str, Any]], missing: list[str], max_new_nodes: int) -> str:
    return f"""
Jsi Composer. Nedostáváš původní kontext, pouze existující uzly a seznam mezer.
Vytvoř maximálně {max_new_nodes} nové uzly, které cíleně kombinují existující uzly
okolo uvedených mezer. Nevymýšlej nové fakty mimo rodiče.

Vrať pouze JSON:
{{"nodes": [
  {{"id": "L1", "level": 4, "thought": "...", "parents": ["T1"], "source": [],
   "operation": "targeted_composition", "support": "entailed", "utility": 0.8}}
]}}

Chybějící myšlenky:
{json.dumps(missing, ensure_ascii=False, indent=2)}

Existující uzly:
{json.dumps(nodes, ensure_ascii=False, indent=2)}
""".strip()


def summary_answer_prompt(summary: str, question: str) -> str:
    return f"""
Odpověz pouze ze shrnutí, nikoli z původního kontextu.
U ano/ne otázky začni slovem Ano nebo Ne.
Vrať pouze JSON:
{{"answer": "...", "rationale": "...", "confidence": 0.0}}

Shrnutí:
{summary}

Otázka:
{question}
""".strip()


def question_slots_prompt(question: str) -> str:
    return f"""
Rozlož otázku na malé ověřitelné sloty, které musí být vyřešeny před odpovědí.
Nevytvářej odpověď. Nevytvářej volné úvahy.

Dbej na typ otázky:
- Pokud se otázka ptá "kdy", "do kdy", "kolik", "kde" nebo "který", slot musí hledat přesnou hodnotu.
- Nepřepisuj hodnotovou otázku na ano/ne otázku.
- Ano/ne slot používej jen pro skutečnou ano/ne otázku.

Vrať pouze JSON:
{{
  "slots": [
    {{"id": "S1", "need": "...", "kind": "temporal_comparison|count_update|rule_application|state_lookup|constraint_check|conflict_resolution|final_decision"}}
  ]
}}

Otázka:
{question}
""".strip()


def task_graph_prompt(
    question: str,
    slots: list[dict[str, Any]],
    base_nodes: list[dict[str, Any]],
    max_nodes: int,
) -> str:
    return f"""
Jsi task-specific Thought VM. Máš obecné myšlenkové uzly a konkrétní otázku.
Vytvoř inferenční graf Q_* uzlů pro tuto otázku. Toto není chain-of-thought:
nepiš odstavce ani narativní postup, pouze adresovatelné uzly.

Pravidla:
- Každý Q uzel musí mít parents: 1 až 5 existujících T/H/Q uzlů.
- Q uzly mohou odkazovat na dřívější Q uzly, takže vzniknou 2 až 4 úrovně.
- Každý Q uzel musí řešit jeden nebo více slotů v poli solves.
- Vyšší Q uzel musí kombinovat nižší uzly do nového významu.
- Nepřidávej fakta mimo rodiče.
- Poslední uzel by měl řešit finální rozhodnutí otázky.
- U časů, počtů a podmínek uveď výpočet přímo v poli thought.
- Pokud otázka žádá přesnou hodnotu ("do kdy", "kolik", "kde", "který"),
  finální Q uzel musí obsahovat tuto hodnotu, ne pouze posouzení proveditelnosti.

Vrať maximálně {max_nodes} uzlů a pouze JSON:
{{
  "nodes": [
    {{
      "id": "Q1",
      "level": 1,
      "thought": "...",
      "parents": ["T1", "H2_1"],
      "solves": ["S1"],
      "operation": "rule_application",
      "support": "entailed",
      "utility": 0.9
    }}
  ]
}}

Otázka:
{question}

Sloty:
{json.dumps(slots, ensure_ascii=False, indent=2)}

Dostupné myšlenkové uzly:
{json.dumps(base_nodes, ensure_ascii=False, indent=2)}
""".strip()


def task_no_parents_prompt(
    question: str,
    slots: list[dict[str, Any]],
    base_nodes: list[dict[str, Any]],
    max_nodes: int,
) -> str:
    return f"""
Vytvoř pro otázku task-specific mezizávěry, ale BEZ rodičovských odkazů.
Toto je kontrolní varianta: smíš psát krátké strukturované body, ale nesmíš používat parents
ani odkazovat na ID uzlů jako oporu.

Vrať pouze JSON:
{{
  "nodes": [
    {{"id": "NP1", "thought": "...", "solves": ["S1"], "operation": "..."}}
  ],
  "answer": "...",
  "rationale": "..."
}}

Otázka:
{question}

Sloty:
{json.dumps(slots, ensure_ascii=False, indent=2)}

Dostupné myšlenkové uzly:
{json.dumps(base_nodes, ensure_ascii=False, indent=2)}
""".strip()


def task_cot_json_prompt(question: str, base_nodes: list[dict[str, Any]]) -> str:
    return f"""
Odpověz na otázku pomocí krátkého JSON chain-of-thought baseline.
Smíš použít dostupné myšlenkové uzly, ale nevytvářej graf, parents ani adresovatelné Q uzly.
U ano/ne otázky začni odpověď slovem Ano nebo Ne.

Vrať pouze JSON:
{{"steps": ["..."], "answer": "...", "rationale": "...", "confidence": 0.0}}

Otázka:
{question}

Dostupné myšlenkové uzly:
{json.dumps(base_nodes, ensure_ascii=False, indent=2)}
""".strip()


def task_graph_answer_prompt(
    question: str,
    slots: list[dict[str, Any]],
    base_nodes: list[dict[str, Any]],
    task_nodes: list[dict[str, Any]],
) -> str:
    return f"""
Jsi Resolver. Nedostáváš původní kontext. Odpověz pouze z task-specific Q uzlů
a případně z jejich rodičů. Nevytvářej nové úvahy mimo uzly.
U ano/ne otázky začni odpověď slovem Ano nebo Ne, ale neodpovídej jen holým Ano/Ne:
přidej rozhodující podmínku, čas, počet nebo místo, které otázka požaduje.
U otázek "kdy/do kdy/kolik/kde/který" vrať přesnou hodnotu, ne odpověď o tom,
zda je něco možné.

Vrať pouze JSON:
{{"answer": "...", "rationale": "...", "used_nodes": ["Q1"], "resolved_slots": ["S1"], "confidence": 0.0}}

Otázka:
{question}

Sloty:
{json.dumps(slots, ensure_ascii=False, indent=2)}

Base uzly:
{json.dumps(base_nodes, ensure_ascii=False, indent=2)}

Task Q uzly:
{json.dumps(task_nodes, ensure_ascii=False, indent=2)}
""".strip()


def task_graph_flat_q_answer_prompt(
    question: str,
    slots: list[dict[str, Any]],
    base_nodes: list[dict[str, Any]],
    task_nodes: list[dict[str, Any]],
    label: str,
) -> str:
    return f"""
Jsi Resolver pro kontrolní variantu {label}. Nedostáváš původní kontext.
Dostáváš task-specific Q mezizávěry jako obyčejný seznam bez spolehlivé grafové struktury.
Odpověz pouze z těchto mezizávěrů a dostupných base uzlů.
U ano/ne otázky začni odpověď slovem Ano nebo Ne.

Vrať pouze JSON:
{{"answer": "...", "rationale": "...", "used_nodes": ["Q1"], "confidence": 0.0}}

Otázka:
{question}

Sloty:
{json.dumps(slots, ensure_ascii=False, indent=2)}

Base uzly:
{json.dumps(base_nodes, ensure_ascii=False, indent=2)}

Q mezizávěry:
{json.dumps(task_nodes, ensure_ascii=False, indent=2)}
""".strip()


def clamp_float(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def clean_nodes(
    graph: Any,
    prefix: str,
    forced_level: int | None = None,
    existing_ids: set[str] | None = None,
    max_nodes: int | None = None,
) -> list[dict[str, Any]]:
    raw_nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    existing_ids = existing_ids or set()

    for i, node in enumerate(raw_nodes, start=1):
        if not isinstance(node, dict):
            continue
        thought = str(node.get("thought") or "").strip()
        if not thought:
            continue
        node_id = f"{prefix}{len(nodes) + 1}"
        if node_id in seen or node_id in existing_ids:
            node_id = f"{prefix}{i}_{len(nodes) + 1}"
        seen.add(node_id)

        parents = node.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        parent_ids = [str(parent) for parent in parents if str(parent) in existing_ids]
        level = forced_level if forced_level is not None else int(node.get("level") or 1)
        if level > 1 and not parent_ids:
            continue

        nodes.append(
            {
                "id": node_id,
                "level": level,
                "thought": thought,
                "parents": parent_ids if level > 1 else [],
                "source": node.get("source") if isinstance(node.get("source"), list) else [],
                "operation": str(node.get("operation") or ("atom" if level == 1 else "composition")),
                "support": str(node.get("support") or "probable"),
                "utility": clamp_float(node.get("utility"), 0.5),
            }
        )
        if max_nodes is not None and len(nodes) >= max_nodes:
            break
    return dedupe_nodes(nodes)


def dedupe_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for node in nodes:
        normalized = normalize_text(str(node.get("thought", "")))
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        deduped.append(node)
    return deduped


def clean_task_nodes(
    graph: Any,
    base_ids: set[str],
    max_nodes: int,
) -> list[dict[str, Any]]:
    raw_nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    nodes: list[dict[str, Any]] = []
    available_ids = set(base_ids)
    seen_texts: set[str] = set()

    for i, node in enumerate(raw_nodes, start=1):
        if not isinstance(node, dict):
            continue
        thought = str(node.get("thought") or "").strip()
        if not thought:
            continue
        normalized = normalize_text(thought)
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)

        parents = node.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        parent_ids = [str(parent) for parent in parents if str(parent) in available_ids]
        if not parent_ids:
            continue

        solves = node.get("solves") or []
        if not isinstance(solves, list):
            solves = []

        node_id = f"Q{len(nodes) + 1}"
        cleaned = {
            "id": node_id,
            "level": int(node.get("level") or min(4, 1 + max((1 for _ in parent_ids), default=0))),
            "thought": thought,
            "parents": parent_ids,
            "solves": [str(item) for item in solves],
            "operation": str(node.get("operation") or "task_composition"),
            "support": str(node.get("support") or "probable"),
            "utility": clamp_float(node.get("utility"), 0.7),
        }
        nodes.append(cleaned)
        available_ids.add(node_id)
        if len(nodes) >= max_nodes:
            break
    return nodes


def get_question_slots(client: GeminiJsonClient, question: str) -> list[dict[str, Any]]:
    raw_slots = client.json_call(
        "question_slots",
        question_slots_prompt(question),
        max_output_tokens=768,
    )
    slots = raw_slots.get("slots", []) if isinstance(raw_slots, dict) else []
    cleaned: list[dict[str, Any]] = []
    for i, slot in enumerate(slots, start=1):
        if not isinstance(slot, dict):
            continue
        need = str(slot.get("need") or "").strip()
        if not need:
            continue
        cleaned.append(
            {
                "id": str(slot.get("id") or f"S{i}"),
                "need": need,
                "kind": str(slot.get("kind") or "constraint_check"),
            }
        )
    if not cleaned:
        cleaned.append({"id": "S1", "need": question, "kind": "final_decision"})
    return cleaned[:6]


def build_task_graph_nodes(
    client: GeminiJsonClient,
    question: Question,
    slots: list[dict[str, Any]],
    base_nodes: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[Any, list[dict[str, Any]]]:
    raw_graph = client.json_call(
        "task_graph",
        task_graph_prompt(question.text, slots, base_nodes, args.max_task_nodes),
        max_output_tokens=args.graph_output_tokens,
    )
    task_nodes = clean_task_nodes(
        raw_graph,
        base_ids={node["id"] for node in base_nodes},
        max_nodes=args.max_task_nodes,
    )
    return raw_graph, task_nodes


def run_task_graph_variant(
    client: GeminiJsonClient,
    question: Question,
    slots: list[dict[str, Any]],
    base_nodes: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_graph, task_nodes = build_task_graph_nodes(client, question, slots, base_nodes, args)
    result = client.json_call(
        "answer_task_graph",
        task_graph_answer_prompt(question.text, slots, base_nodes, task_nodes),
        max_output_tokens=768,
    )
    if isinstance(result, dict):
        result["_task_slots"] = slots
        result["_task_nodes"] = task_nodes
        result["_raw_task_graph"] = raw_graph
    return result if isinstance(result, dict) else {}, task_nodes


def strip_task_node_structure(task_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": node.get("id"),
            "thought": node.get("thought"),
            "solves": node.get("solves", []),
            "operation": node.get("operation", "unknown"),
        }
        for node in task_nodes
    ]


def shuffled_task_node_parents(task_nodes: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    if len(task_nodes) < 2:
        return [dict(node) for node in task_nodes]
    rng = random.Random(seed)
    parent_lists = [list(node.get("parents", [])) for node in task_nodes]
    rng.shuffle(parent_lists)
    shuffled: list[dict[str, Any]] = []
    for node, parents in zip(task_nodes, parent_lists):
        copied = dict(node)
        copied["parents"] = parents
        copied["_parents_shuffled"] = True
        shuffled.append(copied)
    return shuffled


def verify_candidates(
    client: GeminiJsonClient,
    existing_nodes: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    max_accept: int,
    enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        return [], {"accepted_ids": [], "rejected": []}
    if not enabled:
        return candidates[:max_accept], {"accepted_ids": [node["id"] for node in candidates[:max_accept]]}

    result = client.json_call(
        "verifier",
        verifier_prompt(existing_nodes, candidates, max_accept),
        max_output_tokens=1536,
    )
    accepted_ids = result.get("accepted_ids", []) if isinstance(result, dict) else []
    accepted_set = {str(node_id) for node_id in accepted_ids}
    accepted = [node for node in candidates if node["id"] in accepted_set]
    if not accepted:
        fallback = sorted(candidates, key=lambda node: float(node.get("utility", 0.0)), reverse=True)
        accepted = fallback[: max(1, min(max_accept, len(fallback)))]
        if isinstance(result, dict):
            result["_fallback_used"] = True
    return accepted[:max_accept], result if isinstance(result, dict) else {}


def build_graph(
    client: GeminiJsonClient,
    scenario: Scenario,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace: dict[str, Any] = {"atomizer": None, "composer": [], "verifier": []}
    atomized = client.json_call(
        "atomizer",
        atomizer_prompt(scenario.context, args.max_atoms),
        max_output_tokens=args.graph_output_tokens,
    )
    nodes = clean_nodes(atomized, prefix="T", forced_level=1, max_nodes=args.max_atoms)
    trace["atomizer"] = atomized

    for level in range(2, args.max_depth + 1):
        remaining = args.max_nodes - len(nodes)
        if remaining <= 0:
            break
        max_new = min(args.max_new_per_level, remaining)
        raw_candidates = client.json_call(
            f"composer_l{level}",
            composer_prompt(nodes, level, max_new),
            max_output_tokens=args.graph_output_tokens,
        )
        candidates = clean_nodes(
            raw_candidates,
            prefix=f"H{level}_",
            forced_level=level,
            existing_ids={node["id"] for node in nodes},
            max_nodes=max_new * 2,
        )
        accepted, verifier_result = verify_candidates(
            client,
            nodes,
            candidates,
            max_new,
            enabled=args.verify_compositions,
        )
        trace["composer"].append({"level": level, "raw": raw_candidates, "candidates": candidates})
        trace["verifier"].append({"level": level, "result": verifier_result})
        if not accepted:
            break
        nodes.extend(accepted)
    return nodes[: args.max_nodes], trace


def token_set(text: str) -> set[str]:
    terms: set[str] = set()
    for token in normalize_text(text).split():
        if not token or token in STOPWORDS:
            continue
        terms.add(token)
        stem = stem_token(token)
        if stem and stem not in STOPWORDS:
            terms.add(stem)
        if len(token) >= 6:
            terms.add(token[:5])
    return terms


def stem_token(token: str) -> str:
    if re.fullmatch(r"\d{1,2}:\d{2}|\d+", token):
        return token
    for suffix in (
        "ymi",
        "ach",
        "ami",
        "emi",
        "eho",
        "emu",
        "ich",
        "ych",
        "ym",
        "em",
        "ou",
        "ho",
        "mu",
        "mi",
        "ku",
        "ka",
        "ky",
        "ce",
        "ci",
        "u",
        "a",
        "e",
        "i",
        "y",
    ):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def node_score(question: str, node: dict[str, Any], prefer_high_level: bool) -> float:
    q_terms = token_set(question)
    thought_terms = token_set(str(node.get("thought", "")))
    overlap = len(q_terms & thought_terms)
    utility = float(node.get("utility", 0.0))
    level = int(node.get("level", 1))
    level_bonus = 0.25 * max(0, level - 1) if prefer_high_level else 0.0
    return overlap * 2.0 + utility + level_bonus


def fit_node_budget(nodes: list[dict[str, Any]], max_nodes: int) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    by_id = {node["id"]: node for node in nodes}

    def add_with_parents(node: dict[str, Any]) -> None:
        for parent_id in node.get("parents", []):
            parent = by_id.get(parent_id)
            if parent is not None and parent["id"] not in seen and len(ordered) < max_nodes:
                add_with_parents(parent)
        if node["id"] not in seen and len(ordered) < max_nodes:
            seen.add(node["id"])
            ordered.append(node)

    for node in nodes:
        if len(ordered) >= max_nodes:
            break
        add_with_parents(node)
    return ordered[:max_nodes]


def merge_ranked_nodes(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for node in group:
            if node["id"] in seen:
                continue
            seen.add(node["id"])
            merged.append(node)
    return merged


def make_random_hierarchy(nodes: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    flat = [node for node in nodes if int(node.get("level", 1)) == 1]
    shuffled = flat[:]
    rng.shuffle(shuffled)
    random_nodes = [dict(node) for node in flat]
    for i in range(0, max(0, len(shuffled) - 1), 2):
        left, right = shuffled[i], shuffled[i + 1]
        random_nodes.append(
            {
                "id": f"R{i // 2 + 1}",
                "level": 2,
                "thought": f"Náhodné seskupení: {left['thought']} / {right['thought']}",
                "parents": [left["id"], right["id"]],
                "source": [],
                "operation": "random_grouping",
                "support": "hypothetical",
                "utility": 0.2,
            }
        )
    return random_nodes


def select_nodes(
    nodes: list[dict[str, Any]],
    question: str,
    variant: str,
    seed: int,
    budget: int,
) -> list[dict[str, Any]]:
    if variant == "flat":
        return [node for node in nodes if int(node.get("level", 1)) == 1]
    if variant == "hierarchical_full":
        return sorted(nodes, key=lambda node: (int(node.get("level", 1)), str(node.get("id"))))
    if variant == "random_hierarchy":
        return make_random_hierarchy(nodes, seed)

    if variant == "flat_budget":
        candidates = [node for node in nodes if int(node.get("level", 1)) == 1]
        ranked = sorted(candidates, key=lambda node: node_score(question, node, False), reverse=True)
        return fit_node_budget(ranked, budget)
    if variant == "hierarchical_budget":
        atoms = [node for node in nodes if int(node.get("level", 1)) == 1]
        higher = [node for node in nodes if int(node.get("level", 1)) > 1]
        lexical_atoms = sorted(atoms, key=lambda node: node_score(question, node, False), reverse=True)
        ranked_higher = sorted(higher, key=lambda node: node_score(question, node, True), reverse=True)
        utility_atoms = sorted(atoms, key=lambda node: float(node.get("utility", 0.0)), reverse=True)
        ranked = merge_ranked_nodes(
            lexical_atoms[: max(3, budget // 3)],
            ranked_higher,
            utility_atoms[:2],
            lexical_atoms,
        )
        return fit_node_budget(ranked, budget)
    if variant == "random_hierarchy_budget":
        ranked = make_random_hierarchy(nodes, seed)
        random.Random(seed).shuffle(ranked)
        return fit_node_budget(ranked, budget)
    raise ValueError(f"Unknown node-selection variant: {variant}")


def process_scenario(
    args: argparse.Namespace,
    scenario: Scenario,
    scenario_idx: int,
    variants: list[str],
) -> dict[str, Any]:
    client = GeminiJsonClient(model=args.model, temperature=args.temperature)
    rows: list[dict[str, Any]] = []

    nodes, graph_trace = build_graph(client, scenario, args)
    summary = client.json_call("summary", summary_prompt(scenario.context), max_output_tokens=768)
    summary_text = str(summary.get("summary", "")) if isinstance(summary, dict) else ""

    scenario_report: dict[str, Any] = {
        "id": scenario.id,
        "task_type": scenario.task_type,
        "context_chars": len(scenario.context),
        "node_count": len(nodes),
        "nodes_by_level": {},
        "summary": summary_text,
        "nodes": nodes,
        "graph_trace": graph_trace,
    }
    for node in nodes:
        level = str(node.get("level", 1))
        scenario_report["nodes_by_level"][level] = scenario_report["nodes_by_level"].get(level, 0) + 1

    for question in scenario.questions:
        question_slots: list[dict[str, Any]] | None = None
        raw_task_graph: Any | None = None
        cached_task_nodes: list[dict[str, Any]] | None = None
        for variant in variants:
            selected_nodes: list[dict[str, Any]] = []
            if variant == "direct":
                result = client.json_call(
                    "answer_direct",
                    direct_answer_prompt(scenario.context, question.text, reasoning=False),
                    max_output_tokens=512,
                )
            elif variant == "reasoning":
                result = client.json_call(
                    "answer_reasoning",
                    direct_answer_prompt(scenario.context, question.text, reasoning=True),
                    max_output_tokens=768,
                )
            elif variant == "summary":
                result = client.json_call(
                    "answer_summary",
                    summary_answer_prompt(summary_text, question.text),
                    max_output_tokens=512,
                )
            elif variant == "hierarchical_loop":
                selected_nodes = select_nodes(
                    nodes,
                    question.text,
                    "hierarchical_budget",
                    args.seed + scenario_idx,
                    args.answer_node_budget,
                )
                first = client.json_call(
                    "answer_loop_first",
                    loop_answer_prompt(question.text, selected_nodes),
                    max_output_tokens=768,
                )
                if isinstance(first, dict) and first.get("status") == "needs_more_thoughts":
                    missing = first.get("missing") if isinstance(first.get("missing"), list) else []
                    extension = client.json_call(
                        "loop_expand",
                        targeted_expand_prompt(
                            nodes,
                            [str(item) for item in missing],
                            args.max_loop_nodes,
                        ),
                        max_output_tokens=768,
                    )
                    extra_nodes = clean_nodes(
                        extension,
                        prefix="L",
                        forced_level=args.max_depth + 1,
                        existing_ids={node["id"] for node in nodes},
                        max_nodes=args.max_loop_nodes,
                    )
                    selected_nodes = fit_node_budget(
                        merge_ranked_nodes(selected_nodes, extra_nodes, nodes),
                        args.answer_node_budget + args.max_loop_nodes,
                    )
                    result = client.json_call(
                        "answer_loop_second",
                        loop_answer_prompt(question.text, selected_nodes),
                        max_output_tokens=768,
                    )
                    result["_loop_missing"] = missing
                    result["_loop_extra_nodes"] = extra_nodes
                else:
                    result = first
                if not (isinstance(result, dict) and str(result.get("answer", "")).strip()):
                    forced = client.json_call(
                        "answer_loop_forced",
                        graph_answer_prompt(question.text, selected_nodes, "hierarchical_loop_forced"),
                        max_output_tokens=768,
                    )
                    if isinstance(forced, dict):
                        forced["_loop_previous_result"] = result
                    result = forced
            elif variant in {
                "task_graph",
                "task_graph_flat_q",
                "task_graph_shuffled_parents",
                "task_graph_no_parents",
                "task_cot_json",
            }:
                selected_nodes = select_nodes(
                    nodes,
                    question.text,
                    "hierarchical_budget",
                    args.seed + scenario_idx,
                    args.answer_node_budget,
                )
                if variant in {
                    "task_graph",
                    "task_graph_flat_q",
                    "task_graph_shuffled_parents",
                    "task_graph_no_parents",
                } and question_slots is None:
                    question_slots = get_question_slots(client, question.text)
                if variant in {"task_graph", "task_graph_flat_q", "task_graph_shuffled_parents"}:
                    slots = question_slots or [{"id": "S1", "need": question.text, "kind": "final_decision"}]
                    if cached_task_nodes is None:
                        raw_task_graph, cached_task_nodes = build_task_graph_nodes(
                            client,
                            question,
                            slots,
                            selected_nodes,
                            args,
                        )
                    task_nodes = cached_task_nodes
                    if variant == "task_graph":
                        result = client.json_call(
                            "answer_task_graph",
                            task_graph_answer_prompt(question.text, slots, selected_nodes, task_nodes),
                            max_output_tokens=768,
                        )
                        if isinstance(result, dict):
                            result["_task_slots"] = slots
                            result["_task_nodes"] = task_nodes
                            result["_raw_task_graph"] = raw_task_graph
                    elif variant == "task_graph_flat_q":
                        flat_task_nodes = strip_task_node_structure(task_nodes)
                        result = client.json_call(
                            "answer_task_graph_flat_q",
                            task_graph_flat_q_answer_prompt(
                                question.text,
                                slots,
                                selected_nodes,
                                flat_task_nodes,
                                "task_graph_flat_q",
                            ),
                            max_output_tokens=768,
                        )
                        if isinstance(result, dict):
                            result["_task_slots"] = slots
                            result["_task_nodes"] = flat_task_nodes
                            result["_raw_task_graph"] = raw_task_graph
                    else:
                        shuffled_nodes = shuffled_task_node_parents(
                            task_nodes,
                            args.seed + scenario_idx * 1000 + len(rows),
                        )
                        result = client.json_call(
                            "answer_task_graph_shuffled_parents",
                            task_graph_answer_prompt(question.text, slots, selected_nodes, shuffled_nodes),
                            max_output_tokens=768,
                        )
                        if isinstance(result, dict):
                            result["_task_slots"] = slots
                            result["_task_nodes"] = shuffled_nodes
                            result["_raw_task_graph"] = raw_task_graph
                    selected_nodes = selected_nodes + task_nodes
                elif variant == "task_graph_no_parents":
                    result = client.json_call(
                        "answer_task_graph_no_parents",
                        task_no_parents_prompt(
                            question.text,
                            question_slots or [{"id": "S1", "need": question.text, "kind": "final_decision"}],
                            selected_nodes,
                            args.max_task_nodes,
                        ),
                        max_output_tokens=1024,
                    )
                else:
                    result = client.json_call(
                        "answer_task_cot_json",
                        task_cot_json_prompt(question.text, selected_nodes),
                        max_output_tokens=1024,
                    )
            elif variant in {
                "flat",
                "hierarchical_full",
                "random_hierarchy",
                "flat_budget",
                "hierarchical_budget",
                "random_hierarchy_budget",
            }:
                selected_nodes = select_nodes(
                    nodes,
                    question.text,
                    variant,
                    args.seed + scenario_idx,
                    args.answer_node_budget,
                )
                result = client.json_call(
                    f"answer_{variant}",
                    graph_answer_prompt(question.text, selected_nodes, variant),
                    max_output_tokens=768,
                )
            else:
                raise ValueError(f"Unknown variant: {variant}")

            answer = str(result.get("answer", "")) if isinstance(result, dict) else ""
            rows.append(
                {
                    "scenario_id": scenario.id,
                    "task_type": scenario.task_type,
                    "question_id": question.id,
                    "question": question.text,
                    "variant": variant,
                    "answer": answer,
                    "expected_aliases": question.aliases,
                    "required_terms": question.required_terms,
                    "forbidden_terms": question.forbidden_terms,
                    "correct": answer_matches(answer, question),
                    "selected_node_ids": [node["id"] for node in selected_nodes],
                    "raw_result": result,
                }
            )

    for call in client.calls:
        call["scenario_id"] = scenario.id
    return {
        "scenario_idx": scenario_idx,
        "scenario_report": scenario_report,
        "rows": rows,
        "api_calls": client.calls,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    load_env()
    if args.scenario_suite == "large500":
        all_scenarios = make_large_general_scenarios(args.target_questions)
    else:
        all_scenarios = make_scenarios()
    scenarios = all_scenarios[args.scenario_offset : args.scenario_offset + args.limit_scenarios]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    scenario_reports: list[dict[str, Any]] = []
    api_calls: list[dict[str, Any]] = []

    if args.parallelism <= 1:
        results = [
            process_scenario(args, scenario, scenario_idx, variants)
            for scenario_idx, scenario in enumerate(scenarios)
        ]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            futures = {
                executor.submit(process_scenario, args, scenario, scenario_idx, variants): scenario
                for scenario_idx, scenario in enumerate(scenarios)
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"Completed {result['scenario_idx'] + 1}/{len(scenarios)}: "
                    f"{result['scenario_report']['id']}",
                    flush=True,
                )

    for result in sorted(results, key=lambda item: int(item["scenario_idx"])):
        scenario_reports.append(result["scenario_report"])
        rows.extend(result["rows"])
        api_calls.extend(result["api_calls"])

    by_variant: dict[str, dict[str, Any]] = {}
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        correct = sum(1 for row in variant_rows if row["correct"])
        by_variant[variant] = {
            "correct": correct,
            "total": len(variant_rows),
            "accuracy": correct / len(variant_rows) if variant_rows else 0.0,
        }

    by_task_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for task_type in sorted({scenario.task_type for scenario in scenarios}):
        by_task_variant[task_type] = {}
        for variant in variants:
            task_rows = [
                row for row in rows if row["variant"] == variant and row["task_type"] == task_type
            ]
            correct = sum(1 for row in task_rows if row["correct"])
            by_task_variant[task_type][variant] = {
                "correct": correct,
                "total": len(task_rows),
                "accuracy": correct / len(task_rows) if task_rows else 0.0,
            }

    return {
        "model": args.model,
        "temperature": args.temperature,
        "elapsed_s": time.perf_counter() - started,
        "scenario_count": len(scenarios),
        "scenario_suite": args.scenario_suite,
        "scenario_offset": args.scenario_offset,
        "available_scenario_count": len(all_scenarios),
        "question_count": sum(len(s.questions) for s in scenarios),
        "answer_node_budget": args.answer_node_budget,
        "max_task_nodes": args.max_task_nodes,
        "max_nodes": args.max_nodes,
        "max_depth": args.max_depth,
        "parallelism": args.parallelism,
        "verify_compositions": args.verify_compositions,
        "variants": variants,
        "summary_by_variant": by_variant,
        "summary_by_task_variant": by_task_variant,
        "rows": rows,
        "scenarios": scenario_reports,
        "api_calls": api_calls,
    }


def print_summary(report: dict[str, Any]) -> None:
    print(json.dumps(report["summary_by_variant"], ensure_ascii=False, indent=2))
    print("\nFailures:")
    failures = [row for row in report["rows"] if not row["correct"]]
    if not failures:
        print("None")
    max_failures = int(report.get("max_printed_failures", 80))
    for row in failures[:max_failures]:
        print(
            f"FAIL {row['variant']:24} {row['scenario_id']}/{row['question_id']}: "
            f"{row['answer']}"
        )
    if len(failures) > max_failures:
        print(f"... {len(failures) - max_failures} more failures omitted from console output")
    if report.get("print_rows", True):
        print("\nPer-question results:")
        for row in report["rows"]:
            mark = "OK" if row["correct"] else "FAIL"
            print(
                f"{mark:4} {row['variant']:24} {row['scenario_id']}/{row['question_id']}: "
                f"{row['answer']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--scenario-suite", choices=["standard", "large500"], default="standard")
    parser.add_argument("--target-questions", type=int, default=500)
    parser.add_argument("--limit-scenarios", type=int, default=22)
    parser.add_argument("--scenario-offset", type=int, default=0)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--max-atoms", type=int, default=18)
    parser.add_argument("--max-nodes", type=int, default=36)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-new-per-level", type=int, default=7)
    parser.add_argument("--answer-node-budget", type=int, default=12)
    parser.add_argument("--max-task-nodes", type=int, default=10)
    parser.add_argument("--max-loop-nodes", type=int, default=4)
    parser.add_argument("--graph-output-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verify-compositions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--variants",
        default=(
            "direct,reasoning,summary,flat_budget,hierarchical_budget,"
            "random_hierarchy_budget,hierarchical_loop,task_graph,task_graph_no_parents,task_cot_json"
        ),
        help="Comma-separated variants to run.",
    )
    parser.add_argument("--out", default="reports/text_recursive_thought_graph_gemini_report.json")
    parser.add_argument("--print-rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-printed-failures", type=int, default=80)
    args = parser.parse_args()

    report = run_experiment(args)
    report["print_rows"] = args.print_rows
    report["max_printed_failures"] = args.max_printed_failures
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
