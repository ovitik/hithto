from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from thought_tokens.thought_builder import ThoughtBuilder, ThoughtBuilderOutput
from thought_tokens.hierarchy import ThoughtNode


@dataclass
class GemmaLoadConfig:
    model_name: str = "google/gemma-4-E2B"
    dtype: str = "bf16"
    qlora_4bit: bool = False
    trust_remote_code: bool = True
    freeze_backbone: bool = True
    unfreeze_top_layers: int = 0


class GemmaThoughtWrapper(nn.Module):
    """Wrap a Hugging Face causal LM and insert latent thoughts through inputs_embeds."""

    def __init__(
        self,
        model: nn.Module,
        thought_builder: ThoughtBuilder | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.thought_builder = thought_builder
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        config: GemmaLoadConfig,
        thought_builder: ThoughtBuilder | None = None,
    ) -> "GemmaThoughtWrapper":
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        dtype = _dtype_from_name(config.dtype)
        quantization_config = None
        if config.qlora_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        hf_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code)
        model_kwargs = {
            "torch_dtype": None if config.qlora_4bit else dtype,
            "quantization_config": quantization_config,
            "trust_remote_code": config.trust_remote_code,
            "output_hidden_states": True,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        if getattr(hf_config, "model_type", None) == "gemma4":
            model = AutoModelForImageTextToText.from_pretrained(config.model_name, **model_kwargs)
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
            except ValueError:
                model = AutoModelForImageTextToText.from_pretrained(config.model_name, **model_kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if config.freeze_backbone:
            freeze_model(model)
        if config.unfreeze_top_layers > 0:
            unfreeze_top_layers(model, config.unfreeze_top_layers)
        if thought_builder is not None:
            first_param = next(model.parameters())
            thought_builder.to(device=first_param.device, dtype=first_param.dtype)
        return cls(model, thought_builder, tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_thoughts: bool = True,
        max_levels: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.thought_builder is None or not use_thoughts:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                **kwargs,
            )
            return {"model_output": out, "thought_output": None}

        first = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        )
        hidden = first.hidden_states[-1]
        per_layer_inputs = self._get_per_layer_inputs(input_ids, inputs_embeds)
        thought_output = self.thought_builder(
            hidden,
            attention_mask=attention_mask,
            token_ids=input_ids,
            per_layer_inputs=per_layer_inputs,
            max_levels=max_levels,
        )
        labels_for_thoughts = align_labels_after_compression(labels, thought_output) if labels is not None else None
        second_kwargs = {}
        if thought_output.per_layer_inputs is not None:
            second_kwargs["per_layer_inputs"] = thought_output.per_layer_inputs
        second = self.model(
            inputs_embeds=thought_output.embeddings,
            attention_mask=thought_output.attention_mask,
            labels=labels_for_thoughts,
            output_hidden_states=True,
            use_cache=False,
            **second_kwargs,
            **kwargs,
        )
        return {"model_output": second, "teacher_output": first, "thought_output": thought_output}

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.get_input_embeddings()
        return embeddings(input_ids)

    def _get_per_layer_inputs(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if input_ids is None:
            return None
        candidates = [
            getattr(getattr(self.model, "model", None), "language_model", None),
            getattr(self.model, "language_model", None),
            getattr(getattr(self.model, "model", None), "model", None),
        ]
        for candidate in candidates:
            getter = getattr(candidate, "get_per_layer_inputs", None)
            if getter is not None:
                return getter(input_ids, inputs_embeds)
        return None


def align_labels_after_compression(
    labels: torch.Tensor,
    thought_output: ThoughtBuilderOutput,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Conservative label alignment.

    Replacement mode changes sequence length, so labels that no longer map cleanly to a source token
    are ignored. Source tokens keep their original labels; thought nodes receive ignore_index.
    """

    max_len = thought_output.embeddings.shape[1]
    aligned = labels.new_full((labels.shape[0], max_len), ignore_index)
    for b, units in enumerate(thought_output.units):
        for j, unit in enumerate(units[:max_len]):
            if hasattr(unit, "position"):
                pos = int(unit.position)
                if pos < labels.shape[1]:
                    aligned[b, j] = labels[b, pos]
            elif isinstance(unit, ThoughtNode) and unit.source_positions:
                pos = max(unit.source_positions)
                if pos < labels.shape[1]:
                    aligned[b, j] = labels[b, pos]
    return aligned


def freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def unfreeze_top_layers(model: nn.Module, n_layers: int) -> None:
    layers = None
    for path in ("model.layers", "language_model.model.layers", "transformer.h"):
        obj: Any = model
        ok = True
        for name in path.split("."):
            if not hasattr(obj, name):
                ok = False
                break
            obj = getattr(obj, name)
        if ok:
            layers = obj
            break
    if layers is None:
        return
    for layer in list(layers)[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad_(True)


def _dtype_from_name(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")
