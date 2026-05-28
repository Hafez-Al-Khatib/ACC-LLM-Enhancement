"""ACC-enhanced generation wrapper for HuggingFace causal LMs.

Uses a custom LogitsProcessor injected into the standard transformers
.generate() pipeline. This is more robust than a manual decoding loop
because it delegates KV-cache management, attention masking, and beam
search support to the well-tested HuggingFace generator.

Three intervention strategies are supported:
  - "flag":       mark uncertain spans with a [UNCERTAIN] token.
  - "regenerate": re-sample at higher temperature when entropy is high.
  - "warning":    prefix uncertain spans with a warning string.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from .acc_layer import Action, EntropyEvent, EntropyMonitor, ThresholdMode

logger = logging.getLogger(__name__)

FLAG_MARKER = " [UNCERTAIN]"
WARNING_PREFIX = "[WARNING: low-confidence next token] "


@dataclass
class ACCGenerationOutput:
    """Rich return type that mirrors transformers GenerateOutput shape."""

    sequences: torch.LongTensor
    text: List[str]
    per_token_entropy: List[List[float]]
    uncertain_steps: List[List[int]]
    events: List[List[dict]] = field(default_factory=list)
    regenerations: List[int] = field(default_factory=list)
    confidence_score: List[float] = field(default_factory=list)


class _EntropyLogitsProcessor(LogitsProcessor):
    """LogitsProcessor that monitors entropy and optionally regenerates tokens.

    This is called *inside* the HF .generate() loop, after the model forward
    pass but before token selection. It has access to the raw logits for the
    current position and can modify them (e.g. by forcing a resample).
    """

    def __init__(
        self,
        monitor: EntropyMonitor,
        tokenizer,
        action: Action = "flag",
        regen_temperature_multiplier: float = 1.5,
        max_regenerations: int = 3,
    ):
        self.monitor = monitor
        self.tokenizer = tokenizer
        self.action: Action = action
        self.regen_temperature_multiplier = float(regen_temperature_multiplier)
        self.max_regenerations = int(max_regenerations)
        # Per-batch-row tracking
        self.per_token_entropy: List[List[float]] = []
        self.uncertain_steps: List[List[int]] = []
        self.text_inserts: List[List[Tuple[int, str]]] = []
        self.regen_counts: List[int] = []
        self.prompt_len: int = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits for the current generation step.

        input_ids: (batch, seq_len_so_far) — full sequence including prompt
        scores:    (batch, vocab) — logits for the *next* token position
        """
        batch_size = scores.shape[0]
        if not self.per_token_entropy:
            # First call — init tracking structures
            self.per_token_entropy = [[] for _ in range(batch_size)]
            self.uncertain_steps = [[] for _ in range(batch_size)]
            self.text_inserts = [[] for _ in range(batch_size)]
            self.regen_counts = [0] * batch_size
            self.prompt_len = input_ids.shape[1]

        gen_step = input_ids.shape[1] - self.prompt_len
        next_scores = scores.clone()

        for b in range(batch_size):
            row_logits = scores[b]
            entropy = self.monitor.observe(row_logits)
            self.per_token_entropy[b].append(entropy)
            breached = self.monitor.check_threshold(entropy)

            if breached:
                self.uncertain_steps[b].append(gen_step)
                if self.action == "regenerate" and self.regen_counts[b] < self.max_regenerations:
                    # Re-sample at higher temperature by scaling logits down
                    # (lower logits = higher effective temperature)
                    multiplier = self.regen_temperature_multiplier ** (self.regen_counts[b] + 1)
                    next_scores[b] = row_logits / max(multiplier, 1e-5)
                    self.regen_counts[b] += 1
                elif self.action == "flag":
                    # Marker position is prompt_len + gen_step + 1 (after the token)
                    self.text_inserts[b].append((self.prompt_len + gen_step + 1, FLAG_MARKER))
                elif self.action == "warning":
                    # Prefix position is prompt_len + gen_step (before the token)
                    self.text_inserts[b].append((self.prompt_len + gen_step, WARNING_PREFIX))

        return next_scores


class ACCEnhancedGenerator:
    """Wrap a HF CausalLM with per-token entropy monitoring.

    Internally uses a LogitsProcessor injected into the standard
    transformers .generate() pipeline.
    """

    def __init__(
        self,
        model,
        tokenizer,
        monitor: Optional[EntropyMonitor] = None,
        threshold: float = 3.5,
        action: Action = "flag",
        mode: ThresholdMode = "absolute",
        window_size: int = 32,
        warmup: int = 4,
        regen_temperature_multiplier: float = 1.5,
        max_regenerations: int = 3,
    ):
        if action not in ("flag", "regenerate", "warning"):
            raise ValueError(f"unknown action: {action}")
        self.model = model
        self.tokenizer = tokenizer
        self.action: Action = action
        self.regen_temperature_multiplier = float(regen_temperature_multiplier)
        self.max_regenerations = int(max_regenerations)
        self.monitor = monitor or EntropyMonitor(
            threshold=threshold,
            mode=mode,
            action=action,
            window_size=window_size,
            warmup=warmup,
        )
        self.device = getattr(model, "device", None) or next(model.parameters()).device

    # ------------------------------------------------------------------ public

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        return_dict_in_generate: bool = False,
        **kwargs,
    ) -> Union[torch.LongTensor, ACCGenerationOutput]:
        """Entropy-aware generation using the standard HF .generate() pipeline."""
        if input_ids is None:
            raise ValueError("input_ids is required")
        if kwargs:
            logger.debug("ACCEnhancedGenerator ignoring kwargs: %s", list(kwargs))

        self.monitor.reset()
        processor = _EntropyLogitsProcessor(
            monitor=self.monitor,
            tokenizer=self.tokenizer,
            action=self.action,
            regen_temperature_multiplier=self.regen_temperature_multiplier,
            max_regenerations=self.max_regenerations,
        )

        logits_processor = [processor]

        outputs = self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else None,
            do_sample=do_sample,
            pad_token_id=pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            logits_processor=logits_processor,
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequences = outputs.sequences
        batch_size = sequences.shape[0]
        prompt_len = input_ids.shape[1]

        # Decode with markers
        text = self._decode_with_markers(sequences, prompt_len, processor.text_inserts)

        if not return_dict_in_generate:
            return sequences

        return ACCGenerationOutput(
            sequences=sequences,
            text=text,
            per_token_entropy=processor.per_token_entropy,
            uncertain_steps=processor.uncertain_steps,
            events=[[asdict(e) for e in self.monitor.events]] * batch_size,
            regenerations=processor.regen_counts,
            confidence_score=[self.monitor.get_confidence_score()] * batch_size,
        )

    def generate_from_prompt(
        self, prompt: str, return_dict_in_generate: bool = True, **gen_kwargs
    ) -> Union[torch.LongTensor, ACCGenerationOutput]:
        """Convenience wrapper: tokenize a string prompt then call `generate`."""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return self.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            return_dict_in_generate=return_dict_in_generate,
            **gen_kwargs,
        )

    # ------------------------------------------------------------------ helpers

    def _decode_with_markers(
        self,
        generated: torch.LongTensor,
        prompt_len: int,
        text_inserts: List[List[Tuple[int, str]]],
    ) -> List[str]:
        """Decode each sequence with markers spliced in at token boundaries."""
        out: List[str] = []
        for b in range(generated.shape[0]):
            seq = generated[b].tolist()
            pieces = [self.tokenizer.decode(seq[:prompt_len], skip_special_tokens=True)]
            cursor = prompt_len
            for pos, marker in sorted(text_inserts[b], key=lambda x: x[0]):
                pos = min(max(pos, cursor), len(seq))
                if pos > cursor:
                    pieces.append(self.tokenizer.decode(seq[cursor:pos], skip_special_tokens=True))
                pieces.append(marker)
                cursor = pos
            if cursor < len(seq):
                pieces.append(self.tokenizer.decode(seq[cursor:], skip_special_tokens=True))
            out.append("".join(pieces))
        return out
