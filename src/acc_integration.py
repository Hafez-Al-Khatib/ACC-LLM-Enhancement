"""ACC-enhanced generation wrapper for HuggingFace causal LMs.

Uses a custom LogitsProcessor injected into the standard transformers
.generate() pipeline. This is more robust than a manual decoding loop
because it delegates KV-cache management, attention masking, and beam
search support to the well-tested HuggingFace generator.

Intervention strategies:
  - "flag":       mark uncertain spans with a [UNCERTAIN] token.
  - "regenerate": re-sample at lower temperature when entropy is high.
  - "warning":    prefix uncertain spans with a warning string.

Self-consistency checking (optional):
  Generates N candidate continuations and compares their semantic
  embeddings. Low pairwise similarity signals internal contradiction.
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
    consistency_score: List[Optional[float]] = field(default_factory=list)
    contradiction_detected: List[Optional[bool]] = field(default_factory=list)
    scores: Optional[Tuple[torch.FloatTensor, ...]] = None


class _EntropyLogitsProcessor(LogitsProcessor):
    """LogitsProcessor that monitors entropy and optionally regenerates tokens.

    This is called *inside* the HF .generate() loop, after the model forward
    pass but before token selection. It has access to the raw logits for the
    current position and can modify them (e.g., by forcing a resample).
    """

    def __init__(
        self,
        monitor: EntropyMonitor,
        tokenizer,
        action: Action = "flag",
        regen_multiplier: float = 2.0,
        max_regenerations: int = 3,
    ):
        self.monitor = monitor
        self.tokenizer = tokenizer
        self.action: Action = action
        self.regen_multiplier = float(regen_multiplier)
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
                    # Re-sample at lower temperature by scaling logits UP.
                    # HF applies temperature as logits / temperature.
                    # Multiplying logits here is equivalent to dividing the
                    # final temperature by the same factor:
                    #   effective_temperature = base_temperature / multiplier
                    multiplier = self.regen_multiplier ** (self.regen_counts[b] + 1)
                    next_scores[b] = row_logits * multiplier
                    self.regen_counts[b] += 1
                elif self.action == "flag":
                    # Marker position is prompt_len + gen_step + 1 (after the token)
                    self.text_inserts[b].append((self.prompt_len + gen_step + 1, FLAG_MARKER))
                elif self.action == "warning":
                    # Prefix position is prompt_len + gen_step (before the token)
                    self.text_inserts[b].append((self.prompt_len + gen_step, WARNING_PREFIX))

        return next_scores


class SelfConsistencyChecker:
    """Semantic self-consistency checker using multiple sampled continuations.

    Generates N candidates for a prompt, embeds each continuation via
    mean-pooled hidden states from the base model, and checks for
    outlier clusters that indicate contradiction or hallucination.
    """

    def __init__(
        self,
        model,
        tokenizer,
        n_candidates: int = 5,
        similarity_threshold: float = 0.75,
        max_new_tokens: int = 50,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_candidates = int(n_candidates)
        self.similarity_threshold = float(similarity_threshold)
        self.max_new_tokens = int(max_new_tokens)
        self.device = getattr(model, "device", None) or next(model.parameters()).device

    def generate_candidates(self, prompt: str, **gen_kwargs) -> List[str]:
        """Generate N diverse continuation strings for *prompt*."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        # Ensure do_sample is True for diversity
        gen_kwargs.setdefault("do_sample", True)
        gen_kwargs.setdefault("temperature", gen_kwargs.get("temperature", 0.7))
        gen_kwargs.setdefault("top_p", gen_kwargs.get("top_p", 0.9))

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=gen_kwargs.get("max_new_tokens", self.max_new_tokens),
                num_return_sequences=self.n_candidates,
                do_sample=gen_kwargs["do_sample"],
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Strip prompt to keep only continuations
        continuations = []
        for seq in outputs:
            continuation_ids = seq[prompt_len:]
            text = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
            continuations.append(text)
        return continuations

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Return L2-normalized, mean-pooled token embeddings."""
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)

        # Mean pool with attention mask
        mask = inputs.attention_mask.unsqueeze(-1).float()
        sum_embeddings = (last_hidden * mask).sum(dim=1)
        mean_embeddings = sum_embeddings / mask.sum(dim=1).clamp(min=1e-9)
        mean_embeddings = F.normalize(mean_embeddings, p=2, dim=1)
        return mean_embeddings

    def check(self, prompt: str, **gen_kwargs) -> Dict[str, Union[float, bool, List[int], List[str]]]:
        """Run self-consistency check on *prompt*.

        Returns
        -------
        dict with keys:
            consistency_score : float
                Mean pairwise cosine similarity within the largest cluster.
            contradiction_detected : bool
                True if at least one candidate is an outlier w.r.t. the cluster.
            outlier_indices : List[int]
                Indices of outlier candidates.
            candidates : List[str]
                The generated continuation strings.
        """
        candidates = self.generate_candidates(prompt, **gen_kwargs)
        n = len(candidates)
        if n < 2:
            return {
                "consistency_score": 1.0,
                "contradiction_detected": False,
                "outlier_indices": [],
                "candidates": candidates,
            }

        embeddings = self.embed_texts(candidates)
        sim_matrix = embeddings @ embeddings.T  # (n, n)

        # Identify the largest cluster: for each candidate count how many
        # others exceed the similarity threshold.
        counts = (sim_matrix > self.similarity_threshold).sum(dim=1) - 1  # exclude self
        best_idx = int(counts.argmax())
        best_cluster_mask = sim_matrix[best_idx] > self.similarity_threshold

        # Outliers are candidates whose maximum similarity to any member
        # of the best cluster is below threshold.
        best_cluster_embs = embeddings[best_cluster_mask]
        max_sims_to_cluster = (embeddings @ best_cluster_embs.T).max(dim=1).values
        outlier_mask = max_sims_to_cluster < self.similarity_threshold

        if best_cluster_mask.sum() > 1:
            cluster_sims = sim_matrix[best_cluster_mask][:, best_cluster_mask]
            consistency_score = float(cluster_sims.mean())
        else:
            consistency_score = float(max_sims_to_cluster[best_idx])

        return {
            "consistency_score": consistency_score,
            "contradiction_detected": bool(outlier_mask.any().item()),
            "outlier_indices": outlier_mask.nonzero(as_tuple=True)[0].tolist(),
            "candidates": candidates,
        }


class ACCEnhancedGenerator:
    """Wrap a HF CausalLM with per-token entropy monitoring.

    Internally uses a LogitsProcessor injected into the standard
    transformers .generate() pipeline. Optionally enables semantic
    self-consistency checking across multiple sampled continuations.
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
        regen_multiplier: float = 2.0,
        max_regenerations: int = 3,
        use_self_consistency: bool = False,
        self_consistency_candidates: int = 5,
        self_consistency_threshold: float = 0.75,
        self_consistency_max_new_tokens: Optional[int] = None,
    ):
        if action not in ("flag", "regenerate", "warning"):
            raise ValueError(f"unknown action: {action}")
        self.model = model
        self.tokenizer = tokenizer
        self.action: Action = action
        self.regen_multiplier = float(regen_multiplier)
        self.max_regenerations = int(max_regenerations)
        self.monitor = monitor or EntropyMonitor(
            threshold=threshold,
            mode=mode,
            action=action,
            window_size=window_size,
            warmup=warmup,
        )
        self.device = getattr(model, "device", None) or next(model.parameters()).device

        self.use_self_consistency = bool(use_self_consistency)
        self.self_consistency_checker: Optional[SelfConsistencyChecker] = None
        if self.use_self_consistency:
            self.self_consistency_checker = SelfConsistencyChecker(
                model=model,
                tokenizer=tokenizer,
                n_candidates=self_consistency_candidates,
                similarity_threshold=self_consistency_threshold,
                max_new_tokens=self_consistency_max_new_tokens or 50,
            )

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
            regen_multiplier=self.regen_multiplier,
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

        # Self-consistency check
        consistency_scores: List[Optional[float]] = [None] * batch_size
        contradiction_flags: List[Optional[bool]] = [None] * batch_size
        if self.use_self_consistency and self.self_consistency_checker is not None:
            for b in range(batch_size):
                prompt_text = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
                try:
                    sc_result = self.self_consistency_checker.check(
                        prompt_text,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    consistency_scores[b] = sc_result["consistency_score"]
                    contradiction_flags[b] = sc_result["contradiction_detected"]
                except Exception as exc:
                    logger.warning("Self-consistency check failed for batch item %d: %s", b, exc)

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
            consistency_score=consistency_scores,
            contradiction_detected=contradiction_flags,
            scores=outputs.scores,
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
