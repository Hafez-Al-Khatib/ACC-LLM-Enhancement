"""ACC-enhanced generation wrapper for HuggingFace causal LMs.

Uses a custom LogitsProcessor injected into the standard transformers
.generate() pipeline. This is more robust than a manual decoding loop
because it delegates KV-cache management, attention masking, and beam
search support to the well-tested HuggingFace generator.

Intervention strategies:
  - "flag":       mark uncertain spans with a configurable marker.
  - "regenerate": re-sample at lower temperature when entropy is high.
  - "warning":    prefix uncertain spans with a warning string.
  - "suppress":   block the top-1 token, forcing selection of an alternative.

Self-consistency checking (optional):
  Generates N candidate continuations and compares their semantic
  embeddings. Low pairwise similarity signals internal contradiction.

Generation-time conflict detection (optional):
  Integrates a PredictiveCodingDetector into the logits processor for
  real-time per-token classification. The detector runs on hidden states
captured during generation and feeds into a UnifiedDecisionEngine that
fuses entropy and conflict signals into a single intervention decision.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from .acc_layer import Action, EntropyEvent, EntropyMonitor, ThresholdMode
from .acc_conflict_detector import PredictiveCodingDetector, MultiLayerGenerationExtractor

logger = logging.getLogger(__name__)

# Legacy markers (kept for backward compatibility)
FLAG_MARKER = " [UNCERTAIN]"
WARNING_PREFIX = "[WARNING: low-confidence next token] "
HALLUCINATION_MARKER = " [HALLUCINATION DETECTED]"
CONTRADICTION_MARKER = " [CONTRADICTION DETECTED]"


# =============================================================================
# 1. Configurable markers
# =============================================================================

@dataclass
class MarkerConfig:
    """User-configurable text markers for generation-time interventions."""

    hallucination: str = " [HALLUCINATION DETECTED]"
    contradiction: str = " [CONTRADICTION DETECTED]"
    uncertain: str = " [UNCERTAIN]"
    entropy_flag: str = " [ENTROPY FLAG]"
    warning_prefix: str = "[WARNING: low-confidence next token] "
    inline: bool = True
    """If True, markers are spliced inline at token boundaries.
    If False, markers are collected and appended as footnotes at the end."""

    def get_marker(
        self,
        primary: Optional[str] = None,
        secondary: Optional[str] = None,
    ) -> str:
        """Return the appropriate marker string for a classification result."""
        if primary == "unsupported":
            if secondary == "hallucinated":
                return self.hallucination
            elif secondary == "contradictory":
                return self.contradiction
            return self.entropy_flag
        elif primary == "uncertain":
            return self.uncertain
        return self.entropy_flag


# =============================================================================
# 2. Decision dataclass
# =============================================================================

@dataclass
class Decision:
    """A single intervention decision for one generation step."""

    action: Literal["pass", "flag", "regenerate", "suppress", "warning"]
    reason: str
    entropy: float
    conflict_score: Optional[float] = None
    primary: Optional[str] = None
    secondary: Optional[str] = None


# =============================================================================
# 3. Unified Decision Engine
# =============================================================================

@dataclass
class UnifiedDecisionEngine:
    """Fuses entropy monitoring and conflict-detection signals into a single
    intervention decision per generation step.

    The engine supports a configurable decision matrix:

    +-------------------+-------------------+-----------+----------+
    | Entropy Uncertain | Conflict > τ      | Primary   | Decision |
    +-------------------+-------------------+-----------+----------+
    | Yes               | —                 | —         | action   |
    | —                 | Yes               | unsupported| action  |
    | —                 | Yes               | uncertain | flag     |
    | Yes               | Yes               | unsupported| regenerate (if dual_signal_regenerate) |
    | No                | No                | —         | pass     |
    +-------------------+-------------------+-----------+----------+

    Parameters
    ----------
    entropy_threshold : float
        Minimum entropy required for the entropy signal to be considered.
        The actual threshold used by :class:`EntropyMonitor` may differ;
        this is the decision-engine-level filter.
    conflict_score_threshold : float
        Minimum conflict score [0, 1] for the conflict signal to fire.
    action : {"flag", "regenerate", "suppress", "warning"}
        Default action for single-signal breaches.
    regen_multiplier : float
        Temperature-dividing multiplier applied on regenerate.
        ``effective_temperature = base_temperature / multiplier``.
    dual_signal_regenerate : bool
        If True, the simultaneous presence of an entropy breach AND a
        high conflict score on an *unsupported* primary label triggers
        the stronger "regenerate" action regardless of the default.
    marker_config : MarkerConfig
        Marker strings used when the decision resolves to "flag" or "warning".
    """

    entropy_threshold: float = 1.5
    conflict_score_threshold: float = 0.7
    action: Literal["flag", "regenerate", "suppress", "warning"] = "flag"
    regen_multiplier: float = 2.0
    dual_signal_regenerate: bool = True
    marker_config: MarkerConfig = field(default_factory=MarkerConfig)

    def decide(
        self,
        entropy: float,
        is_uncertain: bool,
        conflict_score: Optional[float] = None,
        primary: Optional[str] = None,
        secondary: Optional[str] = None,
    ) -> Decision:
        """Return a :class:`Decision` given the current signals."""
        has_conflict = (
            conflict_score is not None
            and conflict_score > self.conflict_score_threshold
        )
        has_entropy = is_uncertain and entropy > self.entropy_threshold

        # ------------------------------------------------------------------
        # Dual signal → strongest action
        # ------------------------------------------------------------------
        if (
            has_entropy
            and has_conflict
            and primary == "unsupported"
            and self.dual_signal_regenerate
        ):
            return Decision(
                action="regenerate",
                reason=(
                    f"Dual signal: entropy={entropy:.3f} + "
                    f"conflict={conflict_score:.3f}, primary={primary}"
                    f" → REGENERATE"
                ),
                entropy=entropy,
                conflict_score=conflict_score,
                primary=primary,
                secondary=secondary,
            )

        # ------------------------------------------------------------------
        # Single conflict signal
        # ------------------------------------------------------------------
        if has_conflict:
            if primary == "unsupported":
                return Decision(
                    action=self.action,
                    reason=(
                        f"Conflict: score={conflict_score:.3f}, "
                        f"primary={primary}, secondary={secondary}"
                        f" → {self.action.upper()}"
                    ),
                    entropy=entropy,
                    conflict_score=conflict_score,
                    primary=primary,
                    secondary=secondary,
                )
            elif primary == "uncertain":
                return Decision(
                    action="flag",
                    reason=(
                        f"Conflict: score={conflict_score:.3f}, "
                        f"primary={primary} → FLAG"
                    ),
                    entropy=entropy,
                    conflict_score=conflict_score,
                    primary=primary,
                    secondary=secondary,
                )

        # ------------------------------------------------------------------
        # Single entropy signal
        # ------------------------------------------------------------------
        if has_entropy:
            # Suppress only makes sense when backed by a conflict signal;
            # downgrade to flag for pure entropy.
            effective_action = self.action
            if effective_action == "suppress":
                effective_action = "flag"
            return Decision(
                action=effective_action,
                reason=f"Entropy: {entropy:.3f} > threshold → {effective_action.upper()}",
                entropy=entropy,
            )

        # ------------------------------------------------------------------
        # No signal
        # ------------------------------------------------------------------
        return Decision(
            action="pass",
            reason=f"Pass: entropy={entropy:.3f}, no conflict",
            entropy=entropy,
        )


# =============================================================================
# 4. Generation output
# =============================================================================

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
    # Conflict detection results (post-hoc and real-time)
    conflict_records: List[List[Dict]] = field(default_factory=list)
    conflict_scores: List[List[float]] = field(default_factory=list)
    primary_labels: List[List[str]] = field(default_factory=list)
    secondary_labels: List[List[Optional[str]]] = field(default_factory=list)
    # Explainability (new)
    per_token_decisions: List[List[Dict[str, Any]]] = field(default_factory=list)


# =============================================================================
# 5. ACC Logits Processor
# =============================================================================

class _ACCLogitsProcessor(LogitsProcessor):
    """LogitsProcessor that monitors entropy and optionally runs real-time
    conflict detection during generation.

    Called *inside* the HF ``.generate()`` loop, after the model forward
    pass but before token selection. It can modify logits (regenerate,
    suppress) and records intervention decisions for later explainability.
    """

    def __init__(
        self,
        monitor: EntropyMonitor,
        tokenizer,
        decision_engine: UnifiedDecisionEngine,
        detector: Optional[PredictiveCodingDetector] = None,
        extractor: Optional[MultiLayerGenerationExtractor] = None,
        max_regenerations: int = 3,
    ):
        self.monitor = monitor
        self.tokenizer = tokenizer
        self.decision_engine = decision_engine
        self.detector = detector
        self.extractor = extractor
        self.max_regenerations = int(max_regenerations)

        # Per-batch-row tracking
        self.per_token_entropy: List[List[float]] = []
        self.uncertain_steps: List[List[int]] = []
        self.text_inserts: List[List[Tuple[int, str]]] = []
        self.regen_counts: List[int] = []
        self.decisions: List[List[Decision]] = []
        self._detector_states: List[Optional[torch.Tensor]] = []
        self.prompt_len: int = 0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
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
            self.decisions = [[] for _ in range(batch_size)]
            self._detector_states = [None] * batch_size
            self.prompt_len = input_ids.shape[1]

        gen_step = input_ids.shape[1] - self.prompt_len
        next_scores = scores.clone()

        for b in range(batch_size):
            row_logits = scores[b]

            # -------------------------------------------------------------
            # 1. Entropy
            # -------------------------------------------------------------
            entropy = self.monitor.observe(row_logits)
            self.per_token_entropy[b].append(entropy)
            is_uncertain = self.monitor.check_threshold(entropy)
            if is_uncertain:
                self.uncertain_steps[b].append(gen_step)

            # -------------------------------------------------------------
            # 2. Real-time conflict detection
            # -------------------------------------------------------------
            conflict_score: Optional[float] = None
            primary: Optional[str] = None
            secondary: Optional[str] = None

            if (
                self.detector is not None
                and self.extractor is not None
                and gen_step > 0  # skip first step: no generated token yet
            ):
                try:
                    hs_dict = self._get_hidden_states_for_step(gen_step, b, scores.device)
                    if hs_dict:
                        with torch.no_grad():
                            p_logits, s_logits, c_score, next_state = (
                                self.detector.forward(
                                    hs_dict,
                                    prev_state=self._detector_states[b],
                                )
                            )
                        self._detector_states[b] = next_state

                        primary_idx = int(p_logits.argmax(dim=-1).item())
                        primary = self.detector.PRIMARY_LABELS[primary_idx]
                        conflict_score = float(c_score.item())

                        if primary == "unsupported":
                            secondary_idx = int(s_logits.argmax(dim=-1).item())
                            secondary = self.detector.SECONDARY_LABELS[secondary_idx]
                except Exception as exc:
                    logger.warning(
                        "Real-time conflict detection failed at step %d: %s",
                        gen_step,
                        exc,
                    )

            # -------------------------------------------------------------
            # 3. Unified decision
            # -------------------------------------------------------------
            decision = self.decision_engine.decide(
                entropy=entropy,
                is_uncertain=is_uncertain,
                conflict_score=conflict_score,
                primary=primary,
                secondary=secondary,
            )
            self.decisions[b].append(decision)

            # -------------------------------------------------------------
            # 4. Apply action
            # -------------------------------------------------------------
            if decision.action == "regenerate":
                if self.regen_counts[b] < self.max_regenerations:
                    multiplier = (
                        self.decision_engine.regen_multiplier
                        ** (self.regen_counts[b] + 1)
                    )
                    next_scores[b] = row_logits * multiplier
                    self.regen_counts[b] += 1
            elif decision.action == "flag":
                marker = self.decision_engine.marker_config.get_marker(
                    primary, secondary
                )
                self.text_inserts[b].append(
                    (self.prompt_len + gen_step + 1, marker)
                )
            elif decision.action == "warning":
                prefix = self.decision_engine.marker_config.warning_prefix
                self.text_inserts[b].append(
                    (self.prompt_len + gen_step, prefix)
                )
            elif decision.action == "suppress":
                # Block the top-1 token to force selection of an alternative
                top_token = int(next_scores[b].argmax().item())
                next_scores[b, top_token] = -float("inf")

        return next_scores

    def _get_hidden_states_for_step(
        self,
        gen_step: int,
        batch_idx: int,
        target_device: torch.device,
    ) -> Optional[Dict[int, torch.Tensor]]:
        """Build a hidden-state dict for the given batch item and step."""
        hs_dict: Dict[int, torch.Tensor] = {}
        for layer_idx in self.extractor.layer_indices:
            buf = self.extractor._hidden_buffers[layer_idx]
            if gen_step < len(buf):
                # buf[gen_step] shape: (batch, 1, hidden_dim)
                hidden_vec = buf[gen_step][batch_idx, 0, :].to(
                    target_device, dtype=torch.float32
                )
                hs_dict[layer_idx] = hidden_vec
        return hs_dict if hs_dict else None


# =============================================================================
# 6. Legacy entropy-only processor (backward compatibility)
# =============================================================================

class _EntropyLogitsProcessor(_ACCLogitsProcessor):
    """Entropy-only logits processor (deprecated — use _ACCLogitsProcessor).

    Maintained for backward compatibility with existing call sites.
    """

    def __init__(
        self,
        monitor: EntropyMonitor,
        tokenizer,
        action: Action = "flag",
        regen_multiplier: float = 2.0,
        max_regenerations: int = 3,
    ):
        engine = UnifiedDecisionEngine(
            action=action,
            regen_multiplier=regen_multiplier,
        )
        super().__init__(
            monitor=monitor,
            tokenizer=tokenizer,
            decision_engine=engine,
            max_regenerations=max_regenerations,
        )


# =============================================================================
# 7. Self-consistency checker
# =============================================================================

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
            "outlier_indices": outlier_mask.nonzero(as_tuple=False).squeeze(-1).tolist(),
            "candidates": candidates,
        }


# =============================================================================
# 8. ACC Enhanced Generator
# =============================================================================

class ACCEnhancedGenerator:
    """Wraps a HuggingFace causal LM with entropy monitoring and optional
    real-time conflict detection.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The base causal language model.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer matching ``model``.
    action : {"flag", "regenerate", "warning", "suppress"}
        Default intervention action for high-uncertainty tokens.
    threshold : float
        Entropy threshold (interpretation depends on ``mode``).
    mode : {"absolute", "moving_average", "percentile"}
        Threshold strategy for entropy monitoring.
    window_size : int
        Sliding window length for temporal smoothing.
    warmup : int
        Minimum tokens before moving-average / percentile modes evaluate.
    regen_multiplier : float
        Temperature-dividing multiplier on regenerate.
    max_regenerations : int
        Max number of times a single position can be regenerated.
    use_self_consistency : bool
        Enable post-hoc self-consistency checking.
    self_consistency_candidates : int
        Number of candidate continuations for self-consistency.
    self_consistency_threshold : float
        Cosine-similarity threshold for clustering candidates.
    self_consistency_max_new_tokens : int
        Max new tokens per candidate continuation.
    use_conflict_detector : bool
        Enable conflict detection (post-hoc by default; real-time when
        ``use_realtime_conflict_detector=True``).
    use_realtime_conflict_detector : bool
        If True, the detector is wired into the logits processor for
        per-token intervention. If False (default), detection runs only
        after generation completes (post-hoc).
    conflict_detector : PredictiveCodingDetector, optional
        Pre-trained detector instance. If None and conflict detection is
        enabled, a detector must be provided later or generation will skip
        conflict detection with a warning.
    conflict_layer_indices : List[int], optional
        Layers to tap for hidden-state extraction.
    decision_engine : UnifiedDecisionEngine, optional
        Custom decision engine. If None, one is built from the other
        constructor arguments.
    marker_config : MarkerConfig, optional
        Custom marker configuration.
    """

    def __init__(
        self,
        model,
        tokenizer,
        action: Action = "flag",
        threshold: float = 1.5,
        mode: ThresholdMode = "absolute",
        window_size: int = 32,
        warmup: int = 4,
        regen_multiplier: float = 2.0,
        max_regenerations: int = 3,
        use_self_consistency: bool = False,
        self_consistency_candidates: int = 5,
        self_consistency_threshold: float = 0.75,
        self_consistency_max_new_tokens: Optional[int] = None,
        use_conflict_detector: bool = False,
        use_realtime_conflict_detector: bool = False,
        conflict_detector: Optional[PredictiveCodingDetector] = None,
        conflict_layer_indices: Optional[List[int]] = None,
        decision_engine: Optional[UnifiedDecisionEngine] = None,
        marker_config: Optional[MarkerConfig] = None,
    ):
        if action not in ("flag", "regenerate", "warning", "suppress"):
            raise ValueError(f"unknown action: {action}")
        self.model = model
        self.tokenizer = tokenizer
        self.action: Action = action
        self.regen_multiplier = float(regen_multiplier)
        self.max_regenerations = int(max_regenerations)
        self.device = getattr(model, "device", None) or next(model.parameters()).device

        # Build or accept decision engine
        if decision_engine is not None:
            self.decision_engine = decision_engine
        else:
            mc = marker_config or MarkerConfig()
            self.decision_engine = UnifiedDecisionEngine(
                entropy_threshold=threshold,
                action=action,
                regen_multiplier=regen_multiplier,
                marker_config=mc,
            )

        self.monitor = EntropyMonitor(
            threshold=threshold,
            mode=mode,
            action=action,
            window_size=window_size,
            warmup=warmup,
        )

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

        # Conflict detector integration
        self.use_conflict_detector = bool(use_conflict_detector)
        self.use_realtime_conflict_detector = bool(use_realtime_conflict_detector)
        self.conflict_detector = conflict_detector
        self.conflict_layer_indices = conflict_layer_indices or [-1, -4, -8, -12]
        self._conflict_extractor: Optional[MultiLayerGenerationExtractor] = None

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
        """Entropy-aware generation using the standard HF ``.generate()`` pipeline."""
        if input_ids is None:
            raise ValueError("input_ids is required")
        if kwargs:
            logger.debug("ACCEnhancedGenerator ignoring kwargs: %s", list(kwargs))

        self.monitor.reset()

        # Build logits processors
        logits_processor_list: List[LogitsProcessor] = []

        # Conflict detection setup: always attach extractor if detector is enabled
        # so hidden states are captured during the main generation pass.
        use_rt = (
            self.use_realtime_conflict_detector
            and self.conflict_detector is not None
        )
        if self.conflict_detector is not None:
            self._conflict_extractor = MultiLayerGenerationExtractor(
                self.model, layer_indices=self.conflict_layer_indices
            )
            logits_processor_list.append(self._conflict_extractor)

        # Main ACC processor
        acc_processor = _ACCLogitsProcessor(
            monitor=self.monitor,
            tokenizer=self.tokenizer,
            decision_engine=self.decision_engine,
            detector=self.conflict_detector if use_rt else None,
            extractor=self._conflict_extractor if use_rt else None,
            max_regenerations=self.max_regenerations,
        )
        logits_processor_list.append(acc_processor)

        try:
            outputs = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device)
                if attention_mask is not None
                else None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                do_sample=do_sample,
                pad_token_id=pad_token_id
                if pad_token_id is not None
                else self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                logits_processor=logits_processor_list,
                return_dict_in_generate=True,
                output_scores=True,
            )

            sequences = outputs.sequences
            batch_size = sequences.shape[0]
            prompt_len = input_ids.shape[1]

            # --------------------------------------------------------------
            # Post-hoc conflict detection (used when real-time is disabled
            # or as a fallback for batch_size > 1 with real-time)
            # --------------------------------------------------------------
            conflict_records: List[List[Dict]] = [[] for _ in range(batch_size)]
            conflict_scores: List[List[float]] = [[] for _ in range(batch_size)]
            primary_labels: List[List[str]] = [[] for _ in range(batch_size)]
            secondary_labels: List[List[Optional[str]]] = [[] for _ in range(batch_size)]

            if (
                self.use_conflict_detector
                and self.conflict_detector is not None
                and not use_rt
            ):
                # Post-hoc: use hidden states captured during the main generation pass
                # (extractor is now always attached when detector is enabled)
                for b in range(batch_size):
                    try:
                        records = self._conflict_extractor.get_records(
                            sequences[b : b + 1], prompt_len=prompt_len
                        )
                        conflict_records[b] = records

                        hidden_sequence = []
                        for rec in records:
                            hs = {
                                layer_idx: torch.tensor(hidden_vec, dtype=torch.float32)
                                for layer_idx, hidden_vec in rec["hidden_states"].items()
                            }
                            hidden_sequence.append(hs)

                        if hidden_sequence:
                            with torch.no_grad():
                                results = self.conflict_detector.predict_sequence(
                                    hidden_sequence
                                )
                            for r in results:
                                primary_labels[b].append(r["primary"])
                                secondary_labels[b].append(r["secondary"])
                                conflict_scores[b].append(r["conflict_score"])
                    except Exception as exc:
                        logger.warning(
                            "Post-hoc conflict detection failed for batch item %d: %s",
                            b,
                            exc,
                        )
            elif use_rt and self._conflict_extractor is not None:
                # Real-time mode: build post-hoc labels from decisions
                for b in range(batch_size):
                    for dec in acc_processor.decisions[b]:
                        primary_labels[b].append(dec.primary or "supported")
                        secondary_labels[b].append(dec.secondary)
                        conflict_scores[b].append(dec.conflict_score or 0.0)
        finally:
            # Always clean up hooks, even if generation or post-processing failed
            if self._conflict_extractor is not None:
                self._conflict_extractor.remove_hooks()
                self._conflict_extractor = None

        # Build conflict inserts from post-hoc / real-time labels
        conflict_inserts: List[List[Tuple[int, str]]] = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            for i, (primary, secondary) in enumerate(
                zip(primary_labels[b], secondary_labels[b])
            ):
                pos = prompt_len + i + 1  # after the token
                if primary == "unsupported" and secondary == "hallucinated":
                    conflict_inserts[b].append((pos, HALLUCINATION_MARKER))
                elif primary == "unsupported" and secondary == "contradictory":
                    conflict_inserts[b].append((pos, CONTRADICTION_MARKER))
                elif primary == "uncertain":
                    conflict_inserts[b].append((pos, FLAG_MARKER))

        # Merge entropy/conflict inserts with real-time processor inserts
        all_inserts: List[List[Tuple[int, str]]] = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            merged = acc_processor.text_inserts[b] + conflict_inserts[b]
            # Deduplicate by position+marker
            seen = set()
            deduped = []
            for pos, marker in merged:
                key = (pos, marker)
                if key not in seen:
                    seen.add(key)
                    deduped.append((pos, marker))
            all_inserts[b] = sorted(deduped, key=lambda x: x[0])

        # Decode with markers
        text = self._decode_with_markers(sequences, prompt_len, all_inserts)

        # Self-consistency check
        consistency_scores: List[Optional[float]] = [None] * batch_size
        contradiction_flags: List[Optional[bool]] = [None] * batch_size
        if self.use_self_consistency and self.self_consistency_checker is not None:
            for b in range(batch_size):
                prompt_text = self.tokenizer.decode(
                    input_ids[b], skip_special_tokens=True
                )
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
                    logger.warning(
                        "Self-consistency check failed for batch item %d: %s",
                        b,
                        exc,
                    )

        if not return_dict_in_generate:
            return sequences

        # Convert decisions to plain dicts for serialization
        per_token_decisions: List[List[Dict[str, Any]]] = [
            [asdict(d) for d in batch_decisions]
            for batch_decisions in acc_processor.decisions
        ]

        return ACCGenerationOutput(
            sequences=sequences,
            text=text,
            per_token_entropy=acc_processor.per_token_entropy,
            uncertain_steps=acc_processor.uncertain_steps,
            events=[[asdict(e) for e in self.monitor.events] for _ in range(batch_size)],
            regenerations=acc_processor.regen_counts,
            confidence_score=[self.monitor.get_confidence_score()] * batch_size,
            consistency_score=consistency_scores,
            contradiction_detected=contradiction_flags,
            scores=outputs.scores,
            conflict_records=conflict_records,
            conflict_scores=conflict_scores,
            primary_labels=primary_labels,
            secondary_labels=secondary_labels,
            per_token_decisions=per_token_decisions,
        )

    def generate_from_prompt(
        self, prompt: str, return_dict_in_generate: bool = True, **gen_kwargs
    ) -> Union[torch.LongTensor, ACCGenerationOutput]:
        """Convenience wrapper: tokenize a string prompt then call ``generate``."""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return self.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            return_dict_in_generate=return_dict_in_generate,
            **gen_kwargs,
        )

    def explain_decisions(
        self, output: ACCGenerationOutput, batch_idx: int = 0
    ) -> str:
        """Pretty-print the per-token decision trace for a batch item.

        Parameters
        ----------
        output : ACCGenerationOutput
            The generation output to explain.
        batch_idx : int
            Which batch item to explain.

        Returns
        -------
        str
            Human-readable decision trace.
        """
        if batch_idx >= len(output.per_token_decisions):
            return "No decisions available."

        lines = [
            f"Decision trace for batch item {batch_idx}",
            "=" * 60,
        ]
        for step, dec in enumerate(output.per_token_decisions[batch_idx]):
            action = dec.get("action", "pass")
            entropy = dec.get("entropy", 0.0)
            cs = dec.get("conflict_score")
            primary = dec.get("primary")
            secondary = dec.get("secondary")
            reason = dec.get("reason", "")

            cs_str = f"CS={cs:.3f}" if cs is not None else "CS=N/A"
            primary_str = f"P={primary}" if primary else "P=N/A"
            secondary_str = f"S={secondary}" if secondary else "S=N/A"

            lines.append(
                f"Step {step:3d} | {action:12s} | H={entropy:.3f} | "
                f"{cs_str} | {primary_str} | {secondary_str}"
            )
            if reason:
                lines.append(textwrap.indent(f"→ {reason}", "          "))

        # Summarise
        actions = [d.get("action", "pass") for d in output.per_token_decisions[batch_idx]]
        flag_count = actions.count("flag")
        regen_count = actions.count("regenerate")
        suppress_count = actions.count("suppress")
        warn_count = actions.count("warning")

        lines.append("=" * 60)
        lines.append(
            f"Summary: {flag_count} flags, {regen_count} regenerations, "
            f"{suppress_count} suppressions, {warn_count} warnings"
        )
        return "\n".join(lines)

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
            pieces: List[str] = []
            cursor = prompt_len
            for pos, marker in sorted(text_inserts[b], key=lambda x: x[0]):
                pos = min(max(pos, cursor), len(seq))
                if pos > cursor:
                    pieces.append(
                        self.tokenizer.decode(seq[cursor:pos], skip_special_tokens=True)
                    )
                pieces.append(marker)
                cursor = pos
            if cursor < len(seq):
                pieces.append(
                    self.tokenizer.decode(seq[cursor:], skip_special_tokens=True)
                )
            out.append("".join(pieces))
        return out
