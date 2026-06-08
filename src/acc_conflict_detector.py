"""ACC-inspired conflict detection using Predictive Coding.

This module implements a neuroscience-backed approach to hallucination/conflict
detection based on Hierarchical Predictive Coding theory (Rao & Ballard 1999;
Friston 2005). The core idea:

  1. The brain is a hierarchical prediction machine.
  2. Each cortical level predicts the representation at the level below.
  3. Prediction errors (surprise) are transmitted upward.
  4. The Anterior Cingulate Cortex (ACC) monitors high-level prediction errors
     and signals for cognitive control when they are large.

In our LLM analog:
  - Early transformer layers = "sensory" predictions (syntax, local coherence)
  - Late transformer layers = "cognitive" predictions (semantics, world knowledge)
  - Hallucinations create cross-layer surprise: late layers predict something
    that early layers cannot support.
  - Our detector computes explicit prediction errors between hierarchical layers
    and classifies the resulting surprise pattern.

Architecture:
  PredictiveCodingDetector:
    - HierarchicalPredictionErrorModule: computes ||h_{l+1} - pred(h_l)||^2
    - Leaky integrator: temporal integration with exponential decay (biologically
      plausible, unlike LSTM gating)
    - Primary head (3-way): supported / unsupported / uncertain
    - Secondary head (2-way): hallucinated / contradictory (only on unsupported)
    - Conflict score head: continuous [0,1] scalar for threshold-tunable intervention

The module also retains the legacy LatentConflictDetector (2-layer MLP) and
single-layer extractors for backward compatibility.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, LogitsProcessor


# =============================================================================
# 1. Hierarchical Prediction Error Module
# =============================================================================

class HierarchicalPredictionErrorModule(nn.Module):
    """Compute prediction errors between adjacent transformer layers.

    For each adjacent pair of tapped layers (source -> target), a small MLP
    predicts the target representation from the source. The prediction error
    (squared L2 distance) is the "surprise" signal that the ACC would monitor.

    Args:
        hidden_dim: Dimensionality of hidden states (e.g., 4096 for Mistral 7B).
        layer_pairs: List of (source_layer_idx, target_layer_idx) tuples.
            These must be adjacent or near-adjacent layers in the model.
            Example for 4 layers: [(-12, -8), (-8, -4), (-4, -1)].
        predictor_hidden_mult: Multiplier for predictor MLP hidden dim.
            Default 0.5 means predictor is Linear(hidden_dim -> hidden_dim//2 -> hidden_dim).
    """

    def __init__(
        self,
        hidden_dim: int,
        layer_pairs: List[Tuple[int, int]],
        predictor_bottleneck_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_pairs = layer_pairs
        self.num_pairs = len(layer_pairs)

        # One predictor MLP per layer pair
        # Default bottleneck: hidden_dim // 8 (e.g., 512 for 4096-dim models)
        # This keeps total params ~3-6M depending on number of pairs
        mid_dim = predictor_bottleneck_dim or max(64, hidden_dim // 8)
        predictors = []
        for _ in layer_pairs:
            predictors.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, mid_dim),
                    nn.LayerNorm(mid_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mid_dim, hidden_dim),
                )
            )
        self.predictors = nn.ModuleList(predictors)

    def forward(
        self,
        hidden_states: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute prediction errors for all layer pairs.

        Args:
            hidden_states: Dict mapping layer_idx -> tensor of shape (batch, hidden_dim).
                All layers in ``layer_pairs`` must be present.

        Returns:
            errors: Tensor of shape (batch, num_pairs) with squared L2 prediction errors.
        """
        batch_size = None
        for h in hidden_states.values():
            batch_size = h.shape[0]
            break
        if batch_size is None:
            raise ValueError("hidden_states is empty")

        device = next(self.parameters()).device
        errors = []

        for idx, (src_layer, tgt_layer) in enumerate(self.layer_pairs):
            if src_layer not in hidden_states:
                raise KeyError(f"Source layer {src_layer} not in hidden_states. Available: {list(hidden_states.keys())}")
            if tgt_layer not in hidden_states:
                raise KeyError(f"Target layer {tgt_layer} not in hidden_states. Available: {list(hidden_states.keys())}")

            src = hidden_states[src_layer].to(device)  # (batch, hidden_dim)
            tgt = hidden_states[tgt_layer].to(device)  # (batch, hidden_dim)

            pred = self.predictors[idx](src)  # (batch, hidden_dim)
            # Squared L2 error per sample, averaged over hidden dim
            error = F.mse_loss(pred, tgt, reduction="none").mean(dim=-1)  # (batch,)
            errors.append(error)

        return torch.stack(errors, dim=-1)  # (batch, num_pairs)


# =============================================================================
# 2. Predictive Coding Conflict Detector
# =============================================================================

class PredictiveCodingDetector(nn.Module):
    """Neuroscience-backed conflict detector using hierarchical prediction errors.

    This detector replaces the flat 2-layer MLP with a biologically plausible
    architecture grounded in Predictive Coding theory:
      - Prediction errors between hierarchical layers (like cortical columns)
      - Leaky temporal integration (like neural population dynamics)
      - Hierarchical classification (primary 3-way + secondary 2-way)
      - Continuous conflict score for threshold-tunable interventions

    Primary labels (real-time intervention):
        supported    -> continue normally
        unsupported  -> run secondary head + trigger intervention
        uncertain    -> raise temperature, continue with warning

    Secondary labels (only on unsupported tokens):
        hallucinated -> regenerate or abstain
        contradictory -> request clarification

    Args:
        hidden_dim: Dimension of LLM hidden states.
        layer_pairs: Adjacent layer pairs for prediction error computation.
        temporal_decay: Decay constant alpha for leaky integrator (0 = no memory,
            1 = infinite memory). Default 0.7. Biologically plausible values: 0.5-0.9.
        dropout: Dropout rate for classification heads.
    """

    PRIMARY_LABELS = ["supported", "unsupported", "uncertain"]
    SECONDARY_LABELS = ["hallucinated", "contradictory"]
    ALL_LABELS = PRIMARY_LABELS + SECONDARY_LABELS  # For compatibility with old code

    def __init__(
        self,
        hidden_dim: int = 4096,
        layer_pairs: Optional[List[Tuple[int, int]]] = None,
        temporal_decay: float = 0.7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Default layer pairs: deep hierarchy from early to late layers
        if layer_pairs is None:
            layer_pairs = [(-12, -8), (-8, -4), (-4, -1)]
        self.layer_pairs = layer_pairs
        self.num_pairs = len(self.layer_pairs)
        self.temporal_decay = float(temporal_decay)

        if self.num_pairs == 0:
            raise ValueError("At least one layer pair is required")
        if not (0.0 <= self.temporal_decay <= 1.0):
            raise ValueError(f"temporal_decay must be in [0, 1], got {self.temporal_decay}")

        # ------------------------------------------------------------------
        # Prediction error module
        # ------------------------------------------------------------------
        self.prediction_error = HierarchicalPredictionErrorModule(
            hidden_dim=hidden_dim,
            layer_pairs=self.layer_pairs,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Classification heads (lightweight, like cortical readout layers)
        # ------------------------------------------------------------------
        pe_dim = self.num_pairs

        # Primary head: 3-way (supported / unsupported / uncertain)
        self.primary_head = nn.Sequential(
            nn.Linear(pe_dim, pe_dim),
            nn.LayerNorm(pe_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pe_dim, len(self.PRIMARY_LABELS)),
        )

        # Secondary head: 2-way (hallucinated / contradictory)
        self.secondary_head = nn.Sequential(
            nn.Linear(pe_dim, pe_dim),
            nn.LayerNorm(pe_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pe_dim, len(self.SECONDARY_LABELS)),
        )

        # Conflict score head: continuous scalar in [0, 1]
        self.conflict_score_head = nn.Sequential(
            nn.Linear(pe_dim, max(1, pe_dim // 2)),
            nn.LayerNorm(max(1, pe_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, pe_dim // 2), 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: Dict[int, torch.Tensor],
        prev_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass computing prediction errors and classifications.

        Args:
            hidden_states: Dict mapping layer_idx -> (batch, hidden_dim).
            prev_state: Previous leaky integrator state, shape (batch, num_pairs).
                None on first timestep.

        Returns:
            primary_logits: (batch, 3)
            secondary_logits: (batch, 2)
            conflict_score: (batch, 1) in [0, 1]
            next_state: (batch, num_pairs) for next timestep
        """
        # 1. Compute hierarchical prediction errors
        pe = self.prediction_error(hidden_states)  # (batch, num_pairs)

        # 2. Leaky temporal integration (biologically plausible)
        if prev_state is not None:
            # Ensure same device
            prev_state = prev_state.to(pe.device)
            state = self.temporal_decay * prev_state + (1 - self.temporal_decay) * pe
        else:
            state = pe

        # 3. Classification heads
        primary_logits = self.primary_head(state)          # (batch, 3)
        secondary_logits = self.secondary_head(state)      # (batch, 2)
        conflict_score = self.conflict_score_head(state)   # (batch, 1)

        return primary_logits, secondary_logits, conflict_score, state

    def classify(
        self,
        hidden_states: Dict[int, torch.Tensor],
        prev_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """Classify a single timestep and return human-readable results.

        Args:
            hidden_states: Dict mapping layer_idx -> (hidden_dim,) or (1, hidden_dim).
            prev_state: Optional previous state.

        Returns:
            Dict with keys:
                - "primary": str, argmax primary label
                - "primary_probs": Dict[str, float]
                - "secondary": str or None
                - "secondary_probs": Dict[str, float] or None
                - "conflict_score": float in [0, 1]
                - "next_state": Tensor for chaining
        """
        # Ensure batch dim
        hs = {}
        for k, v in hidden_states.items():
            if v.dim() == 1:
                v = v.unsqueeze(0)
            hs[k] = v

        with torch.no_grad():
            primary_logits, secondary_logits, conflict_score, next_state = self.forward(
                hs, prev_state
            )

        primary_probs = F.softmax(primary_logits, dim=-1).squeeze(0)
        secondary_probs = F.softmax(secondary_logits, dim=-1).squeeze(0)

        primary_idx = int(primary_logits.argmax(dim=-1).item())
        primary_label = self.PRIMARY_LABELS[primary_idx]

        result: Dict[str, Union[str, float, Dict[str, float], torch.Tensor, None]] = {
            "primary": primary_label,
            "primary_probs": {
                label: float(primary_probs[i].item())
                for i, label in enumerate(self.PRIMARY_LABELS)
            },
            "conflict_score": float(conflict_score.item()),
            "next_state": next_state,
        }

        # Only run secondary classification for unsupported tokens
        if primary_label == "unsupported":
            secondary_idx = int(secondary_logits.argmax(dim=-1).item())
            secondary_label = self.SECONDARY_LABELS[secondary_idx]
            result["secondary"] = secondary_label
            result["secondary_probs"] = {
                label: float(secondary_probs[i].item())
                for i, label in enumerate(self.SECONDARY_LABELS)
            }
        else:
            result["secondary"] = None
            result["secondary_probs"] = None

        return result

    def predict_sequence(
        self,
        hidden_sequence: List[Dict[int, torch.Tensor]],
    ) -> List[Dict]:
        """Process a sequence of hidden states (one per generation step).

        Args:
            hidden_sequence: List of Dict[int, Tensor], one per timestep.

        Returns:
            List of classification dicts (same format as ``classify``).
        """
        results = []
        state = None
        for hs in hidden_sequence:
            result = self.classify(hs, prev_state=state)
            state = result["next_state"]
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Interpretability hooks
    # ------------------------------------------------------------------

    def get_prediction_errors(
        self,
        hidden_states: Dict[int, torch.Tensor],
    ) -> Dict[Tuple[int, int], float]:
        """Get raw prediction errors for each layer pair (no classification).

        Args:
            hidden_states: Dict mapping layer_idx -> (batch, hidden_dim) or (hidden_dim,).

        Returns:
            Dict mapping (src_layer, tgt_layer) -> scalar prediction error.
        """
        hs = {}
        for k, v in hidden_states.items():
            if v.dim() == 1:
                v = v.unsqueeze(0)
            hs[k] = v

        with torch.no_grad():
            pe = self.prediction_error(hs)  # (batch, num_pairs)

        result = {}
        for idx, pair in enumerate(self.layer_pairs):
            result[pair] = float(pe[0, idx].item())
        return result

    def get_layer_contributions(
        self,
        hidden_states: Dict[int, torch.Tensor],
        target: str = "conflict_score",
    ) -> Dict[Tuple[int, int], float]:
        """Compute gradient-based contribution of each layer pair to a target output.

        Uses simple gradient attribution: how much does each prediction error
        dimension influence the target output?

        Args:
            hidden_states: Dict mapping layer_idx -> tensor.
            target: Which output to attribute. One of:
                - "conflict_score": the continuous conflict score
                - "primary_<label>": primary class logit (e.g., "primary_unsupported")
                - "secondary_<label>": secondary class logit (e.g., "secondary_hallucinated")

        Returns:
            Dict mapping (src_layer, tgt_layer) -> attribution score (higher = more influential).
        """
        hs = {}
        for k, v in hidden_states.items():
            if v.dim() == 1:
                v = v.unsqueeze(0)
            hs[k] = v.clone().requires_grad_(False)

        # Enable gradients for prediction error computation
        pe = self.prediction_error(hs)  # (1, num_pairs)
        pe.requires_grad_(True)
        pe.retain_grad()

        # Forward through heads using detached prediction errors
        primary_logits = self.primary_head(pe)
        secondary_logits = self.secondary_head(pe)
        conflict_score = self.conflict_score_head(pe)

        # Determine target scalar
        if target == "conflict_score":
            scalar = conflict_score[0, 0]
        elif target.startswith("primary_"):
            label = target.replace("primary_", "")
            idx = self.PRIMARY_LABELS.index(label)
            scalar = primary_logits[0, idx]
        elif target.startswith("secondary_"):
            label = target.replace("secondary_", "")
            idx = self.SECONDARY_LABELS.index(label)
            scalar = secondary_logits[0, idx]
        else:
            raise ValueError(f"Unknown target: {target}")

        # Compute gradients w.r.t. prediction errors without modifying model params
        # Use torch.autograd.grad to avoid accumulating gradients on module weights
        grads = torch.autograd.grad(scalar, pe, create_graph=False)[0]

        # Attribution: gradient * activation (simple gradient attribution)
        attributions = (grads * pe).abs().squeeze(0)  # (num_pairs,)

        result = {}
        for idx, pair in enumerate(self.layer_pairs):
            result[pair] = float(attributions[idx].item())
        return result

    def explain(
        self,
        hidden_states: Dict[int, torch.Tensor],
        prev_state: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Comprehensive explanation of a single classification decision.

        Returns a dict with all interpretable components:
            - classification: the standard classify() output
            - prediction_errors: raw PE per layer pair
            - layer_contributions: attribution to conflict_score
            - state_before: integrator state before this step
            - state_after: integrator state after this step
        """
        hs = {}
        for k, v in hidden_states.items():
            if v.dim() == 1:
                v = v.unsqueeze(0)
            hs[k] = v

        with torch.no_grad():
            pe = self.prediction_error(hs)
            if prev_state is not None:
                prev_state = prev_state.to(pe.device)
                state = self.temporal_decay * prev_state + (1 - self.temporal_decay) * pe
            else:
                state = pe

        classification = self.classify(hidden_states, prev_state=prev_state)
        prediction_errors = self.get_prediction_errors(hidden_states)
        layer_contributions = self.get_layer_contributions(hidden_states, target="conflict_score")

        # Primary label contribution
        primary_target = f"primary_{classification['primary']}"
        primary_contributions = self.get_layer_contributions(hidden_states, target=primary_target)

        # Secondary label contribution (if applicable)
        secondary_contributions = None
        if classification["secondary"] is not None:
            secondary_target = f"secondary_{classification['secondary']}"
            secondary_contributions = self.get_layer_contributions(
                hidden_states, target=secondary_target
            )

        return {
            "classification": classification,
            "prediction_errors": prediction_errors,
            "layer_contributions": {
                "conflict_score": layer_contributions,
                "primary": primary_contributions,
                "secondary": secondary_contributions,
            },
            "state_before": prev_state.cpu().tolist() if prev_state is not None else None,
            "state_after": state.cpu().tolist(),
            "prediction_error_vector": pe.cpu().tolist(),
        }


# =============================================================================
# 3. Multi-Layer Hidden State Extractor
# =============================================================================

class MultiLayerGenerationExtractor(LogitsProcessor):
    """Extracts hidden states from **multiple layers** during ``model.generate()``.

    This is the multi-layer successor to ``GenerationHiddenStateExtractor``.
    It registers one forward hook per tapped layer and buffers hidden states
    for all layers simultaneously. Each record in the output contains the
    hidden states from all tapped layers for a single generated token.

    Usage::

        extractor = MultiLayerGenerationExtractor(model, layer_indices=[-1, -4, -8, -12])
        outputs = model.generate(
            input_ids,
            max_new_tokens=50,
            logits_processor=[extractor],
            return_dict_in_generate=True,
            output_scores=True,
        )
        records = extractor.get_records(
            outputs.sequences, prompt_len=input_ids.shape[1]
        )
        # Each record has:
        #   record["hidden_states"] = { -1: [...], -4: [...], -8: [...], -12: [...] }

    Args:
        model: A HuggingFace causal LM.
        layer_indices: Which layers to tap. Negative indices count from the end.
            Default [-1, -4, -8, -12] captures the deep hierarchy.
    """

    def __init__(self, model: AutoModelForCausalLM, layer_indices: List[int] = None):
        super().__init__()
        self.model = model
        self.layer_indices = layer_indices or [-1, -4, -8, -12]
        self.num_layers_total = self._count_model_layers()

        # Normalize negative indices to positive for hook registration
        self._positive_indices = []
        for idx in self.layer_indices:
            if idx < 0:
                pos_idx = self.num_layers_total + idx
            else:
                pos_idx = idx
            if pos_idx < 0 or pos_idx >= self.num_layers_total:
                raise ValueError(
                    f"Layer index {idx} (positive: {pos_idx}) is out of range "
                    f"for model with {self.num_layers_total} layers."
                )
            self._positive_indices.append(pos_idx)

        # Buffers: one list per layer
        self._hidden_buffers: Dict[int, List[torch.Tensor]] = {
            idx: [] for idx in self.layer_indices
        }
        self._hooks: List = []
        self._prompt_len = 0
        self._step = 0

        self.register_hooks()

    def _count_model_layers(self) -> int:
        """Infer total number of transformer layers."""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return len(self.model.transformer.h)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder') and hasattr(self.model.model.decoder, 'layers'):
            return len(self.model.model.decoder.layers)
        else:
            raise ValueError(
                "Unsupported model architecture. Could not find transformer layers. "
                "Expected model.transformer.h (GPT-2), model.model.layers (Llama/Mistral), "
                "or model.model.decoder.layers (OPT)."
            )

    def _get_target_layer(self, pos_idx: int):
        """Get the layer module at a positive index."""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h[pos_idx]
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[pos_idx]
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder') and hasattr(self.model.model.decoder, 'layers'):
            return self.model.model.decoder.layers[pos_idx]
        else:
            raise ValueError("Unsupported model architecture")

    def register_hooks(self):
        """Register one forward hook per tapped layer."""
        for pos_idx, user_idx in zip(self._positive_indices, self.layer_indices):
            def make_hook(layer_key):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    # Capture last position (newly generated token)
                    last_hidden = output[:, -1:, :].detach().cpu()
                    self._hidden_buffers[layer_key].append(last_hidden)
                return hook_fn

            target_layer = self._get_target_layer(pos_idx)
            hook = target_layer.register_forward_hook(make_hook(user_idx))
            self._hooks.append(hook)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------ LogitsProcessor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._step == 0:
            self._prompt_len = input_ids.shape[1]
        self._step += 1
        return scores

    # ------------------------------------------------------------------ public API

    def get_records(
        self,
        sequences: torch.LongTensor,
        prompt_len: Optional[int] = None,
    ) -> List[Dict]:
        """Pair captured hidden states with generated token IDs.

        Returns:
            List of dicts, one per generated token, with keys:
            ``step``, ``token_id``, ``token_position``,
            ``hidden_states`` (Dict[int, List[float]]).
        """
        if prompt_len is None:
            prompt_len = self._prompt_len
        if prompt_len == 0:
            raise ValueError(
                "prompt_len is 0 — was the extractor used inside generation?"
            )

        batch_size = sequences.shape[0]
        if batch_size != 1:
            raise ValueError(
                f"MultiLayerGenerationExtractor currently supports batch_size=1, "
                f"got {batch_size}."
            )

        records: List[Dict] = []
        generated_len = sequences.shape[1] - prompt_len

        # Verify all layers have the expected number of entries
        min_buffer_len = min(len(buf) for buf in self._hidden_buffers.values())
        if min_buffer_len < generated_len:
            warnings.warn(
                f"Shortest hidden buffer has {min_buffer_len} entries but "
                f"{generated_len} tokens were generated. Some records will be missing."
            )

        for i in range(min(generated_len, min_buffer_len)):
            token_id = int(sequences[0, prompt_len + i].item())

            # Collect hidden states from all layers for this step
            hs_dict = {}
            for layer_idx in self.layer_indices:
                # (1, 1, hidden_dim) -> flatten to list
                hidden_vec = self._hidden_buffers[layer_idx][i][0, 0, :].tolist()
                hs_dict[layer_idx] = hidden_vec

            records.append({
                "step": i,
                "token_id": token_id,
                "token_position": prompt_len + i,
                "hidden_states": hs_dict,
            })
        return records

    def reset(self):
        """Clear internal buffers for reuse."""
        for buf in self._hidden_buffers.values():
            buf.clear()
        self._step = 0
        self._prompt_len = 0


# =============================================================================
# 4. Legacy Classes (Backward Compatibility)
# =============================================================================

class LatentConflictDetector(nn.Module):
    """Legacy 2-layer MLP classifier.

    .. deprecated::
        Use :class:`PredictiveCodingDetector` for new work. This class is kept
        for backward compatibility with existing checkpoints and scripts.
    """

    LABELS = ["supported", "hallucinated", "uncertain", "contradictory"]

    def __init__(self, hidden_dim: int = 768, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        layers = []
        in_dim = hidden_dim

        for i in range(num_layers):
            out_dim = hidden_dim // 2 if i < num_layers - 1 else len(self.LABELS)
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        # Remove last dropout and relu
        layers = layers[:-2]
        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)

    def classify(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(hidden_states)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return {label: float(prob) for label, prob in zip(self.LABELS, probs)}

    def get_conflict_score(self, hidden_states: torch.Tensor) -> float:
        probs = self.classify(hidden_states)
        return probs["hallucinated"] + probs["uncertain"] + probs["contradictory"]


class HiddenStateExtractor:
    """Legacy single-layer extractor for prompt-encoding states.

    .. deprecated::
        Use :class:`MultiLayerGenerationExtractor` for generation-time extraction.
    """

    def __init__(self, model: AutoModelForCausalLM, layer_idx: int = -4):
        self.model = model
        self.layer_idx = layer_idx
        self.hidden_states: List[torch.Tensor] = []
        self._hook = None

    def _get_target_layer(self):
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            raise ValueError("Unsupported model architecture")
        return layers[self.layer_idx]

    def register_hook(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.hidden_states.append(output.detach().cpu())

        target_layer = self._get_target_layer()
        self._hook = target_layer.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self._hook:
            self._hook.remove()
            self._hook = None

    def get_states(self) -> List[torch.Tensor]:
        states = self.hidden_states
        self.hidden_states = []
        return states


class GenerationHiddenStateExtractor(LogitsProcessor):
    """Legacy single-layer generation-time extractor.

    .. deprecated::
        Use :class:`MultiLayerGenerationExtractor` instead.
    """

    def __init__(self, model: AutoModelForCausalLM, layer_idx: int = -4):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self._hidden_buffer: List[torch.Tensor] = []
        self._hook = None
        self._prompt_len = 0
        self._step = 0
        self.register_hook()

    def _get_target_layer(self):
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            raise ValueError("Unsupported model architecture")
        return layers[self.layer_idx]

    def register_hook(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            last_hidden = output[:, -1:, :].detach().cpu()
            self._hidden_buffer.append(last_hidden)

        target_layer = self._get_target_layer()
        self._hook = target_layer.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self._hook:
            self._hook.remove()
            self._hook = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._step == 0:
            self._prompt_len = input_ids.shape[1]
        self._step += 1
        return scores

    def get_records(
        self,
        sequences: torch.LongTensor,
        prompt_len: Optional[int] = None,
    ) -> List[Dict]:
        if prompt_len is None:
            prompt_len = self._prompt_len
        if prompt_len == 0:
            raise ValueError("prompt_len is 0")

        batch_size = sequences.shape[0]
        if batch_size != 1:
            raise ValueError(f"batch_size=1 required, got {batch_size}")

        records: List[Dict] = []
        generated_len = sequences.shape[1] - prompt_len

        if len(self._hidden_buffer) < generated_len:
            warnings.warn(
                f"Hidden buffer has {len(self._hidden_buffer)} entries but "
                f"{generated_len} tokens were generated."
            )

        for i in range(generated_len):
            if i >= len(self._hidden_buffer):
                break
            token_id = int(sequences[0, prompt_len + i].item())
            hidden_state = self._hidden_buffer[i][0, 0, :].tolist()
            records.append({
                "step": i,
                "token_id": token_id,
                "token_position": prompt_len + i,
                "hidden_state": hidden_state,
            })
        return records

    def reset(self):
        self._hidden_buffer.clear()
        self._step = 0
        self._prompt_len = 0


# ------------------------------------------------------------------------------
# Legacy training helper (kept for import stability)
# ------------------------------------------------------------------------------

def train_conflict_detector(
    detector: LatentConflictDetector,
    model: AutoModelForCausalLM,
    tokenizer,
    train_data: List[Tuple[str, str]],
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """Legacy training routine.

    .. deprecated::
        Use ``scripts/train_conflict_detector.py`` for generation-time training.
    """
    warnings.warn(
        "train_conflict_detector() is deprecated. "
        "Use scripts/train_conflict_detector.py for generation-time training.",
        DeprecationWarning,
        stacklevel=2,
    )
    model.eval()
    detector = detector.to(device)
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    extractor = HiddenStateExtractor(model, layer_idx=-4)
    extractor.register_hook()

    label_map = {label: idx for idx, label in enumerate(detector.LABELS)}

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for prompt, label_str in train_data:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                _ = model(**inputs)

            states = extractor.get_states()
            if not states:
                continue

            last_hidden = states[-1][:, -1, :].to(device)
            logits = detector(last_hidden)
            label = torch.tensor([label_map[label_str]], device=device)

            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=-1).item()
            correct += (pred == label_map[label_str])
            total += 1

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}: loss={total_loss/total:.4f}, acc={acc:.3f}")

    extractor.remove_hook()
    return detector


if __name__ == "__main__":
    # Quick sanity check
    detector = PredictiveCodingDetector(hidden_dim=768, layer_pairs=[(-4, -1)])
    n_params = sum(p.numel() for p in detector.parameters())
    print(f"PredictiveCodingDetector created with {n_params/1e6:.2f}M params")
    print(f"  Primary labels: {detector.PRIMARY_LABELS}")
    print(f"  Secondary labels: {detector.SECONDARY_LABELS}")
    print(f"  Layer pairs: {detector.layer_pairs}")
    print(f"  Temporal decay: {detector.temporal_decay}")

    # Test forward pass
    dummy_hs = {-4: torch.randn(2, 768), -1: torch.randn(2, 768)}
    p_logits, s_logits, c_score, state = detector(dummy_hs)
    print(f"\nDummy forward pass:")
    print(f"  Primary logits shape: {p_logits.shape}")
    print(f"  Secondary logits shape: {s_logits.shape}")
    print(f"  Conflict score shape: {c_score.shape}")
    print(f"  Next state shape: {state.shape}")

    # Test classification
    result = detector.classify({-4: torch.randn(768), -1: torch.randn(768)})
    print(f"\nDummy classification:")
    for k, v in result.items():
        if k != "next_state":
            print(f"  {k}: {v}")
