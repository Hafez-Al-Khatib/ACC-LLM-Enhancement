"""HaluEval-trained hallucination detector wrapped for ACC integration.

The underlying model is a lightweight MLP (SAPLMA-style) trained on
prediction-error features between layer pairs. This wrapper converts
raw hidden-state dicts into prediction-error features and produces
outputs compatible with PredictiveCodingDetector's interface.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class SimpleHallucinationDetector(nn.Module):
    """Lightweight MLP detector trained on HaluEval QA."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class HaluEvalDetector(nn.Module):
    """Wrapper around SimpleHallucinationDetector for ACC compatibility.

    Args:
        hidden_dim: Dimension of LLM hidden states.
        layer_pairs: List of (layer_a, layer_b) tuples for prediction errors.
        checkpoint_path: Path to trained detector weights.
        device: torch device.
    """

    PRIMARY_LABELS = ["supported", "unsupported", "uncertain"]
    SECONDARY_LABELS = ["hallucinated", "contradictory"]

    def __init__(
        self,
        hidden_dim: int,
        layer_pairs: Optional[List[Tuple[int, int]]] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_pairs = layer_pairs or [(-12, -8), (-8, -4), (-4, -1)]
        self.device = device

        self.mlp = SimpleHallucinationDetector(input_dim=hidden_dim, hidden_dim=256)
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=device)
            self.mlp.load_state_dict(state)
        self.mlp.to(device)
        self.mlp.eval()

    def _compute_features(
        self, hidden_states: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Compute average absolute prediction error across layer pairs."""
        # Ensure all hidden states are 2D (batch, hidden)
        batched_hs = {}
        for k, v in hidden_states.items():
            v = v.to(self.device)
            if v.dim() == 1:
                v = v.unsqueeze(0)
            batched_hs[k] = v

        errors = []
        for l1, l2 in self.layer_pairs:
            if l1 in batched_hs and l2 in batched_hs:
                error = torch.abs(batched_hs[l1] - batched_hs[l2])
                errors.append(error)

        if not errors:
            batch_size = next(iter(batched_hs.values())).shape[0]
            return torch.zeros(batch_size, self.hidden_dim, device=self.device)

        return torch.stack(errors).mean(dim=0).float()

    def forward(
        self,
        hidden_states: Dict[int, torch.Tensor],
        prev_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return primary/secondary logits, conflict score, next state.

        Output shapes match PredictiveCodingDetector for drop-in replacement.
        """
        features = self._compute_features(hidden_states)  # (batch, hidden_dim)
        batch_size = features.shape[0]

        # Ensure model and features are on the same device
        model_device = next(self.mlp.parameters()).device
        features = features.to(model_device)

        with torch.no_grad():
            logits = self.mlp(features)  # (batch,)
            prob = torch.sigmoid(logits)  # P(hallucination)

        # Map probability to 3-way primary and 2-way secondary logits
        # prob < 0.3 -> supported (low hallucination probability)
        # 0.3 <= prob < 0.7 -> uncertain
        # prob >= 0.7 -> unsupported (high hallucination probability)
        primary_logits = torch.zeros(batch_size, 3, device=model_device)
        primary_logits[:, 0] = torch.log(1 - prob + 1e-6)  # supported: high when prob -> 0
        primary_logits[:, 1] = torch.log(prob + 1e-6)      # unsupported: high when prob -> 1
        primary_logits[:, 2] = -torch.abs(prob - 0.5) * 4  # uncertain: peak at prob = 0.5

        secondary_logits = torch.zeros(batch_size, 2, device=model_device)
        secondary_logits[:, 0] = torch.log(prob + 1e-6)       # hallucinated: high when prob -> 1
        secondary_logits[:, 1] = torch.log(1 - prob + 1e-6)   # contradictory/no: high when prob -> 0

        conflict_score = prob.unsqueeze(-1)  # (batch, 1)

        # State is just the conflict score for temporal chaining
        next_state = conflict_score

        return primary_logits, secondary_logits, conflict_score, next_state

    def predict_sequence(
        self, hidden_sequence: List[Dict[int, torch.Tensor]]
    ) -> List[Dict[str, object]]:
        """Convenience method matching PredictiveCodingDetector API."""
        results = []
        prev_state = None
        for hs in hidden_sequence:
            primary_logits, secondary_logits, conflict_score, next_state = self.forward(
                hs, prev_state
            )
            primary_idx = int(primary_logits.argmax(dim=1)[0].item())
            secondary_idx = int(secondary_logits.argmax(dim=1)[0].item())

            results.append({
                "primary": self.PRIMARY_LABELS[primary_idx],
                "secondary": self.SECONDARY_LABELS[secondary_idx],
                "conflict_score": float(conflict_score[0].item()),
                "next_state": next_state,
            })
            prev_state = next_state

        return results
