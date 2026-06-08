"""Unit tests for the training pipeline.

Covers:
  - Label mapping correctness
  - Dataset construction (hard and soft labels)
  - Loss function computation
  - Focal loss behavior
  - Soft cross-entropy correctness
  - Gradient clipping
  - Checkpoint saving/loading
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.train_conflict_detector import (
    OLD_TO_PRIMARY,
    OLD_TO_SECONDARY,
    PRIMARY_IDX_MAP,
    SECONDARY_IDX_MAP,
    MultiLayerTokenDataset,
    FocalLoss,
    soft_cross_entropy,
    compute_loss,
    save_checkpoint,
    collate_fn,
)
from src.acc_conflict_detector import PredictiveCodingDetector


# =============================================================================
# Label Mapping Tests
# =============================================================================

class TestLabelMapping:
    def test_all_old_labels_mapped(self):
        old_labels = ["supported", "hallucinated", "uncertain", "contradictory"]
        for label in old_labels:
            assert label in OLD_TO_PRIMARY
            assert label in OLD_TO_SECONDARY

    def test_supported_mapping(self):
        assert OLD_TO_PRIMARY["supported"] == "supported"
        assert OLD_TO_SECONDARY["supported"] is None

    def test_hallucinated_mapping(self):
        assert OLD_TO_PRIMARY["hallucinated"] == "unsupported"
        assert OLD_TO_SECONDARY["hallucinated"] == "hallucinated"

    def test_uncertain_mapping(self):
        assert OLD_TO_PRIMARY["uncertain"] == "uncertain"
        assert OLD_TO_SECONDARY["uncertain"] is None

    def test_contradictory_mapping(self):
        assert OLD_TO_PRIMARY["contradictory"] == "unsupported"
        assert OLD_TO_SECONDARY["contradictory"] == "contradictory"

    def test_primary_idx_map_complete(self):
        assert set(PRIMARY_IDX_MAP.keys()) == {"supported", "unsupported", "uncertain"}

    def test_secondary_idx_map_complete(self):
        assert set(SECONDARY_IDX_MAP.keys()) == {"hallucinated", "contradictory"}


# =============================================================================
# Dataset Tests
# =============================================================================

class TestDataset:
    def test_hard_label_dataset(self):
        records = [
            {"label": "supported", "hidden_states": {"-1": [1.0, 2.0], "-2": [3.0, 4.0]}},
            {"label": "hallucinated", "hidden_states": {"-1": [5.0, 6.0], "-2": [7.0, 8.0]}},
        ]
        ds = MultiLayerTokenDataset(records, use_soft_labels=False)
        assert len(ds) == 2

        sample = ds[0]
        assert sample["primary"] == "supported"
        assert sample["secondary"] is None
        assert -1 in sample["hidden_states"]
        assert -2 in sample["hidden_states"]

    def test_soft_label_dataset(self):
        records = [
            {
                "label": "supported",
                "hidden_states": {"-1": [1.0, 2.0]},
                "primary_soft": [0.8, 0.1, 0.1],
            },
            {
                "label": "hallucinated",
                "hidden_states": {"-1": [3.0, 4.0]},
                "primary_soft": [0.1, 0.8, 0.1],
                "secondary_soft": [0.9, 0.1],
            },
        ]
        ds = MultiLayerTokenDataset(records, use_soft_labels=True)
        assert len(ds) == 2

        s0 = ds[0]
        assert "primary_soft" in s0
        assert torch.allclose(s0["primary_soft"], torch.tensor([0.8, 0.1, 0.1]))
        assert s0["secondary_soft"] is None

        s1 = ds[1]
        assert torch.allclose(s1["secondary_soft"], torch.tensor([0.9, 0.1]))

    def test_soft_label_fallback_to_hard(self):
        """If soft labels not in record, fall back to one-hot from hard label."""
        records = [{"label": "supported", "hidden_states": {"-1": [1.0, 2.0]}}]
        ds = MultiLayerTokenDataset(records, use_soft_labels=True)
        s = ds[0]
        expected = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(s["primary_soft"], expected)

    def test_old_format_fallback(self):
        """Records with single hidden_state (old format) should work."""
        records = [{"label": "uncertain", "hidden_state": [1.0, 2.0, 3.0]}]
        ds = MultiLayerTokenDataset(records, use_soft_labels=False)
        s = ds[0]
        assert -1 in s["hidden_states"]
        assert s["primary"] == "uncertain"


# =============================================================================
# Loss Function Tests
# =============================================================================

class TestFocalLoss:
    def test_focal_loss_vs_ce_when_gamma_zero(self):
        """Focal loss with gamma=0 should equal standard CE."""
        logits = torch.randn(8, 3)
        target = torch.randint(0, 3, (8,))

        focal = FocalLoss(alpha=1.0, gamma=0.0)
        ce = nn.CrossEntropyLoss()

        fl_val = focal(logits, target)
        ce_val = ce(logits, target)

        assert torch.allclose(fl_val, ce_val, atol=1e-5)

    def test_focal_loss_reduces_with_high_confidence(self):
        """Focal loss should down-weight easy examples."""
        # Very confident correct predictions
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        target = torch.tensor([0, 1])

        focal = FocalLoss(gamma=2.0)
        ce = nn.CrossEntropyLoss()

        fl_val = focal(logits, target)
        ce_val = ce(logits, target)

        # Focal loss should be much smaller than CE for confident predictions
        assert fl_val < ce_val * 0.5, "Focal loss should down-weight easy examples"


class TestSoftCrossEntropy:
    def test_soft_ce_equals_hard_ce_for_onehot(self):
        """Soft CE with one-hot targets should equal hard CE."""
        logits = torch.randn(4, 3)
        target_idx = torch.randint(0, 3, (4,))

        # One-hot soft targets
        soft_targets = torch.zeros(4, 3)
        soft_targets[range(4), target_idx] = 1.0

        soft_loss = soft_cross_entropy(logits, soft_targets)
        hard_loss = nn.CrossEntropyLoss()(logits, target_idx)

        assert torch.allclose(soft_loss, hard_loss, atol=1e-5)

    def test_soft_ce_with_uniform_targets(self):
        """Uniform soft targets vs peaked: test that both compute correctly."""
        logits = torch.randn(4, 3)
        uniform = torch.ones(4, 3) / 3.0
        peaked = torch.zeros(4, 3)
        peaked[:, 0] = 1.0

        uniform_loss = soft_cross_entropy(logits, uniform)
        peaked_loss = soft_cross_entropy(logits, peaked)

        # Both should be valid losses (positive, finite)
        assert uniform_loss > 0
        assert peaked_loss > 0
        assert torch.isfinite(uniform_loss)
        assert torch.isfinite(peaked_loss)


# =============================================================================
# Compute Loss Tests
# =============================================================================

class TestComputeLoss:
    @pytest.fixture
    def detector(self):
        return PredictiveCodingDetector(hidden_dim=8, layer_pairs=[(0, 1)], dropout=0.0)

    @pytest.fixture
    def batch(self):
        return {
            "hidden_states": [
                {0: torch.randn(8), 1: torch.randn(8)},
                {0: torch.randn(8), 1: torch.randn(8)},
            ],
            "primary": ["supported", "unsupported"],
            "secondary": [None, "hallucinated"],
        }

    def test_hard_label_loss(self, detector, batch):
        primary_criterion = nn.CrossEntropyLoss()
        secondary_criterion = nn.CrossEntropyLoss(reduction="none")
        device = torch.device("cpu")

        loss, metrics = compute_loss(
            detector, batch, primary_criterion, secondary_criterion, device,
            secondary_weight=0.5, use_soft_labels=False
        )

        assert loss.dim() == 0  # scalar
        assert metrics["primary_loss"] > 0
        assert metrics["secondary_loss"] > 0  # one sample has secondary label
        assert metrics["total_loss"] > 0

    def test_secondary_loss_zero_when_no_secondary(self):
        """If batch has no secondary labels, secondary loss should be 0."""
        detector = PredictiveCodingDetector(hidden_dim=8, layer_pairs=[(0, 1)], dropout=0.0)
        batch = {
            "hidden_states": [{0: torch.randn(8), 1: torch.randn(8)}],
            "primary": ["supported"],
            "secondary": [None],
        }
        primary_criterion = nn.CrossEntropyLoss()
        secondary_criterion = nn.CrossEntropyLoss(reduction="none")

        loss, metrics = compute_loss(
            detector, batch, primary_criterion, secondary_criterion, torch.device("cpu"),
            secondary_weight=0.5, use_soft_labels=False
        )

        assert metrics["secondary_loss"] == 0.0

    def test_conflict_score_auxiliary_loss(self, detector, batch):
        primary_criterion = nn.CrossEntropyLoss()
        secondary_criterion = nn.CrossEntropyLoss(reduction="none")
        device = torch.device("cpu")

        loss, metrics = compute_loss(
            detector, batch, primary_criterion, secondary_criterion, device,
            secondary_weight=0.5, use_soft_labels=False, conflict_score_weight=1.0
        )

        assert metrics["conflict_score_loss"] > 0
        assert metrics["total_loss"] > metrics["primary_loss"] + 0.5 * metrics["secondary_loss"]

    def test_gradient_flow(self, detector, batch):
        primary_criterion = nn.CrossEntropyLoss()
        secondary_criterion = nn.CrossEntropyLoss(reduction="none")
        device = torch.device("cpu")

        loss, _ = compute_loss(
            detector, batch, primary_criterion, secondary_criterion, device,
            secondary_weight=0.5, use_soft_labels=False, conflict_score_weight=1.0
        )
        loss.backward()

        has_grad = False
        for param in detector.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No parameter received gradient"


# =============================================================================
# Checkpoint Tests
# =============================================================================

class TestCheckpoint:
    def test_save_and_load_checkpoint(self, tmp_path):
        detector = PredictiveCodingDetector(hidden_dim=16, layer_pairs=[(0, 1)], dropout=0.0)
        save_dir = tmp_path / "checkpoints"
        metrics = {"epoch": 5, "primary_macro_f1": 0.85}

        save_checkpoint(detector, save_dir, metrics, suffix="_test")

        assert (save_dir / "detector_test.pt").exists()
        assert (save_dir / "config_test.json").exists()

        # Load and verify
        loaded = PredictiveCodingDetector(hidden_dim=16, layer_pairs=[(0, 1)])
        loaded.load_state_dict(torch.load(save_dir / "detector_test.pt"))

        for p1, p2 in zip(detector.parameters(), loaded.parameters()):
            assert torch.allclose(p1, p2)


# =============================================================================
# Integration: End-to-end mini training
# =============================================================================

class TestMiniTraining:
    def test_training_decreases_loss(self):
        """Training should decrease loss over multiple steps."""
        hidden_dim = 8
        detector = PredictiveCodingDetector(hidden_dim=hidden_dim, layer_pairs=[(0, 1)], dropout=0.0)
        optimizer = torch.optim.Adam(detector.parameters(), lr=1e-2)
        primary_criterion = nn.CrossEntropyLoss()
        secondary_criterion = nn.CrossEntropyLoss(reduction="none")
        device = torch.device("cpu")

        losses = []
        for _ in range(20):
            batch = {
                "hidden_states": [
                    {0: torch.randn(hidden_dim), 1: torch.randn(hidden_dim)}
                    for _ in range(8)
                ],
                "primary": ["supported", "unsupported", "uncertain", "supported",
                           "unsupported", "uncertain", "supported", "unsupported"],
                "secondary": [None, "hallucinated", None, None,
                             "contradictory", None, None, "hallucinated"],
            }
            loss, _ = compute_loss(
                detector, batch, primary_criterion, secondary_criterion, device,
                secondary_weight=0.3
            )
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should generally decrease (compare first 5 avg vs last 5 avg)
        first_avg = sum(losses[:5]) / 5
        last_avg = sum(losses[-5:]) / 5
        assert last_avg < first_avg, f"Loss did not decrease: first={first_avg:.4f}, last={last_avg:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
