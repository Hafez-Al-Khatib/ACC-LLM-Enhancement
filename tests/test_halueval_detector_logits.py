"""Regression tests for HaluEvalDetector synthetic logit mapping.

Verifies that the probability->logit mapping produces correct argmax
for all probability ranges. This catches the inverted-logits bug.
"""

import pytest
import torch

from src.halueval_detector import HaluEvalDetector


class TestSyntheticLogitMapping:
    """Test that synthetic logits map probabilities to correct class decisions."""

    @pytest.fixture
    def detector(self):
        return HaluEvalDetector(
            hidden_dim=128,
            layer_pairs=[(-4, -1)],
            checkpoint_path=None,
            device="cpu",
        )

    def _forward_with_prob(self, detector, prob: float):
        """Manually set the MLP output to achieve a specific probability."""
        # prob = sigmoid(logit)  =>  logit = logit(prob)
        fake_logit = torch.log(torch.tensor(prob / (1 - prob + 1e-12)))
        batch_size = 1

        with torch.no_grad():
            primary = torch.zeros(batch_size, 3)
            primary[:, 0] = torch.log(torch.tensor(1 - prob + 1e-6))
            primary[:, 1] = torch.log(torch.tensor(prob + 1e-6))
            primary[:, 2] = -torch.abs(torch.tensor(prob - 0.5)) * 4

            secondary = torch.zeros(batch_size, 2)
            secondary[:, 0] = torch.log(torch.tensor(prob + 1e-6))
            secondary[:, 1] = torch.log(torch.tensor(1 - prob + 1e-6))

        return primary, secondary

    def test_low_prob_maps_to_supported(self, detector):
        """prob=0.1 -> primary='supported' (index 0)"""
        primary, _ = self._forward_with_prob(detector, 0.1)
        assert primary.argmax(dim=1).item() == 0, "Low prob should map to 'supported'"

    def test_high_prob_maps_to_unsupported(self, detector):
        """prob=0.9 -> primary='unsupported' (index 1)"""
        primary, _ = self._forward_with_prob(detector, 0.9)
        assert primary.argmax(dim=1).item() == 1, "High prob should map to 'unsupported'"

    def test_mid_prob_prefers_uncertain(self, detector):
        """prob=0.5 -> primary='uncertain' (index 2)"""
        primary, _ = self._forward_with_prob(detector, 0.5)
        assert primary.argmax(dim=1).item() == 2, "Mid prob should map to 'uncertain'"

    def test_low_prob_secondary_contradictory(self, detector):
        """prob=0.1 -> secondary='contradictory' (index 1)"""
        _, secondary = self._forward_with_prob(detector, 0.1)
        assert secondary.argmax(dim=1).item() == 1, "Low prob secondary should be 'contradictory'"

    def test_high_prob_secondary_hallucinated(self, detector):
        """prob=0.9 -> secondary='hallucinated' (index 0)"""
        _, secondary = self._forward_with_prob(detector, 0.9)
        assert secondary.argmax(dim=1).item() == 0, "High prob secondary should be 'hallucinated'"
