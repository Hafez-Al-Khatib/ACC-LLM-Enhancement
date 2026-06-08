"""Comprehensive unit tests for PredictiveCodingDetector and related classes.

Covers:
  - Mathematical correctness of prediction error computation
  - Leaky temporal integration behavior
  - Edge cases (batch sizes, missing layers, small dimensions)
  - Interpretability hooks
  - Gradient flow
  - Shape consistency
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.acc_conflict_detector import (
    HierarchicalPredictionErrorModule,
    PredictiveCodingDetector,
    MultiLayerGenerationExtractor,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def hidden_dim() -> int:
    return 64


@pytest.fixture
def layer_pairs() -> list:
    return [(-4, -2), (-2, -1)]


@pytest.fixture
def detector(hidden_dim, layer_pairs):
    return PredictiveCodingDetector(
        hidden_dim=hidden_dim,
        layer_pairs=layer_pairs,
        temporal_decay=0.7,
        dropout=0.0,  # disable dropout for deterministic tests
    )


@pytest.fixture
def sample_hidden_states(hidden_dim):
    return {
        -4: torch.randn(3, hidden_dim),  # batch=3
        -2: torch.randn(3, hidden_dim),
        -1: torch.randn(3, hidden_dim),
    }


# =============================================================================
# 1. HierarchicalPredictionErrorModule Tests
# =============================================================================

class TestHierarchicalPredictionErrorModule:
    def test_output_shape(self, hidden_dim, layer_pairs, sample_hidden_states):
        module = HierarchicalPredictionErrorModule(hidden_dim, layer_pairs, dropout=0.0)
        pe = module(sample_hidden_states)
        assert pe.shape == (3, 2), f"Expected (3, 2), got {pe.shape}"

    def test_non_negative_errors(self, hidden_dim, layer_pairs, sample_hidden_states):
        module = HierarchicalPredictionErrorModule(hidden_dim, layer_pairs, dropout=0.0)
        pe = module(sample_hidden_states)
        assert (pe >= 0).all(), "Prediction errors must be non-negative (MSE property)"

    def test_predictor_parameters_update(self, hidden_dim):
        """Predictor parameters should update during training."""
        module = HierarchicalPredictionErrorModule(hidden_dim, [(0, 1)], dropout=0.0)

        # Get initial weights
        init_weight = module.predictors[0][0].weight.clone()

        # Train one step
        src = torch.randn(4, hidden_dim)
        tgt = torch.randn(4, hidden_dim)
        pe = module({0: src, 1: tgt})
        loss = pe.mean()
        loss.backward()

        # Check gradients exist
        assert module.predictors[0][0].weight.grad is not None
        assert module.predictors[0][0].weight.grad.abs().sum() > 0

        # Optimizer step should change weights
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)
        optimizer.step()
        new_weight = module.predictors[0][0].weight.clone()
        assert not torch.allclose(init_weight, new_weight), "Predictor weights should update during training"

    def test_missing_source_layer_raises(self, hidden_dim, layer_pairs):
        module = HierarchicalPredictionErrorModule(hidden_dim, layer_pairs, dropout=0.0)
        with pytest.raises(KeyError, match="Source layer -4"):
            module({-2: torch.randn(1, hidden_dim), -1: torch.randn(1, hidden_dim)})

    def test_missing_target_layer_raises(self, hidden_dim, layer_pairs):
        module = HierarchicalPredictionErrorModule(hidden_dim, layer_pairs, dropout=0.0)
        with pytest.raises(KeyError, match="Target layer -2"):
            module({-4: torch.randn(1, hidden_dim), -1: torch.randn(1, hidden_dim)})

    def test_empty_hidden_states_raises(self, hidden_dim, layer_pairs):
        module = HierarchicalPredictionErrorModule(hidden_dim, layer_pairs, dropout=0.0)
        with pytest.raises(ValueError, match="hidden_states is empty"):
            module({})

    def test_gradient_flow(self, hidden_dim, layer_pairs, sample_hidden_states):
        module = HierarchicalPredictionErrorModule(hidden_dim, layer_pairs, dropout=0.0)
        for p in module.parameters():
            p.requires_grad = True

        pe = module(sample_hidden_states)
        loss = pe.sum()
        loss.backward()

        # All predictor parameters should have gradients
        for predictor in module.predictors:
            for layer in predictor:
                if isinstance(layer, nn.Linear):
                    assert layer.weight.grad is not None, "Predictor weight has no gradient"
                    assert layer.bias.grad is not None, "Predictor bias has no gradient"


# =============================================================================
# 2. PredictiveCodingDetector Forward Pass Tests
# =============================================================================

class TestPredictiveCodingDetectorForward:
    def test_output_shapes(self, detector, sample_hidden_states):
        p_logits, s_logits, c_score, state = detector(sample_hidden_states)
        batch_size = 3
        assert p_logits.shape == (batch_size, 3), f"Primary logits shape wrong: {p_logits.shape}"
        assert s_logits.shape == (batch_size, 2), f"Secondary logits shape wrong: {s_logits.shape}"
        assert c_score.shape == (batch_size, 1), f"Conflict score shape wrong: {c_score.shape}"
        assert state.shape == (batch_size, detector.num_pairs), f"State shape wrong: {state.shape}"

    def test_conflict_score_range(self, detector, sample_hidden_states):
        _, _, c_score, _ = detector(sample_hidden_states)
        assert (c_score >= 0).all() and (c_score <= 1).all(), \
            f"Conflict score must be in [0, 1], got min={c_score.min()}, max={c_score.max()}"

    def test_temporal_integration_with_prev_state(self, detector, sample_hidden_states):
        p_logits1, s_logits1, c_score1, state1 = detector(sample_hidden_states)
        p_logits2, s_logits2, c_score2, state2 = detector(sample_hidden_states, prev_state=state1)

        # State should be a mixture: alpha * state1 + (1-alpha) * pe2
        alpha = detector.temporal_decay
        # Compute raw PE
        with torch.no_grad():
            pe2 = detector.prediction_error(sample_hidden_states)
        expected_state = alpha * state1 + (1 - alpha) * pe2

        assert torch.allclose(state2, expected_state, atol=1e-5), \
            "Leaky integrator formula incorrect"

    def test_temporal_integration_without_prev_state(self, detector, sample_hidden_states):
        _, _, _, state = detector(sample_hidden_states, prev_state=None)
        with torch.no_grad():
            pe = detector.prediction_error(sample_hidden_states)
        assert torch.allclose(state, pe, atol=1e-5), \
            "Without prev_state, state should equal prediction error"

    def test_different_batch_sizes(self, detector, hidden_dim):
        for batch_size in [1, 4, 8]:
            hs = {
                -4: torch.randn(batch_size, hidden_dim),
                -2: torch.randn(batch_size, hidden_dim),
                -1: torch.randn(batch_size, hidden_dim),
            }
            p_logits, s_logits, c_score, state = detector(hs)
            assert p_logits.shape[0] == batch_size
            assert state.shape[0] == batch_size

    def test_temporal_decay_validation(self, hidden_dim, layer_pairs):
        with pytest.raises(ValueError, match="temporal_decay must be in \\[0, 1\\]"):
            PredictiveCodingDetector(hidden_dim, layer_pairs, temporal_decay=1.5)
        with pytest.raises(ValueError, match="temporal_decay must be in \\[0, 1\\]"):
            PredictiveCodingDetector(hidden_dim, layer_pairs, temporal_decay=-0.1)

    def test_zero_layer_pairs_raises(self, hidden_dim):
        with pytest.raises(ValueError, match="At least one layer pair is required"):
            PredictiveCodingDetector(hidden_dim, layer_pairs=[])

    def test_gradient_flow_end_to_end(self, detector, sample_hidden_states):
        p_logits, s_logits, c_score, state = detector(sample_hidden_states)
        loss = p_logits.sum() + s_logits.sum() + c_score.sum()
        loss.backward()

        # Check that all parameters have gradients
        for name, param in detector.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"


# =============================================================================
# 3. PredictiveCodingDetector classify() Tests
# =============================================================================

class TestPredictiveCodingDetectorClassify:
    def test_classify_output_structure(self, detector, hidden_dim):
        hs = {-4: torch.randn(hidden_dim), -2: torch.randn(hidden_dim), -1: torch.randn(hidden_dim)}
        result = detector.classify(hs)

        assert "primary" in result
        assert "primary_probs" in result
        assert "secondary" in result
        assert "secondary_probs" in result
        assert "conflict_score" in result
        assert "next_state" in result

        assert result["primary"] in detector.PRIMARY_LABELS
        assert 0 <= result["conflict_score"] <= 1
        assert isinstance(result["primary_probs"], dict)
        assert sum(result["primary_probs"].values()) == pytest.approx(1.0, abs=1e-5)

    def test_secondary_only_for_unsupported(self, detector, hidden_dim):
        # We can't force a specific label without controlling weights,
        # but we can check structure consistency
        hs = {-4: torch.randn(hidden_dim), -2: torch.randn(hidden_dim), -1: torch.randn(hidden_dim)}
        result = detector.classify(hs)

        if result["primary"] == "unsupported":
            assert result["secondary"] in detector.SECONDARY_LABELS
            assert result["secondary_probs"] is not None
            assert sum(result["secondary_probs"].values()) == pytest.approx(1.0, abs=1e-5)
        else:
            assert result["secondary"] is None
            assert result["secondary_probs"] is None

    def test_classify_with_prev_state(self, detector, hidden_dim):
        hs = {-4: torch.randn(hidden_dim), -2: torch.randn(hidden_dim), -1: torch.randn(hidden_dim)}
        result1 = detector.classify(hs)
        result2 = detector.classify(hs, prev_state=result1["next_state"])

        # With same input but different state, outputs may differ
        assert "next_state" in result2

    def test_batch_dim_1d_input(self, detector, hidden_dim):
        hs = {-4: torch.randn(hidden_dim), -2: torch.randn(hidden_dim), -1: torch.randn(hidden_dim)}
        result = detector.classify(hs)
        assert result["next_state"].shape[0] == 1  # batch dim added internally


# =============================================================================
# 4. PredictiveCodingDetector Sequence Tests
# =============================================================================

class TestPredictiveCodingDetectorSequence:
    def test_predict_sequence_length(self, detector, hidden_dim):
        seq = [
            {-4: torch.randn(hidden_dim), -2: torch.randn(hidden_dim), -1: torch.randn(hidden_dim)}
            for _ in range(5)
        ]
        results = detector.predict_sequence(seq)
        assert len(results) == 5

    def test_predict_sequence_state_propagation(self, detector, hidden_dim):
        seq = [
            {-4: torch.randn(hidden_dim), -2: torch.randn(hidden_dim), -1: torch.randn(hidden_dim)}
            for _ in range(3)
        ]
        results = detector.predict_sequence(seq)

        # Each step should have a next_state
        for i, r in enumerate(results):
            assert "next_state" in r
            assert r["next_state"].shape == (1, detector.num_pairs)


# =============================================================================
# 5. Interpretability Hooks Tests
# =============================================================================

class TestInterpretabilityHooks:
    def test_get_prediction_errors(self, detector, hidden_dim):
        hs = {-4: torch.randn(hidden_dim), -2: torch.randn(hidden_dim), -1: torch.randn(hidden_dim)}
        errors = detector.get_prediction_errors(hs)

        assert len(errors) == len(detector.layer_pairs)
        for pair in detector.layer_pairs:
            assert pair in errors
            assert isinstance(errors[pair], float)
            assert errors[pair] >= 0

    def test_get_layer_contributions(self, detector, hidden_dim):
        hs = {-4: torch.randn(hidden_dim), -2: torch.randn(hidden_dim), -1: torch.randn(hidden_dim)}
        contributions = detector.get_layer_contributions(hs, target="conflict_score")

        assert len(contributions) == len(detector.layer_pairs)
        for pair in detector.layer_pairs:
            assert pair in contributions
            assert isinstance(contributions[pair], float)
            assert contributions[pair] >= 0

    def test_get_layer_contributions_primary_labels(self, detector, hidden_dim):
        hs = {-4: torch.randn(hidden_dim), -2: torch.randn(hidden_dim), -1: torch.randn(hidden_dim)}
        for label in detector.PRIMARY_LABELS:
            contributions = detector.get_layer_contributions(hs, target=f"primary_{label}")
            assert len(contributions) == len(detector.layer_pairs)

    def test_get_layer_contributions_secondary_labels(self, detector, hidden_dim):
        hs = {-4: torch.randn(hidden_dim), -2: torch.randn(hidden_dim), -1: torch.randn(hidden_dim)}
        for label in detector.SECONDARY_LABELS:
            contributions = detector.get_layer_contributions(hs, target=f"secondary_{label}")
            assert len(contributions) == len(detector.layer_pairs)

    def test_get_layer_contributions_invalid_target(self, detector, hidden_dim):
        hs = {-4: torch.randn(hidden_dim), -2: torch.randn(hidden_dim), -1: torch.randn(hidden_dim)}
        with pytest.raises(ValueError, match="Unknown target"):
            detector.get_layer_contributions(hs, target="invalid_target")

    def test_explain_structure(self, detector, hidden_dim):
        hs = {-4: torch.randn(hidden_dim), -2: torch.randn(hidden_dim), -1: torch.randn(hidden_dim)}
        explanation = detector.explain(hs)

        assert "classification" in explanation
        assert "prediction_errors" in explanation
        assert "layer_contributions" in explanation
        assert "state_before" in explanation
        assert "state_after" in explanation
        assert "prediction_error_vector" in explanation

        assert "conflict_score" in explanation["layer_contributions"]
        assert "primary" in explanation["layer_contributions"]
        assert "secondary" in explanation["layer_contributions"]

    def test_explain_with_prev_state(self, detector, hidden_dim):
        hs = {-4: torch.randn(hidden_dim), -2: torch.randn(hidden_dim), -1: torch.randn(hidden_dim)}
        prev_state = torch.randn(1, detector.num_pairs)
        explanation = detector.explain(hs, prev_state=prev_state)

        assert explanation["state_before"] is not None
        assert explanation["state_after"] is not None


# =============================================================================
# 6. MultiLayerGenerationExtractor Tests
# =============================================================================

class TestMultiLayerGenerationExtractor:
    def test_init_layers_normalization(self):
        """Test that negative indices are correctly normalized."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        extractor = MultiLayerGenerationExtractor(model, layer_indices=[-1, -2])

        # tiny-gpt2 has 2 layers, so -1 -> 1, -2 -> 0
        assert extractor._positive_indices == [1, 0]
        extractor.remove_hooks()

    def test_out_of_range_layer_raises(self):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        with pytest.raises(ValueError, match="out of range"):
            MultiLayerGenerationExtractor(model, layer_indices=[-10])

    def test_reset_clears_buffers(self):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        extractor = MultiLayerGenerationExtractor(model, layer_indices=[0, 1])

        # Simulate some entries
        extractor._hidden_buffers[0].append(torch.randn(1, 1, 2))
        extractor._hidden_buffers[1].append(torch.randn(1, 1, 2))

        extractor.reset()
        assert len(extractor._hidden_buffers[0]) == 0
        assert len(extractor._hidden_buffers[1]) == 0
        assert extractor._step == 0
        assert extractor._prompt_len == 0
        extractor.remove_hooks()


# =============================================================================
# 7. Sanity Checks
# =============================================================================

class TestSanityChecks:
    def test_detector_parameters_update(self, hidden_dim):
        """Detector parameters should receive gradients during backprop."""
        detector = PredictiveCodingDetector(
            hidden_dim=hidden_dim,
            layer_pairs=[(0, 1)],
            temporal_decay=0.0,
            dropout=0.0,
        )

        # Forward + backward with MSE loss to ensure non-trivial gradients
        hs = {0: torch.randn(4, hidden_dim), 1: torch.randn(4, hidden_dim)}
        p_logits, s_logits, c_score, _ = detector(hs)
        # Use MSE against non-zero targets to guarantee non-zero gradients
        target = torch.tensor([[1.0, 0.0, 0.0]] * 4)  # (4, 3)
        loss = F.mse_loss(p_logits, target) + s_logits.pow(2).mean() + c_score.pow(2).mean()
        loss.backward()

        # Check that at least some parameters have non-zero gradients
        has_nonzero_grad = False
        for name, param in detector.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_nonzero_grad = True
                break
        assert has_nonzero_grad, "No detector parameter has a non-zero gradient"

    def test_parameter_count_scales_correctly(self, hidden_dim):
        """Parameter count should scale with hidden_dim and num_pairs."""
        detector1 = PredictiveCodingDetector(hidden_dim=hidden_dim, layer_pairs=[(0, 1)], dropout=0.0)
        detector2 = PredictiveCodingDetector(hidden_dim=hidden_dim, layer_pairs=[(0, 1), (1, 2)], dropout=0.0)

        params1 = sum(p.numel() for p in detector1.parameters())
        params2 = sum(p.numel() for p in detector2.parameters())

        # More layer pairs = more predictor parameters
        assert params2 > params1, "More layer pairs should mean more parameters"

    def test_dropout_zero_deterministic(self, detector, sample_hidden_states):
        """With dropout=0, repeated forward passes should be identical."""
        detector.eval()  # disable dropout if any
        p1, s1, c1, st1 = detector(sample_hidden_states)
        p2, s2, c2, st2 = detector(sample_hidden_states)

        assert torch.allclose(p1, p2, atol=1e-6)
        assert torch.allclose(s1, s2, atol=1e-6)
        assert torch.allclose(c1, c2, atol=1e-6)
        assert torch.allclose(st1, st2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
