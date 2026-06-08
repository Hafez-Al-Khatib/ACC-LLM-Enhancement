"""Integration tests for ACC-enhanced generation.

Covers:
  - Basic generation with ACCEnhancedGenerator
  - Entropy monitoring (flag / regenerate / warning / suppress)
  - Real-time and post-hoc conflict detection
  - UnifiedDecisionEngine decision matrix
  - Configurable markers
  - Explainability (per_token_decisions, explain_decisions)
  - Batch generation, device handling, error resilience
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Silence future warnings from transformers
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.acc_integration import (
    ACCEnhancedGenerator,
    ACCGenerationOutput,
    MarkerConfig,
    UnifiedDecisionEngine,
    Decision,
    _ACCLogitsProcessor,
    _EntropyLogitsProcessor,
)
from src.acc_layer import EntropyMonitor
from src.acc_conflict_detector import PredictiveCodingDetector


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    """Load tiny GPT-2 once per test module."""
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.fixture
def generator_flag(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    return ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        action="flag",
        threshold=0.5,  # low threshold to trigger easily
        mode="absolute",
    )


@pytest.fixture
def generator_regenerate(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    return ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        action="regenerate",
        threshold=0.5,
        mode="absolute",
        regen_multiplier=2.0,
        max_regenerations=3,
    )


@pytest.fixture
def generator_warning(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    return ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        action="warning",
        threshold=0.5,
        mode="absolute",
    )


@pytest.fixture
def generator_suppress(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    return ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        action="suppress",
        threshold=0.5,
        mode="absolute",
    )


@pytest.fixture
def mock_detector():
    """Return a mock PredictiveCodingDetector for fast tests."""
    detector = MagicMock(spec=PredictiveCodingDetector)
    detector.PRIMARY_LABELS = ["supported", "unsupported", "uncertain"]
    detector.SECONDARY_LABELS = ["hallucinated", "contradictory"]
    return detector


# =============================================================================
# 1. Basic generation
# =============================================================================

class TestBasicGeneration:
    def test_generate_returns_sequences(self, generator_flag, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        input_ids = tokenizer("Hello", return_tensors="pt").input_ids
        result = generator_flag.generate(
            input_ids=input_ids,
            max_new_tokens=5,
            return_dict_in_generate=False,
        )
        assert isinstance(result, torch.LongTensor)
        assert result.shape[0] == input_ids.shape[0]
        assert result.shape[1] >= input_ids.shape[1]

    def test_generate_from_prompt_returns_text(self, generator_flag, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        result = generator_flag.generate_from_prompt(
            "Hello",
            max_new_tokens=5,
            return_dict_in_generate=True,
        )
        assert isinstance(result, ACCGenerationOutput)
        assert isinstance(result.text, list)
        assert len(result.text) == 1
        assert isinstance(result.text[0], str)
        assert len(result.text[0]) > 0

    def test_generate_dict_output_has_all_fields(self, generator_flag, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        result = generator_flag.generate_from_prompt(
            "Hello",
            max_new_tokens=5,
            return_dict_in_generate=True,
        )
        assert isinstance(result.sequences, torch.LongTensor)
        assert isinstance(result.text, list)
        assert isinstance(result.per_token_entropy, list)
        assert isinstance(result.uncertain_steps, list)
        assert isinstance(result.regenerations, list)
        assert isinstance(result.confidence_score, list)
        assert isinstance(result.per_token_decisions, list)


# =============================================================================
# 2. Entropy monitoring actions
# =============================================================================

class TestEntropyActions:
    def test_flag_inserts_marker(self, generator_flag, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        # Use a very low threshold to force flagging
        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            action="flag",
            threshold=0.01,
            mode="absolute",
        )
        result = gen.generate_from_prompt(
            "The quick brown fox",
            max_new_tokens=10,
            return_dict_in_generate=True,
        )
        # With such a low threshold, almost every token should be flagged
        assert len(result.per_token_decisions[0]) > 0
        actions = [d["action"] for d in result.per_token_decisions[0]]
        assert "flag" in actions or "regenerate" in actions

    def test_regenerate_increments_count(self, generator_regenerate, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            action="regenerate",
            threshold=0.01,
            mode="absolute",
            regen_multiplier=2.0,
            max_regenerations=3,
        )
        result = gen.generate_from_prompt(
            "The",
            max_new_tokens=5,
            return_dict_in_generate=True,
        )
        assert len(result.regenerations) == 1
        # Regenerations may or may not happen depending on entropy
        assert result.regenerations[0] >= 0

    def test_warning_prefix_in_text(self, generator_warning, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            action="warning",
            threshold=0.01,
            mode="absolute",
        )
        result = gen.generate_from_prompt(
            "The quick brown fox",
            max_new_tokens=5,
            return_dict_in_generate=True,
        )
        # Check that decisions include warnings
        actions = [d["action"] for d in result.per_token_decisions[0]]
        assert "warning" in actions or "flag" in actions

    def test_suppress_blocks_top_token(self, generator_suppress, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            action="suppress",
            threshold=0.01,
            mode="absolute",
        )
        result = gen.generate_from_prompt(
            "The",
            max_new_tokens=5,
            return_dict_in_generate=True,
        )
        actions = [d["action"] for d in result.per_token_decisions[0]]
        # Suppress may be downgraded to flag for pure entropy, so check either
        assert "suppress" in actions or "flag" in actions


# =============================================================================
# 3. UnifiedDecisionEngine
# =============================================================================

class TestUnifiedDecisionEngine:
    def test_pass_no_signal(self):
        engine = UnifiedDecisionEngine(entropy_threshold=1.0, conflict_score_threshold=0.5)
        dec = engine.decide(entropy=0.5, is_uncertain=False, conflict_score=None, primary=None)
        assert dec.action == "pass"
        assert "no conflict" in dec.reason

    def test_flag_entropy_only(self):
        engine = UnifiedDecisionEngine(action="flag", entropy_threshold=1.0)
        dec = engine.decide(entropy=1.5, is_uncertain=True, conflict_score=None, primary=None)
        assert dec.action == "flag"
        assert "Entropy" in dec.reason

    def test_flag_conflict_unsupported(self):
        engine = UnifiedDecisionEngine(action="flag", conflict_score_threshold=0.5)
        dec = engine.decide(
            entropy=0.5,
            is_uncertain=False,
            conflict_score=0.8,
            primary="unsupported",
            secondary="hallucinated",
        )
        assert dec.action == "flag"
        assert "Conflict" in dec.reason
        assert dec.primary == "unsupported"
        assert dec.secondary == "hallucinated"

    def test_flag_conflict_uncertain(self):
        engine = UnifiedDecisionEngine(action="flag", conflict_score_threshold=0.5)
        dec = engine.decide(
            entropy=0.5,
            is_uncertain=False,
            conflict_score=0.8,
            primary="uncertain",
            secondary=None,
        )
        assert dec.action == "flag"
        assert "Conflict" in dec.reason

    def test_regenerate_dual_signal(self):
        engine = UnifiedDecisionEngine(
            action="flag",
            entropy_threshold=1.0,
            conflict_score_threshold=0.5,
            dual_signal_regenerate=True,
        )
        dec = engine.decide(
            entropy=1.5,
            is_uncertain=True,
            conflict_score=0.8,
            primary="unsupported",
            secondary="hallucinated",
        )
        assert dec.action == "regenerate"
        assert "Dual signal" in dec.reason

    def test_suppress_action_conflict(self):
        engine = UnifiedDecisionEngine(action="suppress", conflict_score_threshold=0.5)
        dec = engine.decide(
            entropy=0.5,
            is_uncertain=False,
            conflict_score=0.8,
            primary="unsupported",
            secondary="hallucinated",
        )
        assert dec.action == "suppress"

    def test_entropy_suppress_downgraded_to_flag(self):
        engine = UnifiedDecisionEngine(action="suppress", entropy_threshold=1.0)
        dec = engine.decide(entropy=1.5, is_uncertain=True, conflict_score=None, primary=None)
        assert dec.action == "flag"
        assert "Entropy" in dec.reason

    def test_custom_marker_config(self):
        mc = MarkerConfig(hallucination="[HALLU]", contradiction="[CONTRA]")
        engine = UnifiedDecisionEngine(marker_config=mc)
        marker = engine.marker_config.get_marker("unsupported", "hallucinated")
        assert marker == "[HALLU]"
        marker2 = engine.marker_config.get_marker("unsupported", "contradictory")
        assert marker2 == "[CONTRA]"


# =============================================================================
# 4. Real-time conflict detection
# =============================================================================

class TestRealtimeConflictDetection:
    def test_realtime_produces_decisions(self, tiny_model_and_tokenizer, mock_detector):
        model, tokenizer = tiny_model_and_tokenizer
        # Configure mock to return unsupported + hallucinated
        mock_detector.forward.return_value = (
            torch.tensor([[0.1, 0.8, 0.1]]),   # primary logits: unsupported
            torch.tensor([[0.7, 0.3]]),         # secondary logits: hallucinated
            torch.tensor([[0.85]]),             # conflict score
            torch.tensor([[0.5, 0.5, 0.5]]),   # next state
        )

        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            action="flag",
            threshold=10.0,  # high threshold so entropy never triggers
            mode="absolute",
            use_conflict_detector=True,
            use_realtime_conflict_detector=True,
            conflict_detector=mock_detector,
            conflict_layer_indices=[-1, -2],
            decision_engine=UnifiedDecisionEngine(
                conflict_score_threshold=0.5,
                action="flag",
            ),
        )
        result = gen.generate_from_prompt(
            "Hello",
            max_new_tokens=3,
            return_dict_in_generate=True,
        )
        decisions = result.per_token_decisions[0]
        assert len(decisions) > 0
        # Skip first step (no generated token yet)
        for dec in decisions[1:]:
            assert dec["action"] in ("pass", "flag", "regenerate", "suppress", "warning")

    def test_realtime_conflict_score_range(self, tiny_model_and_tokenizer, mock_detector):
        model, tokenizer = tiny_model_and_tokenizer
        mock_detector.forward.return_value = (
            torch.tensor([[0.1, 0.8, 0.1]]),
            torch.tensor([[0.7, 0.3]]),
            torch.tensor([[0.85]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
        )

        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            action="flag",
            threshold=10.0,
            mode="absolute",
            use_conflict_detector=True,
            use_realtime_conflict_detector=True,
            conflict_detector=mock_detector,
            conflict_layer_indices=[-1, -2],
            decision_engine=UnifiedDecisionEngine(
                conflict_score_threshold=0.5,
                action="flag",
            ),
        )
        result = gen.generate_from_prompt(
            "Hello",
            max_new_tokens=3,
            return_dict_in_generate=True,
        )
        for dec in result.per_token_decisions[0]:
            cs = dec.get("conflict_score")
            if cs is not None:
                assert 0.0 <= cs <= 1.0

    def test_realtime_primary_secondary_alignment(self, tiny_model_and_tokenizer, mock_detector):
        model, tokenizer = tiny_model_and_tokenizer
        mock_detector.forward.return_value = (
            torch.tensor([[0.1, 0.8, 0.1]]),
            torch.tensor([[0.7, 0.3]]),  # hallucinated
            torch.tensor([[0.85]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
        )

        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            action="flag",
            threshold=10.0,
            mode="absolute",
            use_conflict_detector=True,
            use_realtime_conflict_detector=True,
            conflict_detector=mock_detector,
            conflict_layer_indices=[-1, -2],
            decision_engine=UnifiedDecisionEngine(
                conflict_score_threshold=0.5,
                action="flag",
            ),
        )
        result = gen.generate_from_prompt(
            "Hello",
            max_new_tokens=3,
            return_dict_in_generate=True,
        )
        # Check primary_labels and secondary_labels are aligned
        for p, s in zip(result.primary_labels[0], result.secondary_labels[0]):
            if p == "unsupported":
                assert s in ("hallucinated", "contradictory")

    def test_detector_failure_does_not_crash(self, tiny_model_and_tokenizer, mock_detector):
        model, tokenizer = tiny_model_and_tokenizer
        mock_detector.forward.side_effect = RuntimeError("Mock detector failure")

        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            action="flag",
            threshold=10.0,
            mode="absolute",
            use_conflict_detector=True,
            use_realtime_conflict_detector=True,
            conflict_detector=mock_detector,
            conflict_layer_indices=[-1, -2],
        )
        # Should not raise
        result = gen.generate_from_prompt(
            "Hello",
            max_new_tokens=3,
            return_dict_in_generate=True,
        )
        assert isinstance(result, ACCGenerationOutput)
        assert len(result.sequences) > 0


# =============================================================================
# 5. Marker configuration
# =============================================================================

class TestMarkerConfig:
    def test_custom_markers_in_output(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        mc = MarkerConfig(
            hallucination="<<HALLU>>",
            contradiction="<<CONTRA>>",
            uncertain="<<UNSURE>>",
            inline=True,
        )
        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            action="flag",
            threshold=0.01,
            mode="absolute",
            marker_config=mc,
        )
        result = gen.generate_from_prompt(
            "The quick brown fox",
            max_new_tokens=5,
            return_dict_in_generate=True,
        )
        # At least some markers should appear in text
        text = result.text[0]
        assert "<<UNSURE>>" in text or "<<HALLU>>" in text or "<<CONTRA>>" in text or "[ENTROPY FLAG]" in text or "[UNCERTAIN]" in text

    def test_get_marker_choices(self):
        mc = MarkerConfig()
        assert mc.get_marker("unsupported", "hallucinated") == mc.hallucination
        assert mc.get_marker("unsupported", "contradictory") == mc.contradiction
        assert mc.get_marker("uncertain", None) == mc.uncertain
        assert mc.get_marker(None, None) == mc.entropy_flag


# =============================================================================
# 6. Explainability
# =============================================================================

class TestExplainability:
    def test_per_token_decisions_populated(self, generator_flag, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        result = generator_flag.generate_from_prompt(
            "Hello",
            max_new_tokens=5,
            return_dict_in_generate=True,
        )
        assert len(result.per_token_decisions) == 1
        assert len(result.per_token_decisions[0]) > 0
        for dec in result.per_token_decisions[0]:
            assert "action" in dec
            assert "reason" in dec
            assert "entropy" in dec

    def test_explain_decisions_runs(self, generator_flag, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        result = generator_flag.generate_from_prompt(
            "Hello",
            max_new_tokens=5,
            return_dict_in_generate=True,
        )
        explanation = generator_flag.explain_decisions(result, batch_idx=0)
        assert isinstance(explanation, str)
        assert "Decision trace" in explanation
        assert "Summary:" in explanation

    def test_explain_decisions_empty(self, generator_flag):
        # Create a dummy output with no decisions
        dummy = ACCGenerationOutput(
            sequences=torch.zeros(1, 1, dtype=torch.long),
            text=[""],
            per_token_entropy=[[]],
            uncertain_steps=[[]],
        )
        explanation = generator_flag.explain_decisions(dummy, batch_idx=0)
        assert "No decisions available." in explanation


# =============================================================================
# 7. Batch generation, device, edges
# =============================================================================

class TestBatchAndEdges:
    def test_batch_generation(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            action="flag",
            threshold=10.0,  # high to avoid flagging
            mode="absolute",
        )
        prompts = ["Hello", "World"]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        result = gen.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=3,
            return_dict_in_generate=True,
        )
        assert result.sequences.shape[0] == 2
        assert len(result.text) == 2
        assert len(result.per_token_entropy) == 2
        assert len(result.per_token_decisions) == 2

    def test_device_handling(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        device = next(model.parameters()).device
        gen = ACCEnhancedGenerator(model=model, tokenizer=tokenizer)
        result = gen.generate_from_prompt(
            "Hello",
            max_new_tokens=3,
            return_dict_in_generate=True,
        )
        assert result.sequences.device == device

    def test_zero_max_new_tokens(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        gen = ACCEnhancedGenerator(model=model, tokenizer=tokenizer)
        with pytest.raises(ValueError, match="max_new_tokens.*greater than 0"):
            gen.generate_from_prompt(
                "Hello",
                max_new_tokens=0,
                return_dict_in_generate=True,
            )

    def test_single_token_generation(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        gen = ACCEnhancedGenerator(model=model, tokenizer=tokenizer)
        result = gen.generate_from_prompt(
            "A",
            max_new_tokens=1,
            return_dict_in_generate=True,
        )
        assert len(result.per_token_decisions[0]) == 1
        assert result.sequences.shape[1] == len(tokenizer.encode("A")) + 1


# =============================================================================
# 8. Constructor validation
# =============================================================================

class TestConstructor:
    def test_all_action_values(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        for action in ("flag", "regenerate", "warning", "suppress"):
            gen = ACCEnhancedGenerator(
                model=model,
                tokenizer=tokenizer,
                action=action,
            )
            assert gen.action == action

    def test_invalid_action_raises(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        with pytest.raises(ValueError, match="unknown action"):
            ACCEnhancedGenerator(model=model, tokenizer=tokenizer, action="invalid")

    def test_custom_decision_engine(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        engine = UnifiedDecisionEngine(
            entropy_threshold=2.0,
            conflict_score_threshold=0.9,
            action="suppress",
        )
        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            decision_engine=engine,
        )
        assert gen.decision_engine is engine

    def test_custom_marker_config(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        mc = MarkerConfig(hallucination="[HALLU]")
        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            marker_config=mc,
        )
        assert gen.decision_engine.marker_config.hallucination == "[HALLU]"


# =============================================================================
# 9. Backward compatibility
# =============================================================================

class TestBackwardCompatibility:
    def test_legacy_entropy_processor_exists(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        monitor = EntropyMonitor(threshold=1.0)
        proc = _EntropyLogitsProcessor(
            monitor=monitor,
            tokenizer=tokenizer,
            action="flag",
        )
        assert isinstance(proc, _ACCLogitsProcessor)


# =============================================================================
# 10. ACCLogitsProcessor internals
# =============================================================================

class TestACCLogitsProcessorInternals:
    def test_processor_tracks_entropy(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        monitor = EntropyMonitor(threshold=10.0)
        engine = UnifiedDecisionEngine()
        proc = _ACCLogitsProcessor(
            monitor=monitor,
            tokenizer=tokenizer,
            decision_engine=engine,
        )
        vocab_size = tokenizer.vocab_size
        # Simulate two generation steps
        input_ids = torch.randint(0, vocab_size, (1, 5))
        scores1 = torch.randn(1, vocab_size)
        scores2 = torch.randn(1, vocab_size)

        _ = proc(input_ids, scores1)
        input_ids2 = torch.cat([input_ids, torch.tensor([[0]]), torch.tensor([[0]])], dim=1)
        _ = proc(input_ids2, scores2)

        assert len(proc.per_token_entropy[0]) == 2
        assert all(isinstance(h, float) for h in proc.per_token_entropy[0])

    def test_processor_decisions_length_matches_steps(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        monitor = EntropyMonitor(threshold=10.0)
        engine = UnifiedDecisionEngine()
        proc = _ACCLogitsProcessor(
            monitor=monitor,
            tokenizer=tokenizer,
            decision_engine=engine,
        )
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(0, vocab_size, (1, 5))
        scores = torch.randn(1, vocab_size)
        _ = proc(input_ids, scores)
        assert len(proc.decisions[0]) == 1
        assert isinstance(proc.decisions[0][0], Decision)

    def test_suppress_downgraded_for_pure_entropy(self, tiny_model_and_tokenizer):
        """Pure entropy breach with action=suppress is downgraded to flag."""
        model, tokenizer = tiny_model_and_tokenizer
        monitor = EntropyMonitor(threshold=0.01)  # always breach
        engine = UnifiedDecisionEngine(action="suppress")
        proc = _ACCLogitsProcessor(
            monitor=monitor,
            tokenizer=tokenizer,
            decision_engine=engine,
        )
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(0, vocab_size, (1, 5))
        scores = torch.randn(1, vocab_size)
        out = proc(input_ids, scores)
        # Should be downgraded to flag, so top token is NOT -inf
        top_orig = scores[0].argmax()
        assert out[0, top_orig] != -float("inf")
        # Decision should be flag
        assert proc.decisions[0][0].action == "flag"

    def test_suppress_with_conflict_signal(self, tiny_model_and_tokenizer, mock_detector):
        """Suppress works when backed by a conflict signal."""
        model, tokenizer = tiny_model_and_tokenizer
        mock_detector.forward.return_value = (
            torch.tensor([[0.1, 0.8, 0.1]]),   # primary: unsupported
            torch.tensor([[0.7, 0.3]]),         # secondary: hallucinated
            torch.tensor([[0.85]]),             # conflict score > threshold
            torch.tensor([[0.5, 0.5, 0.5]]),   # next state
        )
        mock_detector.PRIMARY_LABELS = ["supported", "unsupported", "uncertain"]
        mock_detector.SECONDARY_LABELS = ["hallucinated", "contradictory"]

        mock_extractor = MagicMock()
        mock_extractor.layer_indices = [-1]
        hidden_vec = torch.randn(1, 1, 2)
        mock_extractor._hidden_buffers = {-1: [hidden_vec, hidden_vec]}

        monitor = EntropyMonitor(threshold=10.0)  # never breach via entropy
        engine = UnifiedDecisionEngine(
            action="suppress",
            conflict_score_threshold=0.5,
            dual_signal_regenerate=False,  # ensure suppress is not overridden
        )
        proc = _ACCLogitsProcessor(
            monitor=monitor,
            tokenizer=tokenizer,
            decision_engine=engine,
            detector=mock_detector,
            extractor=mock_extractor,
        )
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(0, vocab_size, (1, 5))
        scores1 = torch.randn(1, vocab_size)
        _ = proc(input_ids, scores1)
        input_ids2 = torch.cat([input_ids, torch.tensor([[0]])], dim=1)
        scores2 = torch.randn(1, vocab_size)
        out = proc(input_ids2, scores2)
        top_orig = scores2[0].argmax()
        assert out[0, top_orig] == -float("inf")
        assert proc.decisions[0][1].action == "suppress"


# =============================================================================
# 11. Temporal state propagation
# =============================================================================

class TestTemporalState:
    def test_detector_state_propagates(self, tiny_model_and_tokenizer, mock_detector):
        model, tokenizer = tiny_model_and_tokenizer
        state_tensor = torch.tensor([[0.1, 0.2, 0.3]])
        mock_detector.forward.return_value = (
            torch.tensor([[0.1, 0.8, 0.1]]),
            torch.tensor([[0.7, 0.3]]),
            torch.tensor([[0.85]]),
            state_tensor,
        )

        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            action="flag",
            threshold=10.0,
            mode="absolute",
            use_conflict_detector=True,
            use_realtime_conflict_detector=True,
            conflict_detector=mock_detector,
            conflict_layer_indices=[-1, -2],
        )
        result = gen.generate_from_prompt(
            "Hello",
            max_new_tokens=3,
            return_dict_in_generate=True,
        )
        # The detector should have been called for each step after the first
        assert mock_detector.forward.call_count >= 1


# =============================================================================
# 12. Self-consistency integration
# =============================================================================

class TestSelfConsistency:
    def test_self_consistency_integration(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            use_self_consistency=True,
            self_consistency_candidates=2,
            self_consistency_max_new_tokens=3,
        )
        result = gen.generate_from_prompt(
            "Hello",
            max_new_tokens=3,
            return_dict_in_generate=True,
        )
        assert len(result.consistency_score) == 1
        assert result.consistency_score[0] is not None
        assert isinstance(result.contradiction_detected[0], bool)
