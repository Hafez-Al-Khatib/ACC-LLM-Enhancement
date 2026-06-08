"""Lightweight intervention engine for ACC conflict detection.

Uses manual step-by-step generation to avoid memory issues with
output_hidden_states=True on model.generate().
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ACCInterventionEngine:
    """Post-hoc conflict detection with conditional regeneration.

    Memory-optimized: uses step-by-step forward passes instead of
    model.generate(output_hidden_states=True) which stores all layers
    for all tokens.
    """

    def __init__(
        self,
        detector,
        conflict_threshold: float = 0.5,
        relative_threshold: Optional[float] = None,
        calibration_tokens: int = 3,
        max_regenerations: int = 1,
        temperature_bump: float = 0.3,
        top_p_reduce: float = 0.1,
        uncertainty_prompts: Optional[List[str]] = None,
    ):
        self.detector = detector
        self.conflict_threshold = conflict_threshold
        self.relative_threshold = relative_threshold
        self.calibration_tokens = calibration_tokens
        self.max_regenerations = max_regenerations
        self.temperature_bump = temperature_bump
        self.top_p_reduce = top_p_reduce
        self.uncertainty_prompts = uncertainty_prompts or [
            "Wait, let me reconsider. ",
            "Actually, I should be careful here. ",
            "I'm not entirely certain, but ",
        ]

    def _generate_with_scores(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        device: str,
        seed: int = 42,
    ) -> Tuple[List[float], str]:
        """Step-by-step generation returning conflict scores + text."""
        torch.manual_seed(seed)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        generated_ids = []
        conflict_scores = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids, output_hidden_states=True)

                # Extract last token hidden states for all layers
                last_token_hs = {}
                for layer_idx, hs in enumerate(outputs.hidden_states):
                    # hs: (batch=1, seq_len, hidden_dim)
                    h = hs[0, -1, :].detach().cpu()
                    last_token_hs[layer_idx] = h
                    last_token_hs[layer_idx - len(outputs.hidden_states)] = h

                # Run detector
                detector_out = self.detector.forward(last_token_hs)
                score = float(detector_out[2][0].item())
                conflict_scores.append(score)

                # Sample next token
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits / temperature, dim=-1)

                # Top-p filtering (memory-efficient, dtype-safe)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                mask = cumsum <= top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = True
                # Keep only tokens within top-p
                filtered_probs = sorted_probs * mask.to(sorted_probs.dtype)
                filtered_probs = filtered_probs / filtered_probs.sum()
                # Map back to original indices
                probs = torch.zeros_like(probs)
                probs.scatter_(0, sorted_indices, filtered_probs)

                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids.append(next_token.item())

                # Stop at EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

                # Append to input for next step
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return conflict_scores, text

    def generate_with_intervention(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 0.8,
        top_p: float = 0.95,
        device: str = "cpu",
        seed: int = 42,
    ) -> Dict:
        """Generate text with ACC-based intervention."""
        # Phase 1: Draft generation
        conflict_scores, draft_text = self._generate_with_scores(
            model, tokenizer, prompt, max_new_tokens, temperature, top_p, device, seed
        )

        if not conflict_scores:
            return {
                "text": draft_text, "draft_text": draft_text, "intervened": False,
                "conflict_scores": [], "max_conflict": 0.0,
                "calibrated_threshold": self.conflict_threshold,
                "num_regenerations": 0, "reason": "no_scores",
            }

        max_conflict = max(conflict_scores)
        avg_conflict = sum(conflict_scores) / len(conflict_scores)

        # Phase 2: Calibrate threshold
        if self.relative_threshold is not None and len(conflict_scores) >= self.calibration_tokens:
            baseline = sum(conflict_scores[:self.calibration_tokens]) / self.calibration_tokens
            effective_threshold = baseline * self.relative_threshold
        else:
            effective_threshold = self.conflict_threshold

        # Phase 3: Decide
        if max_conflict < effective_threshold:
            return {
                "text": draft_text, "draft_text": draft_text, "intervened": False,
                "conflict_scores": conflict_scores, "max_conflict": max_conflict,
                "avg_conflict": avg_conflict, "calibrated_threshold": effective_threshold,
                "num_regenerations": 0, "reason": "no_conflict",
            }

        # Phase 4: Regenerate with control
        best_text = draft_text
        best_max_conflict = max_conflict
        num_regens = 0

        for regen_idx in range(self.max_regenerations):
            uncertainty_phrase = self.uncertainty_prompts[regen_idx % len(self.uncertainty_prompts)]
            modified_prompt = prompt + uncertainty_phrase

            regen_scores, regen_text = self._generate_with_scores(
                model, tokenizer, modified_prompt, max_new_tokens,
                min(temperature + self.temperature_bump, 1.5),
                max(top_p - self.top_p_reduce, 0.5),
                device, seed=seed + 1000 + regen_idx,
            )
            num_regens += 1

            if regen_scores:
                regen_max = max(regen_scores)
                if regen_max < best_max_conflict:
                    best_text = uncertainty_phrase + regen_text
                    best_max_conflict = regen_max
            elif len(regen_text) > 0:
                best_text = uncertainty_phrase + regen_text

        return {
            "text": best_text, "draft_text": draft_text, "intervened": True,
            "conflict_scores": conflict_scores, "max_conflict": max_conflict,
            "avg_conflict": avg_conflict, "calibrated_threshold": effective_threshold,
            "num_regenerations": num_regens, "reason": "conflict_detected",
        }

    def generate_simple_baseline(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 0.8,
        top_p: float = 0.95,
        device: str = "cpu",
        seed: int = 42,
    ) -> str:
        """Simple baseline generation."""
        torch.manual_seed(seed)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=top_p, pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
