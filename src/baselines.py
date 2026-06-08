"""Baseline hallucination detection methods for comparison.

Implements:
- DoLa (DOLA): Post-hoc contrast of early vs. late layer logits
- SAPLMA: MLP classifier on last-layer hidden states
- Entropy: Threshold on output distribution entropy

All baselines share the same interface for fair comparison.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoLaDetector:
    """Simplified DoLa: detects hallucinations via early/late layer logit contrast.

    Reference: Chuang et al. "DoLa: Decoding by Contrasting Layers Improves
    Factuality in Large Language Models" (ICLR 2024).

    Our post-hoc version: for each generated token, compare the distribution
    from early (premature) layers vs. late (mature) layers. High KL divergence
    indicates the model "changed its mind" — a signature of hallucination.
    """

    def __init__(
        self,
        model,
        premature_layers: List[int] = None,
        mature_layers: List[int] = None,
        threshold: float = 0.1,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.threshold = threshold

        n_layers = model.config.num_hidden_layers
        self.premature_layers = premature_layers or list(range(0, n_layers // 2))
        self.mature_layers = mature_layers or list(range(n_layers // 2, n_layers))

    def _get_layer_logits(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Project hidden states to logits using the LM head."""
        # hidden_states is a tuple of (batch, seq_len, hidden_dim)
        # We want logits for the last token
        last_hidden = hidden_states[-1][:, -1, :]  # (batch, hidden_dim)
        logits = self.model.lm_head(last_hidden)  # (batch, vocab_size)
        return logits

    def detect_sequence(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 15,
    ) -> List[Dict]:
        """Generate and detect hallucinations token-by-token.

        Returns list of dicts with:
            token_id, token_text, kl_divergence, is_hallucination
        """
        results = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids, output_hidden_states=True)

                # Get logits from mature (final) layer
                mature_logits = outputs.logits[0, -1, :]  # (vocab_size,)

                # Get logits from premature layers by projecting their hidden states
                premature_logits_list = []
                for layer_idx in self.premature_layers:
                    h = outputs.hidden_states[layer_idx][:, -1, :]  # (1, hidden_dim)
                    logits = self.model.lm_head(h)[0]  # (vocab_size,)
                    premature_logits_list.append(logits)

                # Average premature logits
                premature_logits = torch.stack(premature_logits_list).mean(dim=0)

                # Compute KL divergence: KL(premature || mature)
                # High KL means the distributions differ significantly
                p_prem = F.softmax(premature_logits, dim=-1)
                p_mat = F.softmax(mature_logits, dim=-1)

                # Jensen-Shannon divergence (symmetric, bounded)
                m = 0.5 * (p_prem + p_mat)
                kl_pm = F.kl_div(m.log(), p_prem, reduction="sum")
                kl_mm = F.kl_div(m.log(), p_mat, reduction="sum")
                js_div = 0.5 * (kl_pm + kl_mm)

                # Sample next token using mature logits
                probs = F.softmax(mature_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                results.append({
                    "token_id": next_token.item(),
                    "js_divergence": js_div.item(),
                    "is_hallucination": js_div.item() > self.threshold,
                })

                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == self.model.config.eos_token_id:
                    break

        return results


class SAPLMADetector(nn.Module):
    """SAPLMA-style detector: MLP on last-layer hidden states.

    Reference: Azaria & Mitchell "The Internal State of an LLM Knows When
    It's Lying" (2023).

    Simplified version: single-layer MLP trained inline on a small set of
    examples. Uses last-layer hidden states as features.
    """

    def __init__(self, hidden_dim: int, device: str = "cpu"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        self.device = device
        self.to(device)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """hidden_state: (..., hidden_dim)"""
        return self.mlp(hidden_state).squeeze(-1)

    def train_on_examples(
        self,
        model,
        tokenizer,
        factual_prompts: List[str],
        hallucinated_prompts: List[str],
        max_new_tokens: int = 10,
        epochs: int = 10,
    ):
        """Quick inline training on a small set of examples."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        features = []
        labels = []

        # Collect features from factual prompts (label=0)
        for prompt in factual_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][0, -1, :].cpu()  # (hidden_dim,)
            features.append(last_hidden)
            labels.append(0.0)

        # Collect features from hallucinated prompts (label=1)
        for prompt in hallucinated_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][0, -1, :].cpu()  # (hidden_dim,)
            features.append(last_hidden)
            labels.append(1.0)

        X = torch.stack(features).to(self.device)
        y = torch.tensor(labels, dtype=torch.float32).to(self.device)

        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.forward(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        self.eval()

    def detect_sequence(
        self,
        model,
        input_ids: torch.Tensor,
        max_new_tokens: int = 15,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """Generate and detect."""
        results = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1][0, -1, :].cpu()  # (hidden_dim,)

                logit = self.forward(last_hidden.to(self.device))
                prob = torch.sigmoid(logit)

                # Sample next token
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                results.append({
                    "token_id": next_token.item(),
                    "prob": prob.item(),
                    "is_hallucination": prob.item() > threshold,
                })

                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == model.config.eos_token_id:
                    break

        return results


class EntropyDetector:
    """Simple entropy-based detector.

    Flags tokens with entropy above a calibrated threshold.
    """

    def __init__(self, threshold: float = 3.9):
        self.threshold = threshold

    def detect_sequence(
        self,
        model,
        input_ids: torch.Tensor,
        max_new_tokens: int = 15,
    ) -> List[Dict]:
        """Generate and detect."""
        results = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]

                probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs + 1e-12)
                entropy = (-(probs * log_probs).sum()).item()

                # Sample next token
                probs_sample = F.softmax(logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs_sample, num_samples=1)

                results.append({
                    "token_id": next_token.item(),
                    "entropy": entropy,
                    "is_hallucination": entropy > self.threshold,
                })

                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == model.config.eos_token_id:
                    break

        return results
