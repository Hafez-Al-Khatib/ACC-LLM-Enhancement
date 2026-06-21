"""Baseline hallucination detection methods for comparison.

Implements:
- DoLa (DOLA): Contrastive decoding + post-hoc early/late layer logit contrast
- SAPLMA: MLP classifier on last-layer hidden states
- Entropy: Threshold on output distribution entropy
- SelfCheckGPT: Self-consistency across multiple sampled generations

All baselines share the same interface for fair comparison.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Ensure NLTK sentence tokenizer data is available; fall back to regex if offline.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass


class DoLaDetector:
    """Simplified DoLa: detects hallucinations via early/late layer logit contrast.

    Reference: Chuang et al. "DoLa: Decoding by Contrasting Layers Improves
    Factuality in Large Language Models" (ICLR 2024).

    Our post-hoc version: for each generated token, compare the distribution
    from early (premature) layers vs. late (mature) layers. High KL divergence
    indicates the model "changed its mind" — a signature of hallucination.

    Also supports generation-time contrastive decoding via
    :meth:`generate_contrastive`.
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

    @staticmethod
    def generate_contrastive(
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        device: str,
        seed: int,
        premature_layers: Optional[List[int]] = None,
        mature_layers: Optional[List[int]] = None,
        alpha: float = 0.1,
        temperature: float = 0.8,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate text using DoLa-style contrastive decoding.

        At each step we compute mature logits (final layer) and average the
        logits of the premature layers. The next token is sampled from the
        contrastive distribution:

            logits_contrast = mature + alpha * (premature - mature)

        where ``premature`` is the averaged premature-layer logit. This
        generation-time contrastive objective is distinct from the post-hoc
        JS-divergence detector in :meth:`detect_sequence`.

        Args:
            model: Causal LM.
            tokenizer: Tokenizer matching ``model``.
            prompt: Text prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            device: Target device ("cuda", "xpu", or "cpu").
            seed: Random seed for sampling.
            premature_layers: Layer indices treated as premature. Defaults to
                the first half of the model layers.
            mature_layers: Layer indices treated as mature. Defaults to the
                second half of the model layers (i.e. the final layer logits).
            alpha: Contrastive scaling factor.
            temperature: Sampling temperature.
            top_p: Optional nucleus cutoff.

        Returns:
            Generated text string (excluding the prompt).
        """
        torch.manual_seed(seed)
        model.eval()

        n_layers = model.config.num_hidden_layers
        if premature_layers is None:
            premature_layers = list(range(0, n_layers // 2))
        if mature_layers is None:
            mature_layers = list(range(n_layers // 2, n_layers))

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = model.config.eos_token_id

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids, output_hidden_states=True)

                # Mature logits from the final layer output
                mature_logits = outputs.logits[0, -1, :]  # (vocab_size,)

                # Average premature-layer logits projected from hidden states
                premature_logits_list = []
                for layer_idx in premature_layers:
                    h = outputs.hidden_states[layer_idx][:, -1, :]  # (1, hidden_dim)
                    logits = model.lm_head(h)[0]  # (vocab_size,)
                    premature_logits_list.append(logits)
                premature_logits = torch.stack(premature_logits_list).mean(dim=0)

                # Contrastive logits
                contrastive_logits = mature_logits + alpha * (premature_logits - mature_logits)

                # Temperature scaling
                logits = contrastive_logits / temperature
                probs = F.softmax(logits, dim=-1)

                # Optional top-p filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=0)
                    mask = cumsum <= top_p
                    mask[1:] = mask[:-1].clone()
                    mask[0] = True
                    filtered_probs = sorted_probs * mask.to(sorted_probs.dtype)
                    filtered_probs = filtered_probs / filtered_probs.sum()
                    probs = torch.zeros_like(probs)
                    probs.scatter_(0, sorted_indices, filtered_probs)

                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                if next_token.item() == eos_token_id:
                    break

        return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

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

    Trains a small MLP on last-layer hidden states extracted from prompts
    labeled as factual (0) or hallucinated (1). Includes a proper train/val
    split, checkpointing, and generation-time detection.
    """

    def __init__(self, hidden_dim: int, device: str = "cpu"):
        super().__init__()
        self.hidden_dim = hidden_dim
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
        # Ensure float32 input to match MLP weights (handles float16/bfloat16 models)
        return self.mlp(hidden_state.float()).squeeze(-1)

    def predict_from_hidden(self, last_hidden: torch.Tensor) -> float:
        """Return hallucination probability for a single last-layer hidden state."""
        self.eval()
        with torch.no_grad():
            logit = self.forward(last_hidden.to(self.device))
            prob = torch.sigmoid(logit)
        return prob.item()

    def _extract_hidden(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract the last-layer hidden state at the end of ``prompt``.

        If ``max_new_tokens`` is provided, a short continuation is generated
        and the hidden state of its last generated token is returned instead.
        """
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            if max_new_tokens is None or max_new_tokens <= 0:
                outputs = model(input_ids, output_hidden_states=True)
                return outputs.hidden_states[-1][0, -1, :].detach().cpu().float()

            # Generate a short continuation and use its final hidden state.
            for _ in range(max_new_tokens):
                outputs = model(input_ids, output_hidden_states=True)
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
            final_outputs = model(input_ids, output_hidden_states=True)
            return final_outputs.hidden_states[-1][0, -1, :].detach().cpu().float()

    def train_on_data(
        self,
        model,
        tokenizer,
        train_samples: List[Dict[str, object]],
        epochs: int = 30,
        lr: float = 1e-3,
        val_split: float = 0.2,
        batch_size: int = 4,
        patience: int = 5,
        continuation_tokens: Optional[int] = None,
    ) -> Dict[str, float]:
        """Train the SAPLMA detector on labeled prompts.

        Args:
            model: Causal LM used to extract hidden states.
            tokenizer: Tokenizer matching ``model``.
            train_samples: List of dicts with keys ``prompt`` (str) and
                ``label`` (int; 0 = factual, 1 = hallucinated).
            epochs: Number of training epochs.
            lr: Learning rate for Adam.
            val_split: Fraction of data held out for validation.
            batch_size: Mini-batch size.
            patience: Early-stopping patience (epochs without val improvement).
            continuation_tokens: If set, generate this many tokens from each
                prompt and extract the hidden state of the final generated
                token. Otherwise the prompt's final hidden state is used.

        Returns:
            Dict with ``best_val_loss`` and ``final_val_loss``.
        """
        if not train_samples:
            raise ValueError("train_samples must not be empty")

        model.eval()
        self.train()

        # Extract features once; this is the costly part.
        features = []
        labels = []
        for sample in train_samples:
            prompt = sample["prompt"]
            label = float(sample.get("label", 0))
            hidden = self._extract_hidden(
                model, tokenizer, prompt, max_new_tokens=continuation_tokens
            )
            features.append(hidden)
            labels.append(label)

        X = torch.stack(features).float()
        y = torch.tensor(labels, dtype=torch.float32)

        # Train/val split
        n = len(X)
        n_val = max(1, int(n * val_split))
        n_train = n - n_val
        indices = torch.randperm(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train = X[train_idx].to(self.device)
        y_train = y[train_idx].to(self.device)
        X_val = X[val_idx].to(self.device)
        y_val = y[val_idx].to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        def make_batches(tensor_x, tensor_y, bs):
            for i in range(0, len(tensor_x), bs):
                yield tensor_x[i : i + bs], tensor_y[i : i + bs]

        for epoch in range(epochs):
            self.train()
            train_losses = []
            for xb, yb in make_batches(X_train, y_train, batch_size):
                optimizer.zero_grad()
                logits = self.forward(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            self.eval()
            with torch.no_grad():
                val_logits = self.forward(X_val)
                val_loss = criterion(val_logits, y_val).item()

            logger.info(
                "SAPLMA epoch %d/%d | train_loss=%.4f | val_loss=%.4f",
                epoch + 1,
                epochs,
                np.mean(train_losses),
                val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info("SAPLMA early stopping at epoch %d", epoch + 1)
                break

        if best_state is not None:
            self.load_state_dict(best_state)
            self.to(self.device)

        self.eval()
        return {"best_val_loss": float(best_val_loss), "final_val_loss": float(val_loss)}

    def save_checkpoint(self, path: str):
        """Save detector weights and hyperparameters."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hidden_dim": self.hidden_dim,
                "device": self.device,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load detector weights from ``path``."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])
        self.to(self.device)

    def train_on_examples(
        self,
        model,
        tokenizer,
        factual_prompts: List[str],
        hallucinated_prompts: List[str],
        max_new_tokens: int = 10,
        epochs: int = 10,
    ):
        """Backward-compatible inline training on two prompt lists.

        Internally delegates to :meth:`train_on_data`.
        """
        samples = []
        for p in factual_prompts:
            samples.append({"prompt": p, "label": 0})
        for p in hallucinated_prompts:
            samples.append({"prompt": p, "label": 1})
        return self.train_on_data(
            model,
            tokenizer,
            samples,
            epochs=epochs,
            continuation_tokens=max_new_tokens,
        )

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
                last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().float()  # (hidden_dim,)

                prob = self.predict_from_hidden(last_hidden)

                # Sample next token
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                results.append({
                    "token_id": next_token.item(),
                    "prob": prob,
                    "is_hallucination": prob > threshold,
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


class SelfCheckGPTDetector:
    """SelfCheckGPT-style consistency detector.

    Generates multiple answers for a prompt, splits them into sentences, and
    scores each sentence by its maximum semantic similarity to sentences in
    the other answers. Low consistency indicates a likely hallucination.

    Reference: Manakul et al. "SelfCheckGPT: Zero-Resource Black-Box
    Hallucination Detection for Generative Large Language Models" (2023).
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        sentence_encoder=None,
        n_samples: int = 5,
        temperature: float = 0.8,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sentence_encoder = sentence_encoder
        self.n_samples = n_samples
        self.temperature = temperature
        self.device = device

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences, falling back to regex if NLTK fails."""
        try:
            return nltk.sent_tokenize(text)
        except Exception:
            # Simple regex fallback: split on punctuation followed by whitespace
            chunks = re.split(r"(?<=[.!?])\s+", text)
            return [c.strip() for c in chunks if c.strip()]

    def _embed_sentences(self, sentences: List[str]) -> torch.Tensor:
        """Return L2-normalized sentence embeddings."""
        if not sentences:
            return torch.zeros((0, 1))

        if self.sentence_encoder is not None:
            embeddings = self.sentence_encoder.encode(
                sentences,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            embeddings = embeddings.to(self.device)
        else:
            embeddings = self._embed_with_base_model(sentences)

        return F.normalize(embeddings, p=2, dim=-1)

    def _embed_with_base_model(self, sentences: List[str]) -> torch.Tensor:
        """Mean-pool last-layer hidden states from the base causal LM."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("base model and tokenizer are required when no sentence_encoder is provided")

        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for sent in sentences:
                inputs = self.tokenizer(sent, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
                mask = inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
                summed = (last_hidden * mask).sum(dim=1)
                mean_pooled = summed / mask.sum(dim=1).clamp(min=1.0)
                embeddings.append(mean_pooled[0].cpu())
        return torch.stack(embeddings).to(self.device)

    def score_samples(
        self,
        samples: List[str],
    ) -> Tuple[float, List[Dict]]:
        """Compute consistency score and per-sentence breakdown.

        Args:
            samples: List of generated texts for the same prompt.

        Returns:
            ``(consistency_score, sentence_scores)`` where ``consistency_score``
            is in [0, 1] (higher = more consistent) and ``sentence_scores`` is
            a list of dicts with ``sentence``, ``score``, and ``max_similarity``.
        """
        if len(samples) < 2:
            return 1.0, []

        sentence_lists = [self._split_sentences(s) for s in samples]
        all_sentences = [s for sl in sentence_lists for s in sl]
        if not all_sentences:
            return 1.0, []

        # Embed each sample's sentences separately so we can compare across samples.
        sample_embeddings = [self._embed_sentences(sl) for sl in sentence_lists]

        per_sentence_scores = []
        sample_max_scores = []
        for sample_idx, (sentences, embeddings) in enumerate(zip(sentence_lists, sample_embeddings)):
            if len(sentences) == 0:
                continue
            # Other samples' embeddings
            other_embeddings = torch.cat(
                [e for j, e in enumerate(sample_embeddings) if j != sample_idx and e.shape[0] > 0],
                dim=0,
            )
            if other_embeddings.shape[0] == 0:
                continue

            # cosine similarity via dot product of L2-normalized vectors
            sims = embeddings @ other_embeddings.T  # (n_sentences, n_other_sentences)
            max_sims, _ = sims.max(dim=1)
            sample_max_scores.extend(max_sims.tolist())
            for sent, score in zip(sentences, max_sims.tolist()):
                per_sentence_scores.append({
                    "sentence": sent,
                    "score": float(score),
                    "max_similarity": float(score),
                    "sample_index": sample_idx,
                })

        consistency = float(np.mean(sample_max_scores)) if sample_max_scores else 1.0
        return consistency, per_sentence_scores

    def detect_sequence(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        device: str,
        seed: int,
    ) -> Dict:
        """SelfCheckGPT-style detection wrapped in the baseline interface.

        Generates ``n_samples`` answers, scores sentence-level consistency, and
        returns a dict compatible with the benchmark evaluation pipeline.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        samples = []
        for i in range(self.n_samples):
            text = self._generate_sample(model, tokenizer, prompt, max_new_tokens, device, seed + i * 1000)
            samples.append(text)

        consistency, sentence_scores = self.score_samples(samples)
        return {
            "samples": samples,
            "consistency": consistency,
            "sentence_scores": sentence_scores,
            "is_hallucination": consistency < 0.5,
        }

    def _generate_sample(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        device: str,
        seed: int,
    ) -> str:
        """Generate one sample answer."""
        torch.manual_seed(seed)
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
        eos_token_id = tokenizer.eos_token_id or model.config.eos_token_id

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == eos_token_id:
                    break

        return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)


def _get_device() -> str:
    """Return the best available torch device string."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "xpu", None) and torch.xpu.is_available():
        return "xpu"
    return "cpu"
