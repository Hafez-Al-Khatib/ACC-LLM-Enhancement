import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, LogitsProcessor
from typing import Optional, Dict, List, Tuple
import json
import warnings


class LatentConflictDetector(nn.Module):
    """Approach B: Detects hallucination/conflict from model's hidden states.

    Inspired by the ACC's role in detecting conflict between expected and actual
    outcomes. This small MLP classifier sits on top of the LLM's intermediate
    hidden states and outputs a classification:
        [supported, hallucinated, uncertain, contradictory]

    Architecture: 2-layer MLP with ~100K parameters.
    Trained on pairs of (hidden_state, label) extracted during generation.
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
        """
        Args:
            hidden_states: (batch, hidden_dim) per-token hidden state from LLM
        Returns:
            logits: (batch, num_labels)
        """
        return self.mlp(hidden_states)

    def classify(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """Classify per-token hidden states.

        Args:
            hidden_states: (batch, hidden_dim) or (hidden_dim,)
        Returns:
            Dict of label -> probability
        """
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(hidden_states)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return {label: float(prob) for label, prob in zip(self.LABELS, probs)}

    def get_conflict_score(self, hidden_states: torch.Tensor) -> float:
        """Return a scalar conflict score (0=supported, 1=hallucinated/uncertain)."""
        probs = self.classify(hidden_states)
        return probs["hallucinated"] + probs["uncertain"] + probs["contradictory"]


class HiddenStateExtractor:
    """Extracts hidden states from a LLM during a standard forward pass.

    This is kept for backward compatibility and for use when you only need
    prompt-encoding states.  For generation-time extraction see
    :class:`GenerationHiddenStateExtractor`.
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
        """Register forward hook to capture hidden states."""
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
        """Return captured hidden states and clear buffer."""
        states = self.hidden_states
        self.hidden_states = []
        return states


class GenerationHiddenStateExtractor(LogitsProcessor):
    """Extracts hidden states for **newly generated tokens** during ``model.generate()``.

    Works by combining a forward hook (captures hidden states) with a
    :class:`~transformers.LogitsProcessor` (tracks generation steps).  The
    hidden state for the **last position** of each forward pass corresponds to
    the token whose logits are being produced at that step, i.e. the newly
    generated token.

    Usage::

        extractor = GenerationHiddenStateExtractor(model, layer_idx=-4)
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

    Args:
        model: A HuggingFace causal LM.
        layer_idx: Which transformer layer to tap (-1 = last, -4 = near output).
    """

    def __init__(self, model: AutoModelForCausalLM, layer_idx: int = -4):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        # Each entry is (batch, 1, hidden_dim) — the hidden state for the
        # position that produced the next-token logits.
        self._hidden_buffer: List[torch.Tensor] = []
        self._hook = None
        self._prompt_len = 0
        self._step = 0
        self.register_hook()

    # ------------------------------------------------------------------ hook

    def _get_target_layer(self):
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            raise ValueError("Unsupported model architecture")
        return layers[self.layer_idx]

    def register_hook(self):
        """Register forward hook to capture last-position hidden states."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            # The *last* position of this forward pass is what produces the
            # next-token logits.  With use_cache=True this is usually a single
            # new token; without cache it is the last position of the full seq.
            last_hidden = output[:, -1:, :].detach().cpu()  # (batch, 1, hidden)
            self._hidden_buffer.append(last_hidden)

        target_layer = self._get_target_layer()
        self._hook = target_layer.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self._hook:
            self._hook.remove()
            self._hook = None

    # ------------------------------------------------------------------ LogitsProcessor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Called after every model forward during generation.

        We only need to record the prompt length on the very first call so
        that ``get_records`` can map hidden states to generated token IDs.
        """
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
        """Pair captured hidden states with the final generated token IDs.

        Args:
            sequences: (batch, total_seq_len) full sequence returned by
                ``model.generate()``.
            prompt_len: Length of the original prompt.  If ``None``, the value
                captured during the first logits-processor step is used.

        Returns:
            A list of dicts, one per generated token, with keys:
            ``step``, ``token_id``, ``token_position``, ``hidden_state``.
        """
        if prompt_len is None:
            prompt_len = self._prompt_len
        if prompt_len == 0:
            raise ValueError(
                "prompt_len is 0 — was the extractor actually used inside generation?"
            )

        batch_size = sequences.shape[0]
        if batch_size != 1:
            raise ValueError(
                f"GenerationHiddenStateExtractor currently supports batch_size=1, "
                f"got {batch_size}."
            )

        records: List[Dict] = []
        generated_len = sequences.shape[1] - prompt_len

        if len(self._hidden_buffer) < generated_len:
            warnings.warn(
                f"Hidden buffer has {len(self._hidden_buffer)} entries but "
                f"{generated_len} tokens were generated. Some records will be missing."
            )

        for i in range(generated_len):
            if i >= len(self._hidden_buffer):
                break
            token_id = int(sequences[0, prompt_len + i].item())
            hidden_state = self._hidden_buffer[i][0, 0, :].tolist()  # (hidden_dim,)
            records.append({
                "step": i,
                "token_id": token_id,
                "token_position": prompt_len + i,
                "hidden_state": hidden_state,
            })
        return records

    def reset(self):
        """Clear internal buffers so the same extractor can be reused."""
        self._hidden_buffer.clear()
        self._step = 0
        self._prompt_len = 0


# ------------------------------------------------------------------
# Legacy helpers (kept for import stability)
# ------------------------------------------------------------------

def train_conflict_detector(
    detector: LatentConflictDetector,
    model: AutoModelForCausalLM,
    tokenizer,
    train_data: List[Tuple[str, str]],  # (prompt, label)
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """Train the conflict detector on synthetic data.

    .. deprecated::
        This legacy routine trains on **prompt-only** hidden states and is
        architecturally flawed for generation-time detection.  Use
        ``scripts/train_conflict_detector.py`` for the proper per-token
        generation-time training pipeline.
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

            last_hidden = states[-1][:, -1, :].to(device)  # (1, hidden_dim)
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
    detector = LatentConflictDetector(hidden_dim=768)
    print(
        f"LatentConflictDetector created with "
        f"{sum(p.numel() for p in detector.parameters())/1e3:.1f}K params"
    )
    print(f"Labels: {detector.LABELS}")
