import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import Optional, Dict, List, Tuple
import json

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
            hidden_states: (batch, seq_len, hidden_dim) from LLM intermediate layer
        Returns:
            logits: (batch, seq_len, num_labels)
        """
        return self.mlp(hidden_states)
    
    def classify(self, hidden_states: torch.Tensor, token_position: int = -1) -> Dict[str, float]:
        """Classify a specific token position.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            token_position: Which token to classify (-1 = last token)
        Returns:
            Dict of label -> probability
        """
        with torch.no_grad():
            logits = self.forward(hidden_states)
            probs = F.softmax(logits[:, token_position], dim=-1)
            probs = probs.squeeze(0).cpu().numpy()
        
        return {label: float(prob) for label, prob in zip(self.LABELS, probs)}
    
    def get_conflict_score(self, hidden_states: torch.Tensor, token_position: int = -1) -> float:
        """Return a scalar conflict score (0=supported, 1=hallucinated/uncertain)."""
        probs = self.classify(hidden_states, token_position)
        # Conflict = hallucinated + uncertain + contradictory
        return probs["hallucinated"] + probs["uncertain"] + probs["contradictory"]


class HiddenStateExtractor:
    """Extracts hidden states from a LLM during forward pass."""
    
    def __init__(self, model: AutoModelForCausalLM, layer_idx: int = -4):
        """
        Args:
            model: The causal LM
            layer_idx: Which transformer layer to extract from (-1 = last, -4 = near output)
        """
        self.model = model
        self.layer_idx = layer_idx
        self.hidden_states = []
        self._hook = None
        
    def register_hook(self):
        """Register forward hook to capture hidden states."""
        def hook_fn(module, input, output):
            # output is typically (batch, seq_len, hidden_dim)
            if isinstance(output, tuple):
                output = output[0]
            self.hidden_states.append(output.detach().cpu())
        
        # Get the target layer
        if hasattr(self.model, 'transformer'):
            # GPT-2 style
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Mistral/Llama style
            layers = self.model.model.layers
        else:
            raise ValueError("Unsupported model architecture")
        
        target_layer = layers[self.layer_idx]
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
    
    Args:
        detector: The LatentConflictDetector to train
        model: Base LLM (frozen)
        tokenizer: Tokenizer
        train_data: List of (prompt, label) pairs
        num_epochs: Training epochs
        lr: Learning rate
        device: "cuda" or "cpu"
    """
    model.eval()
    detector = detector.to(device)
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    extractor = HiddenStateExtractor(model, layer_idx=-4)
    extractor.register_hook()
    
    label_map = {label: idx for idx, label in enumerate(detector.LABELS)}
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for prompt, label_str in train_data:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            states = extractor.get_states()
            if not states:
                continue
            
            # Use last token's hidden state
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
    # Demo/test
    detector = LatentConflictDetector(hidden_dim=768)
    print(f"LatentConflictDetector created with {sum(p.numel() for p in detector.parameters())/1e3:.1f}K params")
    print(f"Labels: {detector.LABELS}")
