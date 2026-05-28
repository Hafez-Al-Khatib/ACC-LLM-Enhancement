"""Create a tiny local model for QLoRA pipeline testing.
No downloads needed - builds a small GPT-like model from scratch.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
import os
import json

class TinyConfig(PretrainedConfig):
    model_type = "gpt2"
    
    def __init__(self, vocab_size=1000, n_positions=512, n_embd=128, n_layer=4, n_head=4, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_embd * 4
        self.activation_function = "gelu"
        self.resid_pdrop = 0.1
        self.embd_pdrop = 0.1
        self.attn_pdrop = 0.1
        self.layer_norm_epsilon = 1e-5
        self.initializer_range = 0.02
        self.use_cache = True
        self.bos_token_id = 0
        self.eos_token_id = 0

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with explicit Q/K/V Linear layers for PEFT LoRA."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Separate Q/K/V projections - PEFT can target these
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self, x):
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool().view(1, 1, T, T), float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        y = self.resid_dropout(y)
        return y

class TinyBlock(nn.Module):
    """Transformer block with causal self-attention + MLP."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = nn.ModuleDict(dict(
            c_fc = nn.Linear(config.n_embd, config.n_inner),
            c_proj = nn.Linear(config.n_inner, config.n_embd),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        self.act = nn.GELU()
    
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.ln_1(x))
        # MLP with residual
        m = self.mlp
        x = x + m.dropout(m.c_proj(self.act(m.c_fc(self.ln_2(x)))))
        return x

class TinyGPT2(PreTrainedModel):
    config_class = TinyConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        self.h = nn.ModuleList([TinyBlock(config) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.post_init()
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        b, t = input_ids.size()
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.h:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return type('Output', (), {'logits': logits, 'loss': None})()

def create_tiny_model(output_dir="models/tiny_gpt"):
    """Create and save a tiny model for testing QLoRA pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    config = TinyConfig(vocab_size=1000, n_positions=512, n_embd=128, n_layer=4, n_head=4)
    model = TinyGPT2(config)
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # Create tokenizer
    from transformers import GPT2Tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    except:
        # Create a minimal tokenizer if download fails
        tokenizer = GPT2Tokenizer(
            vocab_file=os.path.join(output_dir, "vocab.json"),
            merges_file=os.path.join(output_dir, "merges.txt"),
        )
    tokenizer.save_pretrained(output_dir)
    
    print(f"Tiny model created at {output_dir}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Model size: ~{os.path.getsize(os.path.join(output_dir, 'pytorch_model.bin')) / (1024**2):.1f} MB")
    
    return model, tokenizer

if __name__ == "__main__":
    create_tiny_model()
    print("\nTo test QLoRA on this model:")
    print("  python scripts/train.py --config configs/tiny_test.yaml")
