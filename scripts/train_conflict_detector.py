"""Train Approach B: Latent-State Conflict Detector."""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, 'D:/ACC LLM Enhancement')
from src.acc_conflict_detector import LatentConflictDetector, HiddenStateExtractor

def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append((item['prompt'], item['label']))
    return data

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    model_path = 'D:/ACC LLM Enhancement/models/tiny_gpt2_safetensors'
    print(f"Loading model from {model_path}...")
    
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    model.eval()
    
    # Get hidden dimension from model
    hidden_dim = model.config.n_embd if hasattr(model.config, 'n_embd') else 768
    print(f"Hidden dimension: {hidden_dim}")
    
    # Create detector
    detector = LatentConflictDetector(hidden_dim=hidden_dim, num_layers=2, dropout=0.1)
    detector = detector.to(device)
    print(f"Detector parameters: {sum(p.numel() for p in detector.parameters())/1e3:.1f}K")
    
    # Load training data
    data_path = 'D:/ACC LLM Enhancement/data/acc_training/synthetic_conflict_data.jsonl'
    train_data = load_data(data_path)
    print(f"Training samples: {len(train_data)}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(detector.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Setup hidden state extractor
    extractor = HiddenStateExtractor(model, layer_idx=-4)
    extractor.register_hook()
    
    label_map = {label: idx for idx, label in enumerate(detector.LABELS)}
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for prompt, label_str in train_data:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                _ = model(**inputs)
            
            states = extractor.get_states()
            if not states:
                continue
            
            # Get last token's hidden state from the target layer
            last_hidden = states[0][:, -1, :].to(device)  # (1, hidden_dim)
            
            # Forward through detector
            logits = detector(last_hidden)
            
            # Target
            label = torch.tensor([label_map[label_str]], device=device)
            
            # Loss
            loss = criterion(logits, label)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=-1).item()
            correct += (pred == label_map[label_str])
            total += 1
        
        acc = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, acc={acc:.3f}")
    
    extractor.remove_hook()
    
    # Save model
    output_dir = 'D:/ACC LLM Enhancement/adapters/acc_conflict_detector'
    os.makedirs(output_dir, exist_ok=True)
    torch.save(detector.state_dict(), os.path.join(output_dir, 'detector.pt'))
    
    # Save config
    config = {
        'hidden_dim': hidden_dim,
        'num_layers': 2,
        'dropout': 0.1,
        'num_epochs': num_epochs,
        'final_accuracy': acc,
        'final_loss': avg_loss,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final accuracy: {acc:.3f}")
    print(f"Model saved to: {output_dir}/detector.pt")
    print(f"{'='*60}")
    
    # Test inference
    print("\nTest inference:")
    test_prompts = [
        "The capital of France is Paris.",
        "Humans only use 10% of their brain.",
        "What will the stock market do tomorrow?",
        "A square circle has equal sides.",
    ]
    
    detector.eval()
    extractor.register_hook()
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
            logits = model(**inputs).logits  # Need proper forward for this
            
        # Actually we need to do a proper forward with hidden states
        # Let's use the model's output_hidden_states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-4]  # (batch, seq, hidden)
            last_hidden = hidden_states[:, -1, :]  # (batch, hidden)
            
            logits = detector(last_hidden)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        
        pred_label = detector.LABELS[probs.argmax()]
        print(f"  '{prompt[:50]}...' -> {pred_label} (conf={probs.max():.3f})")
    
    extractor.remove_hook()

if __name__ == "__main__":
    main()
