"""GPU validation of ACC Approach A (Entropy Monitor) with tiny_gpt2."""

import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, 'D:/ACC LLM Enhancement')
from src.acc_integration import ACCEnhancedGenerator

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model_path = 'D:/ACC LLM Enhancement/models/tiny_gpt2_safetensors'
    print(f"\nLoading model from {model_path}...")
    
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    print(f"Model on {next(model.parameters()).device}")
    
    # Test prompts: mix of easy and hard
    prompts = [
        ("easy", "The sky is"),
        ("easy", "Water is H"),
        ("uncertain", "The quantum entanglement of consciousness is"),
        ("uncertain", "In the year 2045, medical AI will"),
        ("nonsense", "Zxyqpw mnq klrst uvw"),
    ]
    
    print("\n" + "="*70)
    print("ACC GPU VALIDATION")
    print("="*70)
    
    # Test with different ACC settings
    configs = [
        {"threshold": 3.5, "action": "flag", "mode": "absolute"},
        {"threshold": 3.0, "action": "regenerate", "mode": "absolute"},
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n--- Config: threshold={cfg['threshold']}, action={cfg['action']}, mode={cfg['mode']} ---")
        
        gen = ACCEnhancedGenerator(
            model=model,
            tokenizer=tokenizer,
            threshold=cfg["threshold"],
            action=cfg["action"],
            mode=cfg["mode"],
        )
        
        for category, prompt in prompts:
            out = gen.generate_from_prompt(
                prompt=prompt,
                max_new_tokens=10,
                temperature=0.7,
                return_dict_in_generate=True,
            )
            
            # Extract entropy values
            all_entropy = [h for row in out.per_token_entropy for h in row]
            mean_h = sum(all_entropy) / len(all_entropy) if all_entropy else 0.0
            max_h = max(all_entropy) if all_entropy else 0.0
            total_breaches = sum(len(row) for row in out.uncertain_steps)
            
            result = {
                "config": cfg,
                "category": category,
                "prompt": prompt,
                "generated_text": out.text[0] if out.text else "",
                "mean_entropy": mean_h,
                "max_entropy": max_h,
                "threshold_breaches": total_breaches,
                "confidence_score": out.confidence_score[0] if out.confidence_score else 0.0,
            }
            results.append(result)
            
            print(f"\n  [{category}] Prompt: '{prompt}'")
            print(f"    Output: {out.text[0][:60]}...")
            print(f"    Mean entropy: {mean_h:.2f} | Max: {max_h:.2f} | Breaches: {total_breaches}")
            
            if category == "easy" and total_breaches == 0:
                print(f"    ✓ No uncertainty on easy prompt (correct)")
            elif category in ["uncertain", "nonsense"] and total_breaches > 0:
                print(f"    ✓ Detected uncertainty on hard prompt (correct)")
    
    # Save results
    output_path = 'D:/ACC LLM Enhancement/results/acc_gpu_validation.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"Total tests: {len(results)}")
    print(f"Device used: {device}")
    if device == 'cuda':
        print(f"GPU memory peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
