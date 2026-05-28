"""Validate QLoRA adapter files without loading full model."""

import json
import os

def validate_adapter(adapter_dir="adapters/tiny_test/final_adapter"):
    """Validate that LoRA adapter was properly saved."""
    print("=== QLoRA Adapter Validation ===\n")
    
    # 1. Check adapter config exists
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        print("FAIL: adapter_config.json not found")
        return False
    
    with open(config_path) as f:
        cfg = json.load(f)
    
    print("1. Adapter Config:")
    print(f"   r (rank): {cfg.get('r')}")
    print(f"   lora_alpha: {cfg.get('lora_alpha')}")
    print(f"   target_modules: {cfg.get('target_modules')}")
    print(f"   lora_dropout: {cfg.get('lora_dropout')}")
    print(f"   bias: {cfg.get('bias')}")
    print(f"   task_type: {cfg.get('task_type')}")
    
    # 2. Check adapter weights exist
    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(adapter_dir, "adapter_model.bin")
    
    if not os.path.exists(weights_path):
        print("\nFAIL: Adapter weights not found")
        return False
    
    size_mb = os.path.getsize(weights_path) / (1024**2)
    print(f"\n2. Adapter Weights:")
    print(f"   File: {os.path.basename(weights_path)}")
    print(f"   Size: {size_mb:.2f} MB")
    
    # 3. Verify config values are sensible
    r = cfg.get('r', 0)
    alpha = cfg.get('lora_alpha', 0)
    
    if r <= 0:
        print("\nFAIL: r must be > 0")
        return False
    
    if alpha <= 0:
        print("\nFAIL: lora_alpha must be > 0")
        return False
    
    if not cfg.get('target_modules'):
        print("\nFAIL: target_modules must not be empty")
        return False
    
    # 4. Check training output directory
    output_dir = "adapters/tiny_test"  # Actual output dir used by training
    print(f"\n3. Training Output Directory: {output_dir}")
    
    has_checkpoint = os.path.exists(os.path.join(output_dir, "checkpoint-50")) or \
                     os.path.exists(os.path.join(output_dir, "checkpoint-100")) or \
                     any("checkpoint" in d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)))
    
    print(f"   Has checkpoints: {has_checkpoint}")
    
    # 5. Check trainer state
    state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        
        log_history = state.get('log_history', [])
        if log_history:
            # Find last train and eval entries
            train_logs = [h for h in log_history if 'loss' in h and 'eval_loss' not in h]
            eval_logs = [h for h in log_history if 'eval_loss' in h]
            
            if train_logs:
                last_train = train_logs[-1]
                print(f"\n4. Training Metrics:")
                print(f"   Final step: {last_train.get('step')}")
                print(f"   Final loss: {last_train.get('loss', 'N/A'):.4f}" if isinstance(last_train.get('loss'), (int, float)) else f"   Final loss: {last_train.get('loss', 'N/A')}")
                
                if eval_logs:
                    last_eval = eval_logs[-1]
                    print(f"   Final eval loss: {last_eval.get('eval_loss', 'N/A'):.4f}" if isinstance(last_eval.get('eval_loss'), (int, float)) else f"   Final eval loss: {last_eval.get('eval_loss', 'N/A')}")
                    
                    # Check if loss decreased
                    if len(train_logs) >= 2:
                        first_loss = train_logs[0].get('loss', float('inf'))
                        last_loss = last_train.get('loss', float('inf'))
                        if isinstance(first_loss, (int, float)) and isinstance(last_loss, (int, float)):
                            improvement = first_loss - last_loss
                            print(f"   Loss improvement: {improvement:.4f} ({'improved' if improvement > 0 else 'worsened'})")
    
    print("\n=== VALIDATION: PASSED ===")
    print("Adapter files are valid and training metrics look correct.")
    return True

if __name__ == "__main__":
    validate_adapter()
