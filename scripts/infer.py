"""Inference script — load base model + LoRA adapter and run generation.

Supports an ACC mode (--use-acc) that wraps the model with
ACCEnhancedGenerator to monitor per-token entropy and react on breaches.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running as `python scripts/infer.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acc_integration import ACCEnhancedGenerator  # noqa: E402

logger = logging.getLogger(__name__)


def load_adapter(base_path: str, adapter_path: str, device: str = "auto"):
    """Load base model + LoRA adapter for inference."""
    logger.info("Loading base model from %s", base_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map=device,
        local_files_only=True,
    )
    logger.info("Loading adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()  # optional: merge for faster inference
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_with_acc(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    threshold: float,
    action: str,
    mode: str,
    temperature_adjustment: float,
):
    gen = ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        threshold=threshold,
        action=action,
        mode=mode,
        regen_temperature_multiplier=1.0 + temperature_adjustment,
    )
    out = gen.generate_from_prompt(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        return_dict_in_generate=True,
    )
    # Compute derived stats from dataclass fields
    tokens_generated = out.sequences.shape[1]
    all_entropy = [h for row in out.per_token_entropy for h in row]
    mean_h = sum(all_entropy) / len(all_entropy) if all_entropy else 0.0
    max_h = max(all_entropy) if all_entropy else 0.0
    total_breaches = sum(len(row) for row in out.uncertain_steps)
    return {
        "text": out.text[0] if out.text else "",
        "tokens_generated": tokens_generated,
        "mean_entropy": mean_h,
        "max_entropy": max_h,
        "confidence_score": out.confidence_score[0] if out.confidence_score else 0.0,
        "threshold_breaches": total_breaches,
        "regenerations": sum(out.regenerations) if out.regenerations else 0,
        "warnings": sum(1 for e in out.events for ev in e if ev.get("action") == "warning"),
        "action": action,
        "mode": mode,
        "threshold": threshold,
    }


def _print_acc_report(result: dict) -> None:
    print("\n" + "=" * 50)
    print(result["text"])
    print("=" * 50)
    print("ACC entropy statistics")
    print(f"  tokens generated:      {result['tokens_generated']}")
    print(f"  mean entropy (nats):   {result['mean_entropy']:.4f}")
    print(f"  max entropy (nats):    {result['max_entropy']:.4f}")
    print(f"  confidence score:      {result['confidence_score']:.4f}")
    print(
        f"  threshold breaches:    {result['threshold_breaches']} "
        f"(action={result['action']}, mode={result['mode']}, thr={result['threshold']})"
    )
    print(f"  regenerations:         {result['regenerations']}")
    print(f"  warning steps:         {result['warnings']}")
    print("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="models/mistral_7b")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter folder")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--use-acc",
        action="store_true",
        help="Wrap the model with ACCEnhancedGenerator and report entropy stats",
    )
    parser.add_argument(
        "--acc-threshold",
        type=float,
        default=3.5,
        help="Entropy threshold (interpretation depends on --acc-mode)",
    )
    parser.add_argument(
        "--acc-action",
        choices=["flag", "regenerate", "warning"],
        default="flag",
        help="What to do when entropy crosses the threshold",
    )
    parser.add_argument(
        "--acc-mode",
        choices=["absolute", "moving_average", "percentile"],
        default="absolute",
        help="Threshold strategy for the entropy monitor",
    )
    parser.add_argument(
        "--acc-temp-adjustment",
        type=float,
        default=0.3,
        help="Temperature reduction applied on 'regenerate' action",
    )
    args = parser.parse_args()

    model, tokenizer = load_adapter(args.base_model, args.adapter, args.device)

    if args.use_acc:
        result = generate_with_acc(
            model,
            tokenizer,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.acc_threshold,
            args.acc_action,
            args.acc_mode,
            args.acc_temp_adjustment,
        )
        _print_acc_report(result)
    else:
        result = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature)
        print("\n" + "=" * 50)
        print(result)
        print("=" * 50)
