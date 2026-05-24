"""Inference script — load base model + LoRA adapter and run generation."""

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="models/mistral_7b")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter folder")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, tokenizer = load_adapter(args.base_model, args.adapter, args.device)
    result = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature)
    print("\n" + "=" * 50)
    print(result)
    print("=" * 50)
