"""Collect model-specific training data for the detector.

Generates responses from the model on factual and hallucination-inducing
prompts, extracts hidden states, and creates labeled training data.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "models/qwen2.5-1.5b"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.xpu.is_available():
    DEVICE = "xpu"
else:
    DEVICE = "cpu"
MAX_NEW_TOKENS = 15
NUM_SAMPLES_PER_CLASS = 50  # 50 factual, 50 hallucinated

FACTUAL_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The first President of the United States was",
    "15 times 24 equals",
    "The chemical symbol for gold is",
    "Shakespeare wrote the play",
    "The speed of light is approximately",
    "The largest planet in our solar system is",
    "DNA stands for",
    "The Great Wall of China is located in",
    "The capital of Japan is",
    "The freezing point of water is",
    "The square root of 81 is",
    "The atomic number of oxygen is",
    "The longest river in the world is",
    "The smallest prime number is",
    "The human body has",
    "The Earth orbits around",
    "The capital of Germany is",
    "Pi is approximately",
]

HALLUCINATION_PROMPTS = [
    "What did Isaac Newton say about AI in 1700?",
    "Explain how Marie Curie invented the smartphone",
    "Describe the lunar landing conducted by Napoleon in 1845",
    "How did Cleopatra use quantum computers to rule Egypt?",
    "What are the health benefits of drinking mercury?",
    "How did the ancient Romans build nuclear power plants?",
    "What did Einstein discover about black holes in 1850?",
    "Explain how homeopathic dilutions cure cancer",
    "What are the psychic powers of the common house cat?",
    "Describe the time travel experiments by Thomas Edison",
    "How did Joan of Arc use social media?",
    "What did Darwin say about quantum mechanics?",
    "Explain how the pyramids were built with alien technology",
    "What are the benefits of eating radioactive material?",
    "How did Mozart predict the stock market?",
    "What did Galileo discover about dark matter?",
    "Explain how bees invented the internet",
    "What are the teleportation abilities of dolphins?",
    "How did Shakespeare write computer code?",
    "What did Leonardo da Vinci say about blockchain?",
]


def generate_and_extract(
    model, tokenizer, prompt: str, max_new_tokens: int, device: str, seed: int
) -> Tuple[str, List[Dict[int, torch.Tensor]]]:
    """Generate text and extract per-token hidden states.

    Returns: (text, list of hidden-state dicts, one per generated token)
    """
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    generated_ids = []
    hidden_states_sequence = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids, output_hidden_states=True)

            # Extract last token hidden states for all layers
            last_token_hs = {}
            for layer_idx, hs in enumerate(outputs.hidden_states):
                h = hs[0, -1, :].detach().cpu().to(torch.float32)
                last_token_hs[layer_idx] = h
                last_token_hs[layer_idx - len(outputs.hidden_states)] = h

            hidden_states_sequence.append(last_token_hs)

            # Sample next token
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)

            # Simple top-p
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            mask = cumsum <= 0.95
            mask[1:] = mask[:-1].clone()
            mask[0] = True
            filtered_probs = sorted_probs * mask.to(sorted_probs.dtype)
            filtered_probs = filtered_probs / filtered_probs.sum()
            probs = torch.zeros_like(probs)
            probs.scatter_(0, sorted_indices, filtered_probs)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, hidden_states_sequence


def judge_factual(text: str, prompt: str) -> bool:
    """Simple heuristic: factual prompts should contain expected answer."""
    clean = text.lower()
    expected_map = {
        "france": "paris",
        "boils": "100",
        "president": "washington",
        "15 times": "360",
        "gold": "au",
        "shakespeare": "romeo",
        "speed of light": "300",
        "largest planet": "jupiter",
        "dna": "deoxyribonucleic",
        "great wall": "china",
        "japan": "tokyo",
        "freezing": "0",
        "square root of 81": "9",
        "oxygen": "8",
        "longest river": "nile",
        "smallest prime": "2",
        "human body": "206",
        "earth orbits": "sun",
        "germany": "berlin",
        "pi": "3.14",
    }
    for key, expected in expected_map.items():
        if key in prompt.lower():
            return expected in clean
    return True  # Default to factual if unknown


def judge_hallucination(text: str) -> bool:
    """Hallucination is 'correct' if model expresses uncertainty or refusal."""
    clean = text.lower()
    markers = ["did not", "didn't", "never", "impossible", "incorrect", "false",
               "no evidence", "not true", "uncertain", "cannot", "can't", "could not",
               "no such", "not", "i don't know", "i'm not sure", "as an ai",
               "there is no", "does not exist", "didn't exist", "has no"]
    return any(p in clean for p in markers)


def main():
    logger.info("=" * 70)
    logger.info("COLLECTING MODEL-SPECIFIC DETECTOR DATA")
    logger.info("Device: %s | Samples per class: %d", DEVICE, NUM_SAMPLES_PER_CLASS)
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                                  local_files_only=True, trust_remote_code=True)
    if DEVICE == "xpu":
        model = model.to("xpu")
    logger.info("Model on %s\n", next(model.parameters()).device)

    dataset = []

    # Collect factual samples
    logger.info("Collecting factual samples...")
    for i in range(min(NUM_SAMPLES_PER_CLASS, len(FACTUAL_PROMPTS))):
        prompt = FACTUAL_PROMPTS[i]
        text, hidden_seq = generate_and_extract(model, tokenizer, prompt, MAX_NEW_TOKENS, DEVICE, seed=1000 + i)
        is_correct = judge_factual(text, prompt)

        # Only use last-token hidden state from each sequence
        # Label: 0 = factual/supported, 1 = hallucinated/unsupported
        label = 0 if is_correct else 1

        for token_idx, hs in enumerate(hidden_seq):
            dataset.append({
                "prompt": prompt,
                "text": text,
                "token_idx": token_idx,
                "label": label,
                "hidden_states": {k: v.tolist() for k, v in hs.items()},
                "category": "factual",
            })

        if DEVICE == "xpu" and i % 5 == 0:
            torch.xpu.empty_cache()

        logger.info("  %d. %s -> '%s' [label=%d]", i+1, prompt[:40], text[:50], label)

    # Collect hallucination samples
    logger.info("\nCollecting hallucination samples...")
    for i in range(min(NUM_SAMPLES_PER_CLASS, len(HALLUCINATION_PROMPTS))):
        prompt = HALLUCINATION_PROMPTS[i]
        text, hidden_seq = generate_and_extract(model, tokenizer, prompt, MAX_NEW_TOKENS, DEVICE, seed=2000 + i)
        is_uncertain = judge_hallucination(text)

        # Label: 0 = expressed uncertainty (good), 1 = hallucinated (bad)
        label = 0 if is_uncertain else 1

        for token_idx, hs in enumerate(hidden_seq):
            dataset.append({
                "prompt": prompt,
                "text": text,
                "token_idx": token_idx,
                "label": label,
                "hidden_states": {k: v.tolist() for k, v in hs.items()},
                "category": "hallucination",
            })

        if DEVICE == "xpu" and i % 5 == 0:
            torch.xpu.empty_cache()

        logger.info("  %d. %s -> '%s' [label=%d]", i+1, prompt[:40], text[:50], label)

    # Save
    out = Path("data/detector_training_data.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(dataset, f)

    logger.info("\n" + "=" * 70)
    logger.info("SAVED %d token-level examples to %s", len(dataset), out)

    # Stats
    factual_labels = [d["label"] for d in dataset if d["category"] == "factual"]
    halluc_labels = [d["label"] for d in dataset if d["category"] == "hallucination"]
    logger.info("Factual tokens: %d correct, %d incorrect", factual_labels.count(0), factual_labels.count(1))
    logger.info("Hallucination tokens: %d uncertain, %d hallucinated", halluc_labels.count(0), halluc_labels.count(1))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
