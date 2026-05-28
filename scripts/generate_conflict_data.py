"""Generate per-token training data for the generation-time conflict detector.

Instead of hand-writing 40 labeled sentences, this script:
  1. Loads a small causal LM (configurable).
  2. Generates continuations for four categories of prompts.
  3. Labels *each generated token* using heuristics and (optionally)
     sentence-embedding similarity.
  4. Writes a JSONL file where every line is one token record.

Expected output: ≥ 500 tokens per class (≥ 2 000 total).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow importing src/ modules regardless of where the script is invoked
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.acc_conflict_detector import GenerationHiddenStateExtractor

# ---------------------------------------------------------------------------
# Prompt banks
# ---------------------------------------------------------------------------

SupportedPrompt = Tuple[str, str]  # (prompt, expected_answer)

SUPPORTED_PROMPTS: List[SupportedPrompt] = [
    ("The capital of France is", "Paris"),
    ("Water freezes at", "0 degrees celsius"),
    ("The Earth orbits the", "Sun"),
    ("Humans have", "23 pairs of chromosomes"),
    ("Oxygen is essential for", "human respiration"),
    ("The speed of light is approximately", "300,000 km/s"),
    ("DNA stands for", "deoxyribonucleic acid"),
    ("The heart pumps blood through the", "circulatory system"),
    ("Newton's first law describes", "inertia"),
    ("Photosynthesis converts CO2 and water into", "glucose"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("Shakespeare wrote", "Romeo and Juliet"),
    ("The chemical symbol for gold is", "Au"),
    ("The capital of Japan is", "Tokyo"),
    ("A triangle has", "three sides"),
    ("The freezing point of water is", "0"),
    ("The square root of 64 is", "8"),
    ("The first president of the United States was", "George Washington"),
    ("The Pacific Ocean is the", "largest ocean"),
    ("Mount Everest is located in", "Nepal"),
    ("The human skeleton has", "206 bones"),
    ("Python is a popular", "programming language"),
    ("The Mona Lisa was painted by", "Leonardo da Vinci"),
    ("The Great Barrier Reef is in", "Australia"),
    ("Hydrogen is the", "lightest element"),
]

HALLUCINATED_PROMPTS: List[SupportedPrompt] = [
    ("The Great Wall of China is visible from", "the moon"),
    ("Humans only use", "10% of their brain"),
    ("Goldfish have a", "3-second memory"),
    ("Bats are completely", "blind"),
    ("The Sahara is the largest", "desert in the world"),
    ("Shaving makes hair grow back", "thicker"),
    ("Napoleon was extremely", "short"),
    ("Lightning never strikes the same place", "twice"),
    ("Dropping a penny from a skyscraper can", "kill someone"),
    ("The full moon causes increased", "crime rates"),
    ("Cracking your knuckles causes", "arthritis"),
    ("Carrots improve your", "night vision"),
    ("The tongue has different zones for", "taste"),
    ("Vikings wore helmets with", "horns"),
    ("Chameleons change color to match their", "surroundings"),
    ("Humans have", "five senses"),
    ("The Declaration of Independence was signed in", "1776"),
    ("Einstein failed", "math"),
    ("Ostriches bury their heads in the", "sand"),
    ("Sushi is made from", "raw fish only"),
]

UNCERTAIN_PROMPTS: List[str] = [
    "What will the stock market do tomorrow?",
    "Who will win the next World Cup?",
    "What is the exact population of Earth right now?",
    "Will AI become sentient by 2030?",
    "What is the cure for aging?",
    "Are there aliens in the Andromeda galaxy?",
    "What will technology look like in 100 years?",
    "Is there life after death?",
    "What caused the dinosaurs to actually go extinct?",
    "Will humans colonize Mars by 2050?",
    "What will be the next big social media platform?",
    "Who will be elected president in 2032?",
    "Will quantum computers break all encryption?",
    "What is the meaning of life?",
    "Will teleportation ever be possible?",
    "What will the climate be like in 2100?",
    "Is time travel possible?",
    "What discoveries will neuroscience make next?",
    "Will we achieve universal basic income?",
    "What is the future of education?",
    "Will fusion power be commercially viable?",
    "What will happen to the Arctic ice caps?",
    "Will we discover a new fundamental force?",
    "What is the best diet for longevity?",
    "Will virtual reality replace physical travel?",
]

CONTRADICTORY_PROMPTS: List[str] = [
    "A square circle has equal sides and no corners, which means",
    "The transparent opaque wall lets light through, so",
    "A married bachelor lives with his wife, therefore",
    "The boiling ice was very hot and cold, so",
    "A silent scream echoed through the room, which proves",
    "The invisible ghost was clearly seen by everyone, meaning",
    "The dry water soaked everything, therefore",
    "A burning fire that produces no heat is useful because",
    "The dead living organism grew rapidly, which shows",
    "A stationary moving car drove nowhere, proving",
    "A round square has four equal sides and no corners, so",
    "The deafening silence filled the room, meaning",
    "A solid liquid flowed like a rock, therefore",
    "The empty box was completely full of nothing, which means",
    "A feather made of lead floated gently, proving",
    "The bright darkness illuminated the cave, so",
    "A vegetarian carnivore eats only meat, which means",
    "The frozen lava was extremely cold and hot, therefore",
    "A wooden iron sword is sharp because",
    "The living dead walked slowly, proving",
    "An honest liar always tells false truths, meaning",
    "The wet desert was flooded with sand, so",
    "A straight curve bends in a line, therefore",
    "The soft diamond scratched easily, which shows",
    "A nocturnal daylight creature sleeps at night, proving",
]

# ---------------------------------------------------------------------------
# Optional sentence-transformers
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except Exception:  # pragma: no cover
    _ST_AVAILABLE = False
    SentenceTransformer = None  # type: ignore[misc,assignment]


def _load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if not _ST_AVAILABLE:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        warnings.warn(f"Could not load embedder '{model_name}': {exc}")
        return None


def _embedding_similarity(embedder, text_a: str, text_b: str) -> float:
    if embedder is None:
        return 0.0
    embs = embedder.encode([text_a, text_b], convert_to_tensor=True)
    sim = F.cosine_similarity(embs[0].unsqueeze(0), embs[1].unsqueeze(0), dim=-1)
    return float(sim.item())


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    extractor: GenerationHiddenStateExtractor,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> Tuple[List[Dict], torch.LongTensor, List[float]]:
    """Generate a continuation and return per-token records + scores.

    Returns:
        records: list of dicts with keys ``step``, ``token_id``, ``token_position``,
            ``hidden_state``.
        sequences: (1, total_seq_len) tensor.
        token_probs: list of max-probabilities for each generated token.
    """
    extractor.reset()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        logits_processor=[extractor],
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    sequences = outputs.sequences  # (1, total_seq_len)
    records = extractor.get_records(sequences, prompt_len=prompt_len)

    # Compute per-token max probabilities from score tensors
    token_probs: List[float] = []
    if outputs.scores:
        for score in outputs.scores:
            probs = F.softmax(score, dim=-1)
            token_probs.append(float(probs.max().item()))

    # Attach decoded token text to each record
    for rec in records:
        rec["token_text"] = tokenizer.decode([rec["token_id"]])

    return records, sequences, token_probs


# ---------------------------------------------------------------------------
# Heuristic labelling
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.lower().strip(" .,!?;:\n\t")


def label_supported(
    prompt: str,
    expected: str,
    generated_text: str,
    token_probs: List[float],
    embedder,
) -> Optional[str]:
    """Return 'supported' if the generation matches the expected fact."""
    gen_norm = _normalize(generated_text)
    exp_norm = _normalize(expected)

    # Primary heuristic: substring match
    if exp_norm in gen_norm:
        return "supported"

    # Secondary: embedding similarity against expected answer
    if embedder is not None:
        sim = _embedding_similarity(embedder, generated_text, expected)
        if sim > 0.65:
            return "supported"

    # Tertiary: if the generation is very high-confidence and coherent, accept
    if token_probs and sum(token_probs) / len(token_probs) > 0.85:
        return "supported"

    return None  # discard this sample


def label_hallucinated(
    prompt: str,
    expected_falsehood: str,
    generated_text: str,
    token_probs: List[float],
    embedder,
) -> Optional[str]:
    """Return 'hallucinated' if the generation contains the expected falsehood."""
    gen_norm = _normalize(generated_text)
    false_norm = _normalize(expected_falsehood)

    # If the falsehood appears, definitely hallucinated
    if false_norm in gen_norm:
        return "hallucinated"

    # If generation contradicts known facts via low confidence or incoherence
    if token_probs:
        avg_prob = sum(token_probs) / len(token_probs)
        if avg_prob < 0.5:
            return "hallucinated"

    # Embedding similarity to the falsehood
    if embedder is not None:
        sim = _embedding_similarity(embedder, generated_text, expected_falsehood)
        if sim > 0.60:
            return "hallucinated"

    return None  # discard


def label_uncertain(
    _prompt: str,
    _generated_text: str,
    _token_probs: List[float],
    _embedder,
) -> Optional[str]:
    """Uncertain prompts are always labeled uncertain."""
    return "uncertain"


def label_contradictory(
    _prompt: str,
    generated_text: str,
    token_probs: List[float],
    _embedder,
) -> Optional[str]:
    """Label contradictory if the generation does not resolve the contradiction."""
    # For contradictory prompts, almost any continuation that doesn't explicitly
    # point out the contradiction counts as contradictory for our purposes.
    resolve_keywords = ["impossible", "contradiction", "cannot", "nonsense", "error"]
    gen_norm = _normalize(generated_text)
    if any(kw in gen_norm for kw in resolve_keywords):
        # The model caught the contradiction — skip this sample
        return None
    return "contradictory"


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-token conflict-detector training data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="sshleifer/tiny-gpt2",
        help="HF model id or local path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "acc_training" / "generated_conflict_data.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--tokens_per_class",
        type=int,
        default=500,
        help="Minimum tokens to generate for each label",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=18,
        help="Max new tokens per generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--use_embedder",
        action="store_true",
        help="Use sentence-transformers for semantic similarity checks",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ------------------------------------------------------------------
    # Load model & tokenizer
    # ------------------------------------------------------------------
    print(f"Loading model '{args.model_path}' on {args.device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model = model.to(args.device)
    model.eval()

    hidden_dim = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd
    print(f"Model hidden dim: {hidden_dim}")

    # ------------------------------------------------------------------
    # Optional embedder
    # ------------------------------------------------------------------
    embedder = _load_embedder() if args.use_embedder else None
    if embedder:
        print("Sentence-transformer embedder loaded.")
    else:
        print("Running without embedder (heuristic mode only).")

    # ------------------------------------------------------------------
    # Setup extractor
    # ------------------------------------------------------------------
    num_layers = len(model.transformer.h) if hasattr(model, 'transformer') and hasattr(model.transformer, 'h') else len(model.model.layers)
    layer_idx = -4 if num_layers >= 4 else -2
    print(f"Using layer_idx={layer_idx} (model has {num_layers} layers)")
    extractor = GenerationHiddenStateExtractor(model, layer_idx=layer_idx)

    # ------------------------------------------------------------------
    # Generation loops per category
    # ------------------------------------------------------------------
    counts: Dict[str, int] = {label: 0 for label in ["supported", "hallucinated", "uncertain", "contradictory"]}
    written = 0

    def write_records(records: List[Dict], label: str, prompt: str, fout):
        nonlocal written
        for rec in records:
            rec["label"] = label
            rec["prompt"] = prompt
            fout.write(json.dumps(rec) + "\n")
            written += 1
        counts[label] += len(records)

    with open(args.output, "w", encoding="utf-8") as fout:
        # ---------------- Supported ----------------
        print("\n--- Generating: supported ---")
        random.shuffle(SUPPORTED_PROMPTS)
        for prompt, expected in SUPPORTED_PROMPTS:
            if counts["supported"] >= args.tokens_per_class:
                break
            records, sequences, token_probs = generate_tokens(
                model, tokenizer, extractor, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            if not records:
                continue
            gen_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
            label = label_supported(prompt, expected, gen_text, token_probs, embedder)
            if label:
                write_records(records, label, prompt, fout)
                print(f"  [{counts['supported']:4d}] {prompt[:50]:50s} -> {gen_text[:60]}")

        # ---------------- Hallucinated ----------------
        print("\n--- Generating: hallucinated ---")
        random.shuffle(HALLUCINATED_PROMPTS)
        for prompt, expected_falsehood in HALLUCINATED_PROMPTS:
            if counts["hallucinated"] >= args.tokens_per_class:
                break
            # Slightly higher temperature encourages falsehoods
            records, sequences, token_probs = generate_tokens(
                model, tokenizer, extractor, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=min(args.temperature * 1.1, 1.5),
            )
            if not records:
                continue
            gen_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
            label = label_hallucinated(prompt, expected_falsehood, gen_text, token_probs, embedder)
            if label:
                write_records(records, label, prompt, fout)
                print(f"  [{counts['hallucinated']:4d}] {prompt[:50]:50s} -> {gen_text[:60]}")

        # ---------------- Uncertain ----------------
        print("\n--- Generating: uncertain ---")
        random.shuffle(UNCERTAIN_PROMPTS)
        for prompt in UNCERTAIN_PROMPTS:
            if counts["uncertain"] >= args.tokens_per_class:
                break
            records, sequences, _token_probs = generate_tokens(
                model, tokenizer, extractor, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            if not records:
                continue
            gen_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
            label = label_uncertain(prompt, gen_text, _token_probs, embedder)
            if label:
                write_records(records, label, prompt, fout)
                print(f"  [{counts['uncertain']:4d}] {prompt[:50]:50s} -> {gen_text[:60]}")

        # ---------------- Contradictory ----------------
        print("\n--- Generating: contradictory ---")
        random.shuffle(CONTRADICTORY_PROMPTS)
        for prompt in CONTRADICTORY_PROMPTS:
            if counts["contradictory"] >= args.tokens_per_class:
                break
            records, sequences, token_probs = generate_tokens(
                model, tokenizer, extractor, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            if not records:
                continue
            gen_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
            label = label_contradictory(prompt, gen_text, token_probs, embedder)
            if label:
                write_records(records, label, prompt, fout)
                print(f"  [{counts['contradictory']:4d}] {prompt[:50]:50s} -> {gen_text[:60]}")

    extractor.remove_hook()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Finished writing {written} token records to:")
    print(f"  {args.output}")
    print("\nLabel distribution:")
    for label, c in counts.items():
        print(f"  {label:15s}: {c:4d} tokens")
    print(f"{'='*60}")

    # Sanity check
    for label, c in counts.items():
        if c < args.tokens_per_class:
            print(
                f"WARNING: only generated {c}/{args.tokens_per_class} tokens "
                f"for class '{label}'. Increase prompt bank or adjust heuristics."
            )


if __name__ == "__main__":
    main()
