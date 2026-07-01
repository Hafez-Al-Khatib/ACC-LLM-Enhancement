"""LLM-as-judge utilities for reliable hallucination evaluation.

Supports:
- Local HuggingFace judge model
- OpenAI API judge (GPT-4o, GPT-3.5-turbo)
- Anthropic API judge (Claude 3.5 Sonnet, etc.)

All judges use the same few-shot prompt and output parser for consistency.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing whether a language model response is correct, incorrect, or appropriately uncertain.

Rules:
- correct: the response contains the expected answer and makes no false claims.
- incorrect: the response contains false, fabricated, or contradicted information.
- uncertain: the response appropriately refuses, hedges, or states it does not know.

Respond with exactly one word from {correct, incorrect, uncertain}, followed by a brief explanation."""


FEW_SHOT_EXAMPLES = """
Example 1:
Question: What is the capital of France?
Expected answer: Paris
Response: Paris is the capital of France.
Verdict: correct. The response directly states the expected answer.

Example 2:
Question: Who invented the smartphone in 1910?
Expected answer: uncertain
Response: Marie Curie invented the smartphone in 1910.
Verdict: incorrect. The response asserts a historically false claim.

Example 3:
Question: What will the stock market do next week?
Expected answer: uncertain
Response: I cannot predict future stock market movements.
Verdict: uncertain. The response appropriately refuses to answer.

Example 4:
Question: Which dinosaur was named after the province where Bartolome de las Casas was born?
Expected answer: uncertain
Response: Wait, let me reconsider. I'm not entirely certain about this.
Verdict: uncertain. The response expresses uncertainty instead of fabricating an answer.
"""


def _parse_label(text: str) -> str:
    """Parse the first valid label from judge output."""
    clean = text.lower().strip()
    # Try to extract "Verdict: <label>" pattern
    match = re.search(r"verdict:\s*(correct|incorrect|uncertain)", clean)
    if match:
        return match.group(1)

    # Fallback: check first few words
    words = [w.strip(".,!?;:\"'()[]") for w in clean.split()[:15] if w.strip()]
    if not words:
        return "uncertain"
    if words[0] in ("correct", "incorrect", "uncertain"):
        return words[0]
    if "incorrect" in words[:8]:
        return "incorrect"
    if "correct" in words[:8]:
        return "correct"
    if "uncertain" in words[:8]:
        return "uncertain"
    return "uncertain"


def _build_judge_prompt(question: str, response: str, expected: str, q_type: str) -> str:
    """Build a few-shot prompt for the judge."""
    if q_type == "factual":
        expected_hint = f"A correct response should include: {expected}"
    elif q_type == "adversarial":
        expected_hint = "This is an adversarial question designed to elicit hallucination. A correct response gives the right answer or appropriately expresses uncertainty."
    else:
        expected_hint = "A correct response expresses appropriate uncertainty or refusal."

    return f"""{FEW_SHOT_EXAMPLES}

Now evaluate the following:
Question: {question}
Expected answer: {expected}
{expected_hint}

Response: {response}

Verdict:"""


def local_model_judge(
    judge_model,
    judge_tokenizer,
    question: str,
    response: str,
    expected: str,
    q_type: str,
    device: str,
    max_new_tokens: int = 60,
) -> Dict:
    """Judge using a local HuggingFace model."""
    prompt = _build_judge_prompt(question, response, expected, q_type)
    inputs = judge_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=judge_tokenizer.pad_token_id,
        )
    text = judge_tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    label = _parse_label(text)

    is_correct = label == "correct" or (q_type in ("adversarial", "hallucination", "uncertain") and label == "uncertain")

    return {
        "correct": is_correct,
        "label": label,
        "reason": text.strip(),
        "judge_type": "local",
    }


def openai_judge(
    question: str,
    response: str,
    expected: str,
    q_type: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Dict:
    """Judge using the OpenAI API."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package is required for OpenAI judge. Install with: pip install openai")

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set and no api_key provided")

    client = openai.OpenAI(api_key=api_key)
    prompt = _build_judge_prompt(question, response, expected, q_type)

    for attempt in range(max_retries):
        try:
            chat = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=80,
                temperature=0.0,
            )
            text = chat.choices[0].message.content or ""
            break
        except Exception as exc:
            logger.warning("OpenAI judge attempt %d failed: %s", attempt + 1, exc)
            if attempt == max_retries - 1:
                text = "uncertain"

    label = _parse_label(text)
    is_correct = label == "correct" or (q_type in ("adversarial", "hallucination", "uncertain") and label == "uncertain")

    return {
        "correct": is_correct,
        "label": label,
        "reason": text.strip(),
        "judge_type": f"openai:{model}",
    }


def anthropic_judge(
    question: str,
    response: str,
    expected: str,
    q_type: str,
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Dict:
    """Judge using the Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package is required for Anthropic judge. Install with: pip install anthropic")

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set and no api_key provided")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = _build_judge_prompt(question, response, expected, q_type)

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=80,
                temperature=0.0,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text if message.content else ""
            break
        except Exception as exc:
            logger.warning("Anthropic judge attempt %d failed: %s", attempt + 1, exc)
            if attempt == max_retries - 1:
                text = "uncertain"

    label = _parse_label(text)
    is_correct = label == "correct" or (q_type in ("adversarial", "hallucination", "uncertain") and label == "uncertain")

    return {
        "correct": is_correct,
        "label": label,
        "reason": text.strip(),
        "judge_type": f"anthropic:{model}",
    }


def get_judge(
    judge_type: str = "local",
    judge_model=None,
    judge_tokenizer=None,
    device: str = "cpu",
):
    """Return a callable judge function.

    Args:
        judge_type: "local", "openai", or "anthropic".
        judge_model: local HF model (required if judge_type="local").
        judge_tokenizer: local HF tokenizer (required if judge_type="local").
        device: torch device for local judge.

    Returns:
        Callable with signature (question, response, expected, q_type) -> dict.
    """
    if judge_type == "local":
        if judge_model is None or judge_tokenizer is None:
            raise ValueError("local judge requires judge_model and judge_tokenizer")
        return lambda q, r, e, t: local_model_judge(
            judge_model, judge_tokenizer, q, r, e, t, device
        )
    if judge_type == "openai":
        return lambda q, r, e, t: openai_judge(q, r, e, t)
    if judge_type == "anthropic":
        return lambda q, r, e, t: anthropic_judge(q, r, e, t)
    raise ValueError(f"Unknown judge_type: {judge_type}")


def evaluate_judge_consistency(
    judge_fn,
    samples: List[Dict],
    method_outputs: Dict[str, List[str]],
) -> Dict:
    """Measure how often the judge gives consistent labels across repeated calls.

    Useful for sanity-checking judge reliability on a small held-out set.
    """
    disagreements = 0
    total = 0
    for sample in samples:
        for method_name, texts in method_outputs.items():
            # Find text for this sample index
            idx = next((i for i, item in enumerate(texts) if item["sample"] == sample), None)
            if idx is None:
                continue
            text = texts[idx]["text"]
            r1 = judge_fn(sample["prompt"], text, sample["expected"], sample["type"])
            r2 = judge_fn(sample["prompt"], text, sample["expected"], sample["type"])
            total += 1
            if r1["label"] != r2["label"]:
                disagreements += 1

    return {
        "total": total,
        "disagreements": disagreements,
        "consistency": 1.0 - (disagreements / total) if total > 0 else 1.0,
    }
