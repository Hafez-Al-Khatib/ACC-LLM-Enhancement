"""Download / cache Mistral 7B model weights."""

import argparse
import logging
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def download_model(model_id: str, output_dir: str, trust_remote_code: bool = False):
    """Download model + tokenizer from HuggingFace hub to a local path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.save_pretrained(out)

    logger.info("Downloading model (this may take several minutes): %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
        device_map="cpu",  # download to CPU RAM, then save
    )
    model.save_pretrained(out)

    logger.info("Model saved to %s", out.resolve())
    logger.info("Size: ~%.1f GB", sum(
        f.stat().st_size for f in out.rglob("*") if f.is_file()
    ) / 1e9)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        default="models/mistral_7b",
        help="Local path to save weights",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    download_model(args.model_id, args.output_dir, args.trust_remote_code)
