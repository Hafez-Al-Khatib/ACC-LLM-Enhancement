"""Training script for QLoRA fine-tuning."""

import json
import logging
import os
from pathlib import Path

import yaml
from datasets import Dataset
from trl import SFTTrainer

from src.model_utils import (
    attach_lora,
    load_model,
    load_tokenizer,
    make_bnb_config,
)

logger = logging.getLogger(__name__)


def load_dataset_from_config(ds_cfg: dict):
    """Load dataset from JSONL or HF hub.

    Expected JSONL format (one dict per line):
        {"text": "instruction\ninput\noutput"}
    or with explicit fields:
        {"instruction": "...", "input": "...", "output": "..."}
    """
    path = ds_cfg["path"]
    fmt = ds_cfg.get("format", "jsonl")
    text_col = ds_cfg.get("text_column", "text")

    if fmt == "jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
        ds = Dataset.from_list(records)
    elif fmt == "json":
        with open(path, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        ds = Dataset.from_list(records)
    elif fmt == "hf":
        from datasets import load_dataset as hf_load
        ds = hf_load(path, split="train")
    else:
        raise ValueError(f"Unknown dataset format: {fmt}")

    # If text column doesn't exist but instruction/output do, format it
    if text_col not in ds.column_names:
        logger.info("Formatting dataset from instruction/output fields")

        def format_example(ex):
            inst = ex.get("instruction", "")
            inp = ex.get("input", "")
            out = ex.get("output", "")
            text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            return {text_col: text}

        ds = ds.map(format_example)

    return ds


def build_trainer(config: dict):
    """Build an SFTTrainer from a config dict."""
    # --- Model ---
    bnb = make_bnb_config(config["quantization"])
    dtype_name = config["model"].get("torch_dtype", "bfloat16")
    import torch
    torch_dtype = getattr(torch, dtype_name)

    model = load_model(
        config["model"]["base_model"],
        bnb_config=bnb,
        torch_dtype=torch_dtype,
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    tokenizer = load_tokenizer(config["model"]["base_model"])
    model = attach_lora(model, config["lora"])

    # --- Dataset ---
    train_ds = load_dataset_from_config(config["dataset"])
    eval_ds = None
    if config["dataset"].get("eval_path"):
        eval_cfg = dict(config["dataset"])
        eval_cfg["path"] = eval_cfg.pop("eval_path")
        eval_ds = load_dataset_from_config(eval_cfg)

    # --- Training args ---
    tc = config["training"]
    training_args = dict(
        output_dir=tc["output_dir"],
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=tc.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        warmup_steps=tc.get("warmup_steps", 0),
        logging_steps=tc["logging_steps"],
        save_steps=tc["save_steps"],
        eval_steps=tc.get("eval_steps", tc["save_steps"]),
        evaluation_strategy=tc.get("evaluation_strategy", "steps"),
        save_strategy=tc.get("save_strategy", "steps"),
        load_best_model_at_end=tc.get("load_best_model_at_end", True),
        metric_for_best_model=tc.get("metric_for_best_model", "eval_loss"),
        greater_is_better=tc.get("greater_is_better", False),
        optim=tc.get("optim", "paged_adamw_8bit"),
        lr_scheduler_type=tc.get("lr_scheduler_type", "cosine"),
        max_grad_norm=tc.get("max_grad_norm", 0.3),
        logging_dir=os.path.join(tc["output_dir"], "logs"),
        report_to="wandb" if config.get("wandb") else None,
        run_name=config.get("wandb", {}).get("run_name"),
        fp16=torch_dtype == torch.float16,
        bf16=torch_dtype == torch.bfloat16,
        # Dataloader
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # --- SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        max_seq_length=tc.get("max_seq_length", 512),
        packing=tc.get("packing", False),
        args=training_args,
        dataset_text_field=config["dataset"]["text_column"],
    )
    return trainer


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # WandB init
    wandb_cfg = config.get("wandb")
    if wandb_cfg:
        import wandb
        wandb.init(
            project=wandb_cfg["project"],
            name=wandb_cfg.get("run_name"),
            tags=wandb_cfg.get("tags", []),
            config=config,
        )

    trainer = build_trainer(config)
    logger.info("Starting training...")
    trainer.train()

    # Save final adapter
    output_dir = config["training"]["output_dir"]
    trainer.save_model(os.path.join(output_dir, "final_adapter"))
    logger.info("Training complete. Adapter saved to %s", output_dir)

    if wandb_cfg:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Mistral 7B")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    main(args.config)
