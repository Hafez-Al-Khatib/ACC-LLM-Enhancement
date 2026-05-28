"""Training script for QLoRA fine-tuning."""

import json
import logging
import os
from pathlib import Path

import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import torch

logger = logging.getLogger(__name__)


def load_dataset_from_config(ds_cfg: dict):
    """Load dataset from JSONL or HF hub."""
    path = ds_cfg["path"]
    fmt = ds_cfg.get("format", "jsonl")
    text_col = ds_cfg.get("text_column", "text")

    if fmt in ("jsonl", "json"):
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

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


def load_tokenizer(model_path: str, trust_remote_code: bool = False, local_files_only: bool = False):
    """Load tokenizer with padding side fix for decoder-only models."""
    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_model(
    model_path: str,
    bnb_config=None,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
    device_map: str = "auto",
    local_files_only: bool = False,
):
    """Load causal LM with optional 4-bit quantization."""
    logger.info("Loading model from %s (dtype=%s, 4bit=%s)",
                model_path, torch_dtype, bnb_config is not None)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype if bnb_config is None else None,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        local_files_only=local_files_only,
    )
    if getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    return model


def attach_lora(model, lora_cfg: dict):
    """Attach LoRA adapters."""
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, config)
    logger.info("LoRA attached: r=%d, alpha=%d, trainable params=%s",
                config.r, config.lora_alpha,
                sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def make_bnb_config(quant_cfg: dict):
    """Build BitsAndBytesConfig from YAML dict."""
    if not quant_cfg.get("load_in_4bit", False):
        return None

    compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
    )


def build_trainer(config: dict):
    """Build a Trainer from a config dict (no TRL dependency)."""
    bnb = make_bnb_config(config["quantization"])
    dtype_name = config["model"].get("torch_dtype", "bfloat16")
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

    # Tokenize
    max_length = config["training"].get("max_seq_length", 512)
    text_col = config["dataset"]["text_column"]

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    if eval_ds:
        eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names)

    # --- Training args ---
    tc = config["training"]
    training_args = TrainingArguments(
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
        eval_strategy=tc.get("evaluation_strategy", "steps"),
        save_strategy=tc.get("save_strategy", "steps"),
        load_best_model_at_end=tc.get("load_best_model_at_end", True),
        metric_for_best_model=tc.get("metric_for_best_model", "eval_loss"),
        greater_is_better=tc.get("greater_is_better", False),
        optim=tc.get("optim", "paged_adamw_8bit"),
        lr_scheduler_type=tc.get("lr_scheduler_type", "cosine"),
        max_grad_norm=tc.get("max_grad_norm", 0.3),
        logging_dir=os.path.join(tc["output_dir"], "logs"),
        report_to=["wandb"] if config.get("wandb") else [],
        fp16=torch_dtype == torch.float16,
        bf16=torch_dtype == torch.bfloat16,
        dataloader_num_workers=0,
    )

    # mlm=False tells the collator to create labels = input_ids shifted by 1 for causal LM.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        data_collator=data_collator,
    )
    return trainer


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

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

    output_dir = config["training"]["output_dir"]
    trainer.save_model(os.path.join(output_dir, "final_adapter"))
    logger.info("Training complete. Adapter saved to %s", output_dir)

    if wandb_cfg:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="QLoRA fine-tuning")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    main(args.config)
