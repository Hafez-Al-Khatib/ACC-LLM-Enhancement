"""Model loading with QLoRA quantization.

Supports Intel Arc (XPU), NVIDIA CUDA, and CPU backends.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .device_utils import get_best_device

logger = logging.getLogger(__name__)


def load_tokenizer(
    model_path: str,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
):
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
    bnb_config: Optional[BitsAndBytesConfig] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
    device_map: Optional[str] = None,
    device: Optional[str] = None,
    local_files_only: bool = False,
):
    """Load causal LM with optional 4-bit quantization.

    Parameters
    ----------
    model_path : str
        Local path or HuggingFace hub ID.
    bnb_config : BitsAndBytesConfig | None
        If provided, enables 4-bit (or 8-bit) quantization.
    torch_dtype : torch.dtype
        Compute dtype. Use float16 on Jetson (no bfloat16 support).
    device_map : str | None
        ``"auto"`` for multi-GPU, ``"cuda:0"`` for single GPU,
        ``"cpu"`` for CPU-only. If None, the model is loaded to the
        best available device (XPU → CUDA → CPU).
    device : str | None
        Explicit device override (e.g. ``"xpu"``, ``"cuda:0"``, ``"cpu"``).
        Takes precedence over *device_map*.
    local_files_only : bool
        If False (default), allows HF to download missing files.
    """
    logger.info(
        "Loading model from %s (dtype=%s, 4bit=%s)",
        model_path,
        torch_dtype,
        bnb_config is not None,
    )

    # Determine target device
    target_device = device
    if target_device is None:
        if device_map is not None:
            target_device = device_map
        else:
            target_device = str(get_best_device())

    # For XPU, load to CPU first then move (avoid device_map quirks)
    if target_device.startswith("xpu"):
        logger.info("XPU detected: loading to CPU first, then moving to XPU")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype if bnb_config is None else None,
            trust_remote_code=trust_remote_code,
            device_map="cpu",
            local_files_only=local_files_only,
        )
        model = model.to(target_device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype if bnb_config is None else None,
            trust_remote_code=trust_remote_code,
            device_map=target_device if not target_device.startswith("xpu") else "cpu",
            local_files_only=local_files_only,
        )

    # Gradient checkpointing saves ~30% activation memory
    if getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    logger.info("Model loaded on %s", next(model.parameters()).device)
    return model


def attach_lora(model, lora_cfg: dict):
    """Attach LoRA adapters to a quantized model.

    Must call `prepare_model_for_kbit_training` before this for 4-bit.
    """
    # Enable gradient on input embeddings (required for some models)
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
    logger.info(
        "LoRA attached: r=%d, alpha=%d, trainable params=%s",
        config.r,
        config.lora_alpha,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return model


def make_bnb_config(quant_cfg: dict) -> Optional[BitsAndBytesConfig]:
    """Build BitsAndBytesConfig from YAML dict.

    Returns None if quantization is disabled.
    """
    if not quant_cfg.get("load_in_4bit", False):
        return None

    compute_dtype = getattr(
        torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
    )
