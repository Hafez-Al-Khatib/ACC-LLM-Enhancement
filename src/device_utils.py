"""Device auto-detection and management for cross-platform GPU support.

Supports:
  - Intel Arc / XPU (via PyTorch native XPU or IPEX)
  - NVIDIA CUDA
  - CPU fallback

Usage::

    from src.device_utils import get_best_device, move_to_device
    device = get_best_device()
    model = move_to_device(model, device)
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


def get_best_device(prefer: Optional[str] = None) -> torch.device:
    """Return the best available torch device.

    Priority order (unless overridden by *prefer*):
      1. XPU (Intel Arc / Core Ultra iGPU)
      2. CUDA (NVIDIA GPU)
      3. CPU (fallback)

    Parameters
    ----------
    prefer : {"xpu", "cuda", "cpu"}, optional
        Force a specific backend. If that backend is unavailable,
        falls back through the normal priority order.

    Returns
    -------
    torch.device
    """
    candidates = ["xpu", "cuda", "cpu"]

    if prefer is not None:
        prefer = prefer.lower()
        if prefer not in candidates:
            raise ValueError(f"Unknown device preference: {prefer}")
        # Move preferred device to front
        candidates = [prefer] + [c for c in candidates if c != prefer]

    for backend in candidates:
        if backend == "xpu":
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                device = torch.device("xpu:0")
                logger.info(
                    "Intel XPU detected: %s (Arc/Core Ultra)",
                    torch.xpu.get_device_name(0),
                )
                return device
        elif backend == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                logger.info(
                    "NVIDIA CUDA detected: %s",
                    torch.cuda.get_device_name(0),
                )
                return device
        elif backend == "cpu":
            device = torch.device("cpu")
            logger.info("Using CPU (no GPU detected)")
            return device

    # Should never reach here because CPU is always available
    return torch.device("cpu")


def get_device_info(device: Optional[torch.device] = None) -> dict:
    """Return a dict with human-readable device information."""
    if device is None:
        device = get_best_device()

    info = {"type": device.type, "index": device.index or 0}

    if device.type == "xpu":
        info["name"] = torch.xpu.get_device_name(device.index or 0)
        info["total_memory_mb"] = torch.xpu.get_device_properties(
            device.index or 0
        ).total_memory // (1024 * 1024)
    elif device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device.index or 0)
        info["total_memory_mb"] = torch.cuda.get_device_properties(
            device.index or 0
        ).total_memory // (1024 * 1024)
    else:
        info["name"] = "CPU"
        info["total_memory_mb"] = None

    return info


def move_to_device(
    obj: Union[torch.nn.Module, torch.Tensor],
    device: Optional[torch.device] = None,
) -> Union[torch.nn.Module, torch.Tensor]:
    """Move a model or tensor to *device* (auto-detect if None)."""
    if device is None:
        device = get_best_device()
    return obj.to(device)


def synchronize(device: Optional[torch.device] = None) -> None:
    """Synchronize the current device (no-op for CPU)."""
    if device is None:
        device = get_best_device()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
