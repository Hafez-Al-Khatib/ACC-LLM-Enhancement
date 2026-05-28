"""Entropy-based uncertainty monitoring inspired by the anterior cingulate cortex.

The ACC in the brain detects conflict and uncertainty, signaling when additional
cognitive control is needed. This module mirrors that role for language models:
per-token Shannon entropy over the output distribution serves as a proxy for
predictive uncertainty, and configurable thresholds trigger downstream actions
(flagging, regenerating, or surfacing a warning to the caller).

Typical usage::

    monitor = EntropyMonitor(threshold=3.5, mode="absolute", action="flag")
    for step_logits in generation_loop():
        h = monitor.compute_entropy(step_logits)
        if monitor.check_threshold(h):
            ...  # handle high-uncertainty token
    score = monitor.get_confidence_score()
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Literal, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ThresholdMode = Literal["absolute", "moving_average", "percentile"]
Action = Literal["flag", "regenerate", "warning"]

_EPS = 1e-12


@dataclass
class EntropyEvent:
    """A single threshold breach recorded by the monitor."""

    step: int
    entropy: float
    threshold: float
    action: Action


class SlidingWindowEntropy:
    """Fixed-size ring buffer of recent per-token entropies.

    Provides O(1) updates and cheap summary statistics (mean, std, percentile)
    used by the higher-level monitor for temporal smoothing. Keeping this
    decoupled from EntropyMonitor lets callers reuse it standalone (e.g. for
    plotting) without dragging in threshold/action logic.
    """

    def __init__(self, window_size: int = 32):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self._buf: Deque[float] = deque(maxlen=window_size)

    def update(self, entropy: float) -> None:
        self._buf.append(float(entropy))

    def __len__(self) -> int:
        return len(self._buf)

    def is_full(self) -> bool:
        return len(self._buf) == self.window_size

    def mean(self) -> float:
        if not self._buf:
            return 0.0
        return sum(self._buf) / len(self._buf)

    def std(self) -> float:
        n = len(self._buf)
        if n < 2:
            return 0.0
        m = self.mean()
        return math.sqrt(sum((x - m) ** 2 for x in self._buf) / (n - 1))

    def percentile(self, q: float) -> float:
        """Linear-interpolated percentile, q in [0, 100]."""
        if not self._buf:
            return 0.0
        if not 0.0 <= q <= 100.0:
            raise ValueError("q must be in [0, 100]")
        s = sorted(self._buf)
        if len(s) == 1:
            return s[0]
        idx = (q / 100.0) * (len(s) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return s[lo]
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    def values(self) -> List[float]:
        return list(self._buf)

    def reset(self) -> None:
        self._buf.clear()


class EntropyMonitor:
    """Per-token entropy monitor with configurable thresholds and actions.

    Parameters
    ----------
    threshold : float
        Interpretation depends on `mode`:
          - "absolute": entropy in nats above this value is a breach.
          - "moving_average": breach when current entropy exceeds
            `window.mean() * threshold` (treated as a multiplier, e.g. 1.5).
          - "percentile": breach when entropy exceeds the given percentile of
            the sliding window (e.g. threshold=95 for the 95th percentile).
    mode : {"absolute", "moving_average", "percentile"}
        Threshold strategy.
    action : {"flag", "regenerate", "warning"}
        Suggested response on breach. The monitor only records the action;
        executing it is the caller's responsibility.
    window_size : int
        Sliding window length for moving-average / percentile modes and for
        the running confidence score.
    base : float
        Logarithm base for entropy. Default e (nats); pass 2.0 for bits.
    warmup : int
        Minimum number of observed tokens before moving-average / percentile
        thresholds are evaluated. Below this, `check_threshold` returns False.

    Notes
    -----
    Entropy is computed in float32 regardless of input dtype, to avoid the
    catastrophic precision loss that bf16/fp16 softmax + log produces in the
    tails of the distribution.
    """

    def __init__(
        self,
        threshold: float = 3.5,
        mode: ThresholdMode = "absolute",
        action: Action = "flag",
        window_size: int = 32,
        base: float = math.e,
        warmup: int = 4,
    ):
        if mode not in ("absolute", "moving_average", "percentile"):
            raise ValueError(f"unknown mode: {mode}")
        if action not in ("flag", "regenerate", "warning"):
            raise ValueError(f"unknown action: {action}")
        if base <= 1.0:
            raise ValueError("base must be > 1")
        if warmup < 0:
            raise ValueError("warmup must be >= 0")

        self.threshold = float(threshold)
        self.mode = mode
        self.action = action
        self.base = float(base)
        self.warmup = warmup
        self.window = SlidingWindowEntropy(window_size)
        self._log_base = math.log(self.base)
        self._step = 0
        self._events: List[EntropyEvent] = []

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """Shannon entropy of the next-token distribution from logits.

        Accepts logits of shape (vocab,), (1, vocab), or (batch, seq, vocab);
        in the last case the final-position row is used (standard generation
        convention). Returned value is a Python float in the configured base.
        """
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")

        if logits.dim() == 3:
            row = logits[0, -1]
        elif logits.dim() == 2:
            row = logits[-1]
        elif logits.dim() == 1:
            row = logits
        else:
            raise ValueError(f"unsupported logits shape: {tuple(logits.shape)}")

        row = row.detach().to(torch.float32)
        log_probs = F.log_softmax(row, dim=-1)
        probs = log_probs.exp()
        # nats; convert to requested base afterwards.
        entropy_nats = (-(probs * log_probs).sum()).clamp(min=0.0).item()
        h = entropy_nats / self._log_base
        return float(h)

    def observe(self, logits: torch.Tensor) -> float:
        """Compute entropy, push it into the window, and return it.

        Convenience wrapper for the common case where the caller wants both
        the value and to advance the monitor's internal state in one call.
        """
        h = self.compute_entropy(logits)
        self.window.update(h)
        self._step += 1
        return h

    def _current_threshold(self) -> Optional[float]:
        if self.mode == "absolute":
            return self.threshold
        if len(self.window) < max(self.warmup, 1):
            return None
        if self.mode == "moving_average":
            return self.window.mean() * self.threshold
        return self.window.percentile(self.threshold)

    def check_threshold(self, entropy: float) -> bool:
        """Return True if `entropy` breaches the active threshold.

        Records an EntropyEvent on breach. For moving-average / percentile
        modes, returns False until `warmup` tokens have been observed.
        """
        thr = self._current_threshold()
        if thr is None:
            return False
        breached = entropy > thr
        if breached:
            self._events.append(
                EntropyEvent(
                    step=self._step,
                    entropy=float(entropy),
                    threshold=float(thr),
                    action=self.action,
                )
            )
        return breached

    def get_confidence_score(self) -> float:
        """Aggregate confidence in [0, 1] derived from the sliding window.

        Defined as ``1 - mean(H) / log(V_eff)`` where ``V_eff`` is implied
        by the maximum entropy observed so far (clamped to avoid div-by-zero
        on a freshly reset monitor). Higher is more confident. With no
        observations yet, returns 1.0.
        """
        if len(self.window) == 0:
            return 1.0
        mean_h = self.window.mean()
        # Anchor normalization on the empirical max, which upper-bounds the
        # distribution's entropy without requiring vocab size to be passed in.
        max_h = max(self.window.values()) + _EPS
        return float(max(0.0, min(1.0, 1.0 - mean_h / max_h)))

    @property
    def events(self) -> List[EntropyEvent]:
        return list(self._events)

    def reset(self) -> None:
        self.window.reset()
        self._events.clear()
        self._step = 0
