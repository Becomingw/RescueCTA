"""EMA (Exponential Moving Average) utilities."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class ModelEMA:
    """Maintain an exponential moving average (EMA) of model weights.

    This keeps a detached copy of the model as ``ema`` and updates it after each
    generator optimizer step. In DDP, call ``update`` on every rank (model
    weights are synchronized by DDP so EMA stays consistent across ranks).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        update_every: int = 1,
        device: Optional[torch.device] = None,
        update_buffers: bool = True,
    ):
        if not (0.0 <= float(decay) <= 1.0):
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}")
        self.decay = float(decay)
        self.update_every = max(1, int(update_every))
        self.update_buffers = bool(update_buffers)
        self.num_updates = 0

        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.ema.to(device)

        # cache params/buffers for fast updates (structure matches due to deepcopy)
        self._ema_params = list(self.ema.parameters())
        self._ema_buffers = list(self.ema.buffers())

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights from ``model``."""
        self.num_updates += 1
        if self.num_updates % self.update_every != 0:
            return

        d = self.decay

        for ema_p, model_p in zip(self._ema_params, model.parameters()):
            if torch.is_floating_point(ema_p):
                ema_p.mul_(d).add_(model_p.detach(), alpha=1.0 - d)
            else:
                ema_p.copy_(model_p.detach())

        if self.update_buffers:
            for ema_b, model_b in zip(self._ema_buffers, model.buffers()):
                ema_b.copy_(model_b.detach())

    def get_meta(self) -> Dict[str, Any]:
        return {
            'decay': float(self.decay),
            'update_every': int(self.update_every),
            'update_buffers': bool(self.update_buffers),
            'num_updates': int(self.num_updates),
        }

    def load_meta(self, meta: Dict[str, Any]):
        if not isinstance(meta, dict):
            return
        if 'decay' in meta:
            self.decay = float(meta['decay'])
        if 'update_every' in meta:
            self.update_every = max(1, int(meta['update_every']))
        if 'update_buffers' in meta:
            self.update_buffers = bool(meta['update_buffers'])
        if 'num_updates' in meta:
            self.num_updates = int(meta['num_updates'])
