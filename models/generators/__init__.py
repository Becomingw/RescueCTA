"""RescueCTA generator factory."""

import torch
import torch.nn as nn

from .rescuecta import RescueCTAGenerator, build_rescuecta_generator


def weights_init(module: nn.Module) -> None:
    """Initialize convolutional and batch-normalization layers."""
    classname = module.__class__.__name__
    if "Conv" in classname and getattr(module, "weight", None) is not None:
        module.weight.data.normal_(0.0, 0.02)
    elif "BatchNorm2d" in classname:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


def build_generator(
    name: str = "rescuecta",
    device: torch.device | None = None,
    **kwargs,
) -> nn.Module:
    """Build the RescueCTA generator.

    ``HR`` is accepted as a compatibility alias for historical configs and
    checkpoints; new configs should use ``rescuecta``.
    """
    normalized_name = name.lower()
    if normalized_name not in {"rescuecta", "hr"}:
        raise NotImplementedError(
            f"Unknown generator {name!r}; expected 'rescuecta'."
        )

    in_channels = kwargs.get("in_channels", 3)
    out_channels = kwargs.get("out_channels", in_channels)
    num_stages = kwargs.get("num_stages", 6)
    generator = build_rescuecta_generator(
        num_stages=num_stages,
        in_channels=in_channels,
        out_channels=out_channels,
    )

    generator.apply(weights_init)
    if device is not None:
        generator = generator.to(device)
    return generator


__all__ = [
    "RescueCTAGenerator",
    "build_generator",
    "build_rescuecta_generator",
]
