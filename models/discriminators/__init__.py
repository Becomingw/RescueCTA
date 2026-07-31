"""Discriminator模块 - 提供判别器架构"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .multiscale import MultiscaleDiscriminator, NLayerDiscriminator


def weights_init(m):
    """初始化网络权重"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def build_discriminator(name: str, device: torch.device = None, **kwargs) -> nn.Module:
    """
    构建判别器

    Args:
        name: 判别器类型 - "multi", "multi_feat"
        device: 目标设备
        **kwargs: 额外参数

    Returns:
        初始化后的判别器模型
    """
    input_nc = kwargs.get('input_nc', 2)

    if name == "multi":
        discriminator = MultiscaleDiscriminator(
            input_nc=input_nc, ndf=64, n_layers=3,
            norm_layer=spectral_norm, getIntermFeat=False
        )
    elif name == "multi_feat":
        discriminator = MultiscaleDiscriminator(
            input_nc=input_nc, ndf=64, n_layers=3,
            norm_layer=spectral_norm, getIntermFeat=True
        )
    else:
        raise NotImplementedError(f'Discriminator model name [{name}] is not recognized')

    if device is not None:
        discriminator = discriminator.to(device)

    discriminator.apply(weights_init)
    return discriminator


__all__ = ['build_discriminator', 'MultiscaleDiscriminator', 'NLayerDiscriminator']
