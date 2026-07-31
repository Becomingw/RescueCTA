"""损失函数模块"""

from .gan_loss import GANLoss
from .vgg_loss import VGGLoss, Vgg19, EdgeLoss, LaplacianLoss
from .loss_module import LossModule, add_3channel, focal_loss_weighted_loss
from .loss_scheduler import LossScheduler, LOSS_SCHEDULE_TEMPLATES, get_template

__all__ = [
    'GANLoss', 'VGGLoss', 'Vgg19', 'EdgeLoss', 'LaplacianLoss',
    'LossModule', 'add_3channel', 'focal_loss_weighted_loss',
    'LossScheduler', 'LOSS_SCHEDULE_TEMPLATES', 'get_template'
]
