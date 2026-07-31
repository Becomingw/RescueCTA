"""其他工具函数"""

import torch
import numpy as np
import random


def set_random_seed(seed):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def L1_linear_decay(total_epochs, current_epoch, initial_value, final_value=0):
    """
    L1损失权重线性衰减

    Args:
        total_epochs: 总训练轮数
        current_epoch: 当前轮数 (从0开始)
        initial_value: 初始L1权重值
        final_value: 最终L1权重值 (默认为0)

    Returns:
        当前epoch应该使用的L1权重值

    注意: 这个函数应该传入初始值，而不是上一轮的值
    """
    if total_epochs <= 1:
        return initial_value

    # 计算线性衰减
    progress = current_epoch / (total_epochs - 1)
    new_value = initial_value + (final_value - initial_value) * progress

    # 确保不会低于最终值
    if initial_value > final_value:
        new_value = max(new_value, final_value)
    else:
        new_value = min(new_value, final_value)

    return new_value


def add_3channel(input_tensor):
    """将单通道tensor复制为3通道"""
    return torch.cat((input_tensor, input_tensor, input_tensor), 1)
