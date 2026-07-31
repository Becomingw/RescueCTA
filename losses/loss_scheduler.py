"""Loss权重调度器 - 统一管理各个损失函数的权重变化

设计原则：
1. 完全动态：不预定义任何 loss 名称，所有 loss 通过配置文件定义
2. 可扩展：添加新 loss 只需在配置中添加对应项
3. 统一接口：所有 loss 权重通过 get_weight(name, epoch) 获取
"""

from typing import Dict, Any, Optional, Callable
import math


class LossScheduler:
    """统一的Loss权重调度器

    完全动态的调度器，不预定义任何 loss 名称。
    所有 loss 及其调度策略通过配置文件定义。

    支持的调度类型:
    - constant: 恒定权重
    - linear: 线性变化 (从start到end)
    - cosine: 余弦退火
    - step: 阶梯式变化

    通用参数 (适用于 linear, cosine):
    - start: 起始权重值
    - end: 结束权重值
    - start_epoch: 开始变化的 epoch (默认 0)
    - end_epoch: 结束变化的 epoch (默认 total_epoch)
      - 在 start_epoch 之前保持 start 值
      - 在 end_epoch 之后保持 end 值

    可扩展性:
    - 通过 register_schedule_type() 注册自定义调度类型
    - 配置中添加任意名称的 loss 即可自动支持

    配置示例:
    ```yaml
    loss_schedule:
      # 恒定权重
      L1:
        type: "constant"
        value: 100

      # 从第0轮到第300轮，0.1 -> 1.0
      GAN:
        type: "linear"
        start: 0.1
        end: 1.0

      # 从第100轮到第200轮，0 -> 5（之前为0，之后为5）
      edge:
        type: "linear"
        start: 0
        end: 5
        start_epoch: 100
        end_epoch: 200

      # 余弦退火：从第50轮到第250轮
      VGG:
        type: "cosine"
        start: 10
        end: 5
        start_epoch: 50
        end_epoch: 250

      # 阶梯式变化
      feat:
        type: "step"
        steps: [[0, 10], [100, 8], [200, 5]]
    ```
    """

    # 内置调度类型
    BUILTIN_TYPES = {'constant', 'linear', 'cosine', 'step'}

    def __init__(self, total_epochs: int, schedule_config: Dict[str, Any]):
        """
        Args:
            total_epochs: 总训练轮数
            schedule_config: 调度配置字典，格式为 {loss_name: schedule_params}
        """
        if not schedule_config:
            raise ValueError("schedule_config cannot be empty. "
                           "Please define at least one loss in loss_schedule.")

        self.total_epochs = total_epochs
        self.config = schedule_config
        self._custom_schedulers: Dict[str, Callable] = {}
        self._validate_config()

    def register_schedule_type(self, type_name: str,
                                scheduler_fn: Callable[[int, Dict[str, Any], int], float]):
        """注册自定义调度类型

        Args:
            type_name: 调度类型名称
            scheduler_fn: 调度函数，签名为 (current_epoch, config, total_epochs) -> float

        Example:
            def my_scheduler(epoch, cfg, total):
                # 自定义调度逻辑
                return cfg['base'] * (1 + epoch / total)

            scheduler.register_schedule_type('my_type', my_scheduler)
        """
        self._custom_schedulers[type_name] = scheduler_fn

    def _validate_config(self):
        """验证配置的有效性"""
        for loss_name, cfg in self.config.items():
            schedule_type = cfg.get('type', 'constant')

            # 检查是否为已知类型
            if schedule_type not in self.BUILTIN_TYPES and schedule_type not in self._custom_schedulers:
                raise ValueError(f"Unknown schedule type '{schedule_type}' for loss '{loss_name}'. "
                               f"Built-in types: {self.BUILTIN_TYPES}. "
                               f"Custom types: {set(self._custom_schedulers.keys())}")

            # 验证内置类型的必要参数
            if schedule_type == 'constant':
                if 'value' not in cfg:
                    raise ValueError(f"Schedule '{loss_name}' with type 'constant' requires 'value'")
            elif schedule_type in ('linear', 'cosine'):
                if 'start' not in cfg or 'end' not in cfg:
                    raise ValueError(f"Schedule '{loss_name}' with type '{schedule_type}' "
                                   "requires 'start' and 'end'")
                # start_epoch 和 end_epoch 是可选的
            elif schedule_type == 'step':
                if 'steps' not in cfg:
                    raise ValueError(f"Schedule '{loss_name}' with type 'step' requires 'steps' list")

    def get_weight(self, loss_name: str, current_epoch: int, default: float = 0.0) -> float:
        """获取指定loss在当前epoch的权重

        Args:
            loss_name: 损失函数名称
            current_epoch: 当前epoch (从0开始)
            default: 如果 loss_name 不存在时返回的默认值

        Returns:
            当前epoch的权重值
        """
        if loss_name not in self.config:
            return default

        cfg = self.config[loss_name]
        schedule_type = cfg.get('type', 'constant')

        # 检查自定义调度器
        if schedule_type in self._custom_schedulers:
            return self._custom_schedulers[schedule_type](current_epoch, cfg, self.total_epochs)

        # 内置调度器
        if schedule_type == 'constant':
            return float(cfg['value'])

        elif schedule_type == 'linear':
            start_epoch = cfg.get('start_epoch', 0)
            end_epoch = cfg.get('end_epoch', self.total_epochs)
            return self._linear_schedule(
                current_epoch,
                cfg['start'],
                cfg['end'],
                start_epoch=start_epoch,
                end_epoch=end_epoch
            )

        elif schedule_type == 'cosine':
            start_epoch = cfg.get('start_epoch', 0)
            end_epoch = cfg.get('end_epoch', self.total_epochs)
            return self._cosine_schedule(
                current_epoch,
                cfg['start'],
                cfg['end'],
                start_epoch=start_epoch,
                end_epoch=end_epoch
            )

        elif schedule_type == 'step':
            return self._step_schedule(current_epoch, cfg['steps'])

        return default

    def _linear_schedule(self, current_epoch: int, start: float, end: float,
                         start_epoch: int = 0, end_epoch: Optional[int] = None) -> float:
        """线性调度

        Args:
            current_epoch: 当前 epoch
            start: 起始值
            end: 结束值
            start_epoch: 开始变化的 epoch
            end_epoch: 结束变化的 epoch
        """
        if end_epoch is None:
            end_epoch = self.total_epochs

        # 在 start_epoch 之前，返回 start 值
        if current_epoch < start_epoch:
            return float(start)

        # 在 end_epoch 之后，返回 end 值
        if current_epoch >= end_epoch:
            return float(end)

        # 计算变化区间内的进度
        duration = end_epoch - start_epoch
        if duration <= 0:
            return float(end)

        progress = (current_epoch - start_epoch) / duration
        value = start + (end - start) * progress

        return float(value)

    def _cosine_schedule(self, current_epoch: int, start: float, end: float,
                         start_epoch: int = 0, end_epoch: Optional[int] = None) -> float:
        """余弦退火调度

        Args:
            current_epoch: 当前 epoch
            start: 起始值
            end: 结束值
            start_epoch: 开始变化的 epoch
            end_epoch: 结束变化的 epoch
        """
        if end_epoch is None:
            end_epoch = self.total_epochs

        # 在 start_epoch 之前，返回 start 值
        if current_epoch < start_epoch:
            return float(start)

        # 在 end_epoch 之后，返回 end 值
        if current_epoch >= end_epoch:
            return float(end)

        # 计算变化区间内的进度
        duration = end_epoch - start_epoch
        if duration <= 0:
            return float(end)

        progress = (current_epoch - start_epoch) / duration
        # 余弦退火: 从start平滑过渡到end
        cosine_factor = (1 - math.cos(math.pi * progress)) / 2
        return start + (end - start) * cosine_factor

    def _step_schedule(self, current_epoch: int, steps: list) -> float:
        """阶梯式调度

        steps格式: [[epoch1, value1], [epoch2, value2], ...]
        """
        if not steps:
            return 0.0

        sorted_steps = sorted(steps, key=lambda x: x[0])
        value = sorted_steps[0][1]

        for epoch, step_value in sorted_steps:
            if current_epoch >= epoch:
                value = step_value
            else:
                break

        return float(value)

    def get_all_weights(self, current_epoch: int) -> Dict[str, float]:
        """获取所有loss在当前epoch的权重"""
        return {
            loss_name: self.get_weight(loss_name, current_epoch)
            for loss_name in self.config.keys()
        }

    def get_loss_names(self) -> list:
        """获取所有配置的 loss 名称"""
        return list(self.config.keys())

    def has_loss(self, loss_name: str) -> bool:
        """检查是否配置了指定的 loss"""
        return loss_name in self.config

    def get_summary(self) -> str:
        """获取调度器配置摘要"""
        lines = ["Loss Scheduler Configuration:"]
        lines.append(f"  Total epochs: {self.total_epochs}")
        lines.append(f"  Configured losses: {len(self.config)}")
        lines.append("")

        for loss_name, cfg in self.config.items():
            schedule_type = cfg.get('type', 'constant')
            if schedule_type == 'constant':
                lines.append(f"  {loss_name}: constant = {cfg['value']}")
            elif schedule_type == 'linear':
                start_e = cfg.get('start_epoch', 0)
                end_e = cfg.get('end_epoch', self.total_epochs)
                lines.append(f"  {loss_name}: linear {cfg['start']} -> {cfg['end']} "
                           f"(epoch {start_e}-{end_e})")
            elif schedule_type == 'cosine':
                start_e = cfg.get('start_epoch', 0)
                end_e = cfg.get('end_epoch', self.total_epochs)
                lines.append(f"  {loss_name}: cosine {cfg['start']} -> {cfg['end']} "
                           f"(epoch {start_e}-{end_e})")
            elif schedule_type == 'step':
                lines.append(f"  {loss_name}: step {cfg['steps']}")
            else:
                lines.append(f"  {loss_name}: {schedule_type} (custom)")

        return "\n".join(lines)

    def __repr__(self):
        return f"LossScheduler(total_epochs={self.total_epochs}, losses={list(self.config.keys())})"


# 预定义的常用配置模板
LOSS_SCHEDULE_TEMPLATES = {
    # 医学图像翻译推荐配置：L1为主，GAN渐进增强
    'medical_translation': {
        'L1': {'type': 'constant', 'value': 100},
        'GAN': {'type': 'linear', 'start': 0.1, 'end': 1.0},  # 全程渐增
        'VGG': {'type': 'constant', 'value': 10},
        'feat': {'type': 'constant', 'value': 10},
        'edge': {'type': 'constant', 'value': 0},
        'laplacian': {'type': 'constant', 'value': 0},
    },

    # 标准 GAN 配置
    'standard_gan': {
        'L1': {'type': 'linear', 'start': 10, 'end': 0},  # 全程递减
        'GAN': {'type': 'constant', 'value': 1.0},
        'VGG': {'type': 'constant', 'value': 10},
        'feat': {'type': 'constant', 'value': 10},
        'edge': {'type': 'constant', 'value': 0},
        'laplacian': {'type': 'constant', 'value': 0},
    },

    # 细节增强配置 (300 epochs 为例)
    'detail_enhanced': {
        'L1': {'type': 'constant', 'value': 50},
        'GAN': {'type': 'linear', 'start': 0.1, 'end': 1.0},
        'VGG': {'type': 'constant', 'value': 10},
        'feat': {'type': 'constant', 'value': 10},
        # 从第100轮到第200轮启动边缘损失
        'edge': {'type': 'linear', 'start': 0, 'end': 5, 'start_epoch': 100, 'end_epoch': 200},
        # 从第150轮到第250轮启动拉普拉斯损失
        'laplacian': {'type': 'linear', 'start': 0, 'end': 3, 'start_epoch': 150, 'end_epoch': 250},
    },
}


def get_template(name: str) -> Dict[str, Any]:
    """获取预定义的配置模板

    Args:
        name: 模板名称 ('medical_translation', 'standard_gan', 'detail_enhanced')

    Returns:
        配置字典的深拷贝
    """
    if name not in LOSS_SCHEDULE_TEMPLATES:
        raise ValueError(f"Unknown template: {name}. "
                        f"Available: {list(LOSS_SCHEDULE_TEMPLATES.keys())}")

    import copy
    return copy.deepcopy(LOSS_SCHEDULE_TEMPLATES[name])
