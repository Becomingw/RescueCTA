"""Loss Guard - 智能异常 loss 检测模块

支持多种检测模式：
1. zscore: 基于 Z-score 的统计异常检测（推荐）
2. ratio: 基于相对变化率的检测
3. threshold: 传统固定阈值检测（保留兼容性）
"""

from collections import deque
from typing import Dict, List, Optional, Tuple
import math


class LossGuard:
    """智能 loss 异常检测器

    使用滑动窗口统计，自动适应不同训练阶段的 loss 范围。

    Args:
        mode: 检测模式 - 'zscore', 'ratio', 'threshold', 'off'
        window_size: 滑动窗口大小（用于计算统计量）
        warmup_iters: 预热期迭代数（预热期内只收集数据，不触发异常）
        zscore_threshold: Z-score 阈值（mode='zscore' 时生效，默认 4.0）
        ratio_threshold: 变化率阈值（mode='ratio' 时生效，默认 3.0 表示 3 倍）
        skip_on_non_finite: 是否跳过 NaN/Inf
        max_thresholds: 固定阈值字典（mode='threshold' 时生效）

    Example config:
        loss_guard:
          enabled: true
          mode: "zscore"           # 推荐使用 zscore 模式
          window_size: 100         # 滑动窗口大小
          warmup_iters: 50         # 预热期
          zscore_threshold: 4.0    # Z-score 阈值（超过 4 个标准差视为异常）
          skip_on_non_finite: true # 跳过 NaN/Inf
    """

    def __init__(
        self,
        mode: str = 'zscore',
        window_size: int = 100,
        warmup_iters: int = 50,
        zscore_threshold: float = 4.0,
        ratio_threshold: float = 3.0,
        skip_on_non_finite: bool = True,
        max_thresholds: Optional[Dict[str, float]] = None,
        always_update_history: bool = True,
    ):
        self.mode = mode.lower() if mode else 'off'
        self.window_size = max(window_size, 10)
        self.warmup_iters = max(warmup_iters, 0)
        self.zscore_threshold = zscore_threshold
        self.ratio_threshold = ratio_threshold
        self.skip_on_non_finite = skip_on_non_finite
        self.max_thresholds = max_thresholds or {}
        self.always_update_history = always_update_history

        # 滑动窗口历史记录
        self._history: Dict[str, deque] = {}
        self._iter_count = 0

        # 统计信息
        self.skip_count = 0
        self.total_checks = 0

    def _get_history(self, key: str) -> deque:
        """获取或创建指定 loss 的历史记录"""
        if key not in self._history:
            self._history[key] = deque(maxlen=self.window_size)
        return self._history[key]

    def _compute_stats(self, history: deque) -> Tuple[float, float]:
        """计算均值和标准差"""
        if len(history) < 2:
            return 0.0, float('inf')

        values = list(history)
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 1e-8
        return mean, std

    def _check_zscore(self, key: str, value: float) -> Tuple[bool, str]:
        """基于 Z-score 的异常检测"""
        history = self._get_history(key)

        # 预热期：使用全局 iter 计数，只收集数据，不触发异常
        if self._iter_count <= self.warmup_iters:
            history.append(value)
            return False, ""

        mean, std = self._compute_stats(history)
        zscore = (value - mean) / std if std > 1e-8 else 0.0

        is_anomaly = zscore > self.zscore_threshold
        reason = ""
        if is_anomaly:
            reason = f"{key}_zscore={zscore:.2f}>(threshold={self.zscore_threshold}),mean={mean:.4f},std={std:.4f}"

        # 关键修复：总是更新历史，防止统计量冻结
        # 异常值也会被加入历史，但由于滑动窗口机制，会逐渐被正常值稀释
        if self.always_update_history:
            history.append(value)
        elif not is_anomaly:
            history.append(value)

        return is_anomaly, reason

    def _check_ratio(self, key: str, value: float) -> Tuple[bool, str]:
        """基于变化率的异常检测"""
        history = self._get_history(key)

        if len(history) < 1:
            history.append(value)
            return False, ""

        last_value = history[-1]
        if last_value > 1e-8:
            ratio = value / last_value
        else:
            ratio = 1.0

        is_anomaly = ratio > self.ratio_threshold
        reason = ""
        if is_anomaly:
            reason = f"{key}_ratio={ratio:.2f}>(threshold={self.ratio_threshold}),last={last_value:.4f}"

        # 总是更新历史
        if self.always_update_history:
            history.append(value)
        elif not is_anomaly:
            history.append(value)

        return is_anomaly, reason

    def _check_threshold(self, key: str, value: float) -> Tuple[bool, str]:
        """传统固定阈值检测"""
        threshold = self.max_thresholds.get(key)
        if threshold is None:
            return False, ""

        is_anomaly = value > threshold
        reason = ""
        if is_anomaly:
            reason = f"{key}={value:.4f}>(max={threshold})"

        return is_anomaly, reason

    def check(self, losses: Dict[str, float], sample_names: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """检查一组 loss 是否异常

        Args:
            losses: loss 名称到值的字典，如 {'loss_G': 1.5, 'loss_D': 0.8, ...}
            sample_names: 当前 batch 的样本名称列表（可选，用于记录异常样本）

        Returns:
            (should_skip, reasons): 是否应该跳过，以及异常原因列表
        """
        self._iter_count += 1
        self.total_checks += 1

        if self.mode == 'off':
            return False, []

        reasons = []
        should_skip = False

        for key, value in losses.items():
            if value is None:
                continue

            # 检查 NaN/Inf
            if self.skip_on_non_finite and (math.isnan(value) or math.isinf(value)):
                should_skip = True
                reasons.append(f"{key}_non_finite")
                continue

            # 根据模式选择检测方法
            if self.mode == 'zscore':
                is_anomaly, reason = self._check_zscore(key, value)
            elif self.mode == 'ratio':
                is_anomaly, reason = self._check_ratio(key, value)
            elif self.mode == 'threshold':
                is_anomaly, reason = self._check_threshold(key, value)
            else:
                is_anomaly, reason = False, ""

            if is_anomaly:
                should_skip = True
                if reason:
                    reasons.append(reason)

        # 如果有异常且提供了样本名称，添加到原因中
        if should_skip and sample_names:
            names_str = ", ".join(sample_names[:5])  # 最多显示5个
            if len(sample_names) > 5:
                names_str += f" ... (+{len(sample_names)-5} more)"
            reasons.append(f"samples=[{names_str}]")

        if should_skip:
            self.skip_count += 1

        return should_skip, reasons

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """获取当前统计信息"""
        stats = {}
        for key, history in self._history.items():
            if len(history) >= 2:
                mean, std = self._compute_stats(history)
                stats[key] = {
                    'mean': mean,
                    'std': std,
                    'count': len(history),
                    'min': min(history),
                    'max': max(history),
                }
        return stats

    def get_summary(self) -> str:
        """获取摘要信息"""
        skip_rate = self.skip_count / max(self.total_checks, 1) * 100
        return (
            f"LossGuard(mode={self.mode}, checks={self.total_checks}, "
            f"skips={self.skip_count}, skip_rate={skip_rate:.2f}%)"
        )

    @classmethod
    def from_config(cls, cfg: dict) -> 'LossGuard':
        """从配置字典创建 LossGuard

        Args:
            cfg: loss_guard 配置字典

        Example config:
            loss_guard:
              enabled: true
              mode: "zscore"
              window_size: 100
              warmup_iters: 50
              zscore_threshold: 4.0
              ratio_threshold: 3.0
              skip_on_non_finite: true
              # 兼容旧配置的固定阈值
              max_loss_g: 10
              max_loss_d: 10
        """
        if not cfg or not cfg.get('enabled', False):
            return cls(mode='off')

        mode = cfg.get('mode', 'zscore')

        # 提取固定阈值（兼容旧配置）
        max_thresholds = {}
        threshold_keys = ['max_loss_g', 'max_loss_d', 'max_loss_gan',
                          'max_loss_l1', 'max_loss_vgg', 'max_loss_feat',
                          'max_loss_edge', 'max_loss_laplacian']
        key_mapping = {
            'max_loss_g': 'loss_G',
            'max_loss_d': 'loss_D',
            'max_loss_gan': 'loss_GAN',
            'max_loss_l1': 'loss_L1',
            'max_loss_vgg': 'loss_vgg',
            'max_loss_feat': 'loss_feat',
            'max_loss_edge': 'loss_edge',
            'max_loss_laplacian': 'loss_laplacian',
        }
        for old_key in threshold_keys:
            if old_key in cfg and cfg[old_key] is not None:
                new_key = key_mapping.get(old_key, old_key)
                max_thresholds[new_key] = float(cfg[old_key])

        return cls(
            mode=mode,
            window_size=cfg.get('window_size', 100),
            warmup_iters=cfg.get('warmup_iters', 50),
            zscore_threshold=cfg.get('zscore_threshold', 4.0),
            ratio_threshold=cfg.get('ratio_threshold', 3.0),
            skip_on_non_finite=cfg.get('skip_on_non_finite', True),
            max_thresholds=max_thresholds,
            always_update_history=cfg.get('always_update_history', True),
        )
