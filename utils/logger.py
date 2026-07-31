"""日志和训练监控"""

import os
import json
import time
import threading
from collections import deque

import torch
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from .image import tensor2image


class TrainingDataStore:
    """训练数据存储，用于Web可视化

    Args:
        max_history: 最大历史数据点数。默认 500000 点，约占用 40MB 内存。
                    支持约 500 epoch × 1000 iters/epoch 的完整训练历史。
                    设置为 None 则无限制（内存占用会随训练持续增长）。
    """

    def __init__(self, max_history=500000):
        self.max_history = max_history
        self.loss_history = {
            'loss_2D': deque(maxlen=max_history),
            'loss_G': deque(maxlen=max_history),
            'loss_GAN': deque(maxlen=max_history),
            'loss_feat': deque(maxlen=max_history),
            'loss_vgg': deque(maxlen=max_history),
            'loss_L1': deque(maxlen=max_history),
            # 判别器监控
            'D_real': deque(maxlen=max_history),
            'D_fake': deque(maxlen=max_history),
        }
        self.val_history = {
            'eva_ssim': deque(maxlen=max_history),
            'eva_psnr': deque(maxlen=max_history),
        }
        self.steps = deque(maxlen=max_history)
        self.val_steps = deque(maxlen=max_history)
        self.latest_images = {
            'train_input': None,
            'train_target': None,
            'train_fake': None,
            # Optional: raw vs EMA comparison
            'train_fake_raw': None,
            'train_fake_ema': None,
            'val_input': None,
            'val_target': None,
            'val_fake': None,
            # Optional: raw vs EMA comparison
            'val_fake_raw': None,
            'val_fake_ema': None,
        }
        # 按 epoch 存储验证生成图历史（input/target 只保存一次，所以只存 fake）
        self.val_fake_by_epoch = {}  # {epoch: fake_img}
        # 事件历史（例如 loss_guard 触发时记录触发样本名）
        self.events = deque(maxlen=2000)
        self.latest_update_time = time.time()
        self.current_epoch = 0
        self.current_iter = 0
        self.total_step = 0
        self._lock = threading.Lock()
        # 验证图像目录（由TrainingManager设置）
        self.val_pic_dir = None

    def add_train_loss(self, step, losses: dict):
        """添加训练loss数据"""
        with self._lock:
            self.steps.append(step)
            for key in self.loss_history:
                val = losses.get(key)
                self.loss_history[key].append(float(val) if val is not None else None)
            self.latest_update_time = time.time()

    def add_val_metrics(self, step, metrics: dict):
        """添加验证指标"""
        with self._lock:
            self.val_steps.append(step)
            for key in self.val_history:
                val = metrics.get(key)
                self.val_history[key].append(float(val) if val is not None else None)

    def update_images(self, prefix, input_img, target_img, fake_img, fake_raw=None, fake_ema=None):
        """更新最新图像（numpy数组或PIL Image）

        Args:
            prefix: 'train' or 'val'
            fake_img: default fake (backward compatible)
            fake_raw/fake_ema: optional comparison variants
        """
        with self._lock:
            self.latest_images[f'{prefix}_input'] = input_img
            self.latest_images[f'{prefix}_target'] = target_img
            self.latest_images[f'{prefix}_fake'] = fake_img
            # Always set (including None) to avoid stale values across runs/config changes
            if f'{prefix}_fake_raw' in self.latest_images:
                self.latest_images[f'{prefix}_fake_raw'] = fake_raw
            if f'{prefix}_fake_ema' in self.latest_images:
                self.latest_images[f'{prefix}_fake_ema'] = fake_ema
            self.latest_update_time = time.time()

    def update_val_fake_for_epoch(self, epoch, fake_img):
        """存储指定 epoch 的验证生成图"""
        with self._lock:
            self.val_fake_by_epoch[epoch] = fake_img

    def add_event(self, event: dict):
        """记录事件（例如 loss_guard 跳过某个 batch）"""
        if not isinstance(event, dict):
            return
        with self._lock:
            self.events.append(event)
            self.latest_update_time = time.time()

    def get_events(self, last_n: int = 50):
        """获取最近事件"""
        with self._lock:
            evs = list(self.events)
        if last_n and last_n > 0:
            evs = evs[-int(last_n):]
        return {'events': evs}

    def get_val_epochs(self):
        """获取所有有验证图像的 epoch 列表"""
        with self._lock:
            return sorted(self.val_fake_by_epoch.keys())

    def get_val_images_for_epoch(self, epoch):
        """获取指定 epoch 的验证图像（fake 从历史获取，input/target 用最新的）"""
        with self._lock:
            fake_img = self.val_fake_by_epoch.get(epoch)
            if fake_img is None:
                return None
            return {
                'val_input': self.latest_images.get('val_input'),
                'val_target': self.latest_images.get('val_target'),
                'val_fake': fake_img,
            }

    def update_progress(self, epoch, iter_step, total_step):
        """更新训练进度"""
        with self._lock:
            self.current_epoch = epoch
            self.current_iter = iter_step
            self.total_step = total_step

    def get_loss_data(self, last_n=None):
        """获取loss历史数据"""
        with self._lock:
            steps = list(self.steps)
            data = {k: list(v) for k, v in self.loss_history.items()}
            if last_n and last_n < len(steps):
                steps = steps[-last_n:]
                data = {k: v[-last_n:] for k, v in data.items()}
            return {'steps': steps, 'losses': data}

    def get_val_data(self):
        """获取验证指标历史"""
        with self._lock:
            return {
                'steps': list(self.val_steps),
                'metrics': {k: list(v) for k, v in self.val_history.items()}
            }

    def get_latest_images(self):
        """获取最新图像"""
        with self._lock:
            return dict(self.latest_images)

    def get_status(self):
        """获取训练状态"""
        with self._lock:
            return {
                'epoch': self.current_epoch,
                'iter': self.current_iter,
                'total_step': self.total_step,
                'last_update': self.latest_update_time,
            }

    def set_val_pic_dir(self, val_pic_dir):
        """设置验证图像目录"""
        self.val_pic_dir = val_pic_dir

    def scan_val_images(self):
        """扫描验证图像目录，获取可用的epochs和samples
        返回: {'epochs': [epoch1, epoch2, ...], 'samples': [sample1, sample2, ...]}
        """
        import re
        import glob

        if not self.val_pic_dir or not os.path.isdir(self.val_pic_dir):
            return {'epochs': [], 'samples': []}

        epochs = set()
        samples = set()

        # 扫描 fake 图像文件（兼容 jpg/png，兼容 raw/ema 变体）:
        #   fake_ep{epoch}_{sample}.jpg
        #   fake_raw_ep{epoch}_{sample}.jpg
        #   fake_ema_ep{epoch}_{sample}.jpg
        for ext in ("jpg", "png"):
            fake_pattern = os.path.join(self.val_pic_dir, f'fake*_ep*_*.{ext}')
            for path in glob.glob(fake_pattern):
                basename = os.path.basename(path)
                match = re.match(r'fake(?:_(raw|ema))?_ep(\d+)_(.+)\.(jpg|png)$', basename)
                if match:
                    epochs.add(int(match.group(2)))
                    samples.add(match.group(3))

        return {
            'epochs': sorted(epochs),
            'samples': sorted(samples)
        }

    def load_val_image_from_disk(self, epoch, sample):
        """从磁盘加载指定epoch和sample的验证图像
        返回: {'val_input': np_array, 'val_target': np_array, 'val_fake': np_array}
        """
        if not self.val_pic_dir or not os.path.isdir(self.val_pic_dir):
            return None

        result = {}

        def _try_open(path: str):
            if not path or not os.path.exists(path):
                return None
            try:
                img = Image.open(path)
                return np.array(img)
            except Exception:
                return None

        # fake图像（默认/兼容）：fake_ep{epoch}_{sample}.(jpg|png)
        for ext in ("jpg", "png"):
            base = _try_open(os.path.join(self.val_pic_dir, f'fake_ep{epoch}_{sample}.{ext}'))
            if base is not None:
                result['val_fake'] = base
                break

        # fake图像（对比）：fake_raw_ep{epoch}_{sample}.(jpg|png), fake_ema_ep{epoch}_{sample}.(jpg|png)
        for ext in ("jpg", "png"):
            raw = _try_open(os.path.join(self.val_pic_dir, f'fake_raw_ep{epoch}_{sample}.{ext}'))
            if raw is not None:
                result['val_fake_raw'] = raw
                break
        for ext in ("jpg", "png"):
            ema = _try_open(os.path.join(self.val_pic_dir, f'fake_ema_ep{epoch}_{sample}.{ext}'))
            if ema is not None:
                result['val_fake_ema'] = ema
                break

        # 若没有默认 fake，则用 EMA/RAW 兜底
        if 'val_fake' not in result:
            if 'val_fake_ema' in result:
                result['val_fake'] = result['val_fake_ema']
            elif 'val_fake_raw' in result:
                result['val_fake'] = result['val_fake_raw']
        # 若缺失对比项，则用默认 fake 兜底（旧 checkpoint/旧图像目录兼容）
        if 'val_fake' in result:
            result.setdefault('val_fake_raw', result['val_fake'])
            result.setdefault('val_fake_ema', result['val_fake'])

        # input图像: 优先 input_{sample}.* 其次 input_ep{epoch}_{sample}.*
        input_img = None
        for ext in ("jpg", "png"):
            input_img = _try_open(os.path.join(self.val_pic_dir, f'input_{sample}.{ext}'))
            if input_img is not None:
                break
        if input_img is None:
            for ext in ("jpg", "png"):
                input_img = _try_open(os.path.join(self.val_pic_dir, f'input_ep{epoch}_{sample}.{ext}'))
                if input_img is not None:
                    break
        if input_img is not None:
            result['val_input'] = input_img

        # target图像: 优先 real_{sample}.* 其次 real_ep{epoch}_{sample}.*
        target_img = None
        for ext in ("jpg", "png"):
            target_img = _try_open(os.path.join(self.val_pic_dir, f'real_{sample}.{ext}'))
            if target_img is not None:
                break
        if target_img is None:
            for ext in ("jpg", "png"):
                target_img = _try_open(os.path.join(self.val_pic_dir, f'real_ep{epoch}_{sample}.{ext}'))
                if target_img is not None:
                    break
        if target_img is not None:
            result['val_target'] = target_img

        return result if result else None

    def load_latest_val_image_for_sample(self, sample):
        """加载指定sample的最新验证图像（最大epoch）"""
        scan_result = self.scan_val_images()
        epochs = scan_result.get('epochs', [])
        if not epochs:
            return None
        # 取最大epoch
        latest_epoch = max(epochs)
        return self.load_val_image_from_disk(latest_epoch, sample)


# 全局数据存储实例
_data_store = None


def get_data_store():
    """获取全局数据存储实例"""
    global _data_store
    if _data_store is None:
        _data_store = TrainingDataStore()
    return _data_store


class TrainingManager:
    """训练过程管理器 - 处理日志、保存模型和图像"""

    def __init__(
        self,
        summary_log_dir,
        temp_pic_dir,
        val_pic_dir,
        model_save_dir,
        checkpoint_save_every_epochs=10,
        train_image_save_every_iters=10,
        val_image_save_every_n=10,
        val_image_save_until_epoch=40,
        val_save_input_real_once: bool = False,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled)
        self.writer = None
        if self.enabled:
            from .config import make_dirs
            make_dirs(summary_log_dir, temp_pic_dir, val_pic_dir, model_save_dir)
            self.writer = SummaryWriter(summary_log_dir)
        self.temp_pic_dir = temp_pic_dir
        self.val_pic_dir = val_pic_dir
        self.model_save_dir = model_save_dir

        self.checkpoint_save_every_epochs = int(checkpoint_save_every_epochs) if checkpoint_save_every_epochs else 0
        self.train_image_save_every_iters = int(train_image_save_every_iters) if train_image_save_every_iters else 0
        self.val_image_save_every_n = int(val_image_save_every_n) if val_image_save_every_n else 0
        self.val_image_save_until_epoch = int(val_image_save_until_epoch) if val_image_save_until_epoch else 0
        self.val_save_input_real_once = bool(val_save_input_real_once)

        # 获取全局数据存储用于Web可视化
        self.data_store = get_data_store()
        # 设置验证图像目录供Web可视化使用
        self.data_store.set_val_pic_dir(val_pic_dir)

    def log_scalars(self, scalars: dict, step: int, prefix: str):
        """记录标量到tensorboard"""
        if not self.enabled or self.writer is None:
            return
        for k, v in scalars.items():
            if v is None:
                continue
            try:
                self.writer.add_scalar(f"{prefix}/{k}", float(v), step)
            except Exception:
                continue

    def _to_pil_image(self, image_tensor, inference=False):
        image_array = tensor2image(image_tensor, inference=inference)
        if image_array.ndim == 3:
            if image_array.shape[2] == 1:
                image_array = image_array[:, :, 0]
            elif image_array.shape[2] >= 2:
                # 2.5D/多通道可视化：仅保存中间通道（第二通道）
                image_array = image_array[:, :, image_array.shape[2] // 2]
        image = Image.fromarray(image_array)
        return image

    def _array_to_pil(self, image_array):
        """将numpy数组转换为PIL Image"""
        if image_array is None:
            return None
        if image_array.ndim == 3:
            if image_array.shape[2] == 1:
                image_array = image_array[:, :, 0]
            elif image_array.shape[2] >= 2:
                # 2.5D/多通道可视化：仅保存中间通道（第二通道）
                image_array = image_array[:, :, image_array.shape[2] // 2]
        return Image.fromarray(image_array)

    def save_image(self, image_tensor, tag, save_dir, inference=False):
        """保存tensor为图像"""
        image = self._to_pil_image(image_tensor, inference=inference)
        image.save(os.path.join(save_dir, tag + ".jpg"), format="JPEG")

    def save_image_from_array(self, image_array, tag, save_dir):
        """从numpy数组保存图像"""
        image = self._array_to_pil(image_array)
        if image is not None:
            image.save(os.path.join(save_dir, tag + ".jpg"), format="JPEG")

    def save_image_if_missing(self, image_tensor, tag, save_dir, inference=False):
        """若文件不存在则保存（用于val input/real只保存一次）"""
        path = os.path.join(save_dir, tag + ".jpg")
        if os.path.exists(path):
            return
        self.save_image(image_tensor, tag, save_dir, inference=inference)

    def save_image_from_array_if_missing(self, image_array, tag, save_dir):
        """从numpy数组保存图像（若不存在）"""
        path = os.path.join(save_dir, tag + ".jpg")
        if os.path.exists(path):
            return
        self.save_image_from_array(image_array, tag, save_dir)

    @staticmethod
    def _safe_stem(sample_name: str) -> str:
        """Make a stable, filesystem-safe stem from dataset sample name."""
        if not sample_name:
            return ""
        import os
        stem = os.path.splitext(os.path.basename(str(sample_name)))[0]
        # keep it readable but safe
        stem = stem.replace(" ", "_").replace("/", "_").replace("\\", "_")
        return stem[:120]

    def save_latest_checkpoint(
        self,
        discriminator,
        generator,
        D_optim,
        G_optim,
        scheduler_D,
        scheduler_G,
        epoch_next: int,
        iter_step: int,
        L1_weight=None,
        generator_ema=None,
        ema_meta=None,
    ):
        """保存可精确续跑的 latest checkpoint（每个 epoch 都保存）"""
        ckpt_D = {
            '2D_state_dict': discriminator.state_dict(),
            'optimizer_2D_state': D_optim.state_dict(),
            'scheduler_2D_state': scheduler_D.state_dict() if scheduler_D is not None else None,
            'epoch': int(epoch_next),
            'epoch_is_next': True,
            'iter': int(iter_step),
        }
        ckpt_G = {
            'G_state_dict': generator.state_dict(),
            'G_ema_state_dict': generator_ema.state_dict() if generator_ema is not None else None,
            'optimizer_G_state': G_optim.state_dict(),
            'scheduler_G_state': scheduler_G.state_dict() if scheduler_G is not None else None,
            'L1_weight': L1_weight,
            'ema': ema_meta,
            'epoch': int(epoch_next),
            'epoch_is_next': True,
            'iter': int(iter_step),
        }
        torch.save(ckpt_D, os.path.join(self.model_save_dir, "net_2D_latest.pth"))
        torch.save(ckpt_G, os.path.join(self.model_save_dir, "net_G_latest.pth"))

    def save_periodic_checkpoint(
        self,
        discriminator,
        generator,
        D_optim,
        G_optim,
        scheduler_D,
        scheduler_G,
        epoch_next: int,
        iter_step: int,
        L1_weight=None,
        generator_ema=None,
        ema_meta=None,
    ):
        """按配置周期保存 checkpoint（用于回溯）"""
        if self.checkpoint_save_every_epochs <= 0:
            return
        if epoch_next % self.checkpoint_save_every_epochs != 0:
            return

        ckpt_D = {
            '2D_state_dict': discriminator.state_dict(),
            'optimizer_2D_state': D_optim.state_dict(),
            'scheduler_2D_state': scheduler_D.state_dict() if scheduler_D is not None else None,
            'epoch': int(epoch_next),
            'epoch_is_next': True,
            'iter': int(iter_step),
        }
        ckpt_G = {
            'G_state_dict': generator.state_dict(),
            'G_ema_state_dict': generator_ema.state_dict() if generator_ema is not None else None,
            'optimizer_G_state': G_optim.state_dict(),
            'scheduler_G_state': scheduler_G.state_dict() if scheduler_G is not None else None,
            'L1_weight': L1_weight,
            'ema': ema_meta,
            'epoch': int(epoch_next),
            'epoch_is_next': True,
            'iter': int(iter_step),
        }
        torch.save(ckpt_D, os.path.join(self.model_save_dir, f"net_2D_ep{epoch_next}.pth"))
        torch.save(ckpt_G, os.path.join(self.model_save_dir, f"net_G_ep{epoch_next}.pth"))

    def __call__(self, package, save_image=False):
        """主要调用接口"""
        if not self.enabled:
            return
        event = package.get("event") if isinstance(package, dict) else None
        if isinstance(event, dict):
            self.data_store.add_event(event)
        step = int(package.get("current_step", 0))
        epoch = int(package.get("Epoch", 0))
        total_step = int(package.get("total_step", 0))

        # 更新进度
        self.data_store.update_progress(epoch, step, total_step)

        if package.get("inference"):
            scalars = {k: package[k] for k in ("eva_ssim", "eva_psnr") if k in package}
            self.log_scalars(scalars, step, prefix="val")
            # 更新验证指标到数据存储
            self.data_store.add_val_metrics(step, scalars)
        else:
            # 训练损失
            scalars = {k: package[k] for k in ("loss_2D", "loss_G", "loss_GAN", "loss_feat", "loss_vgg", "loss_L1") if k in package}
            self.log_scalars(scalars, step, prefix="train")

            # 判别器监控分数 - 单独记录到 discriminator 组
            d_scalars = {k: package[k] for k in ("D_real", "D_fake") if k in package}
            if d_scalars:
                self.log_scalars(d_scalars, step, prefix="discriminator")

            # 更新训练loss到数据存储（包括判别器分数）
            scalars.update(d_scalars)
            self.data_store.add_train_loss(step, scalars)

        # 训练图像保存和Web更新
        if save_image and not package.get("inference"):
            input_img = tensor2image(package["input_tensor"][0], inference=False)
            target_img = tensor2image(package["target_tensor"][0], inference=False)
            fake_default_tensor = package.get("fake_tensor")
            fake_raw_tensor = package.get("fake_tensor_raw")
            if fake_raw_tensor is None:
                fake_raw_tensor = fake_default_tensor
            fake_ema_tensor = package.get("fake_tensor_ema")

            fake_img = tensor2image(fake_default_tensor[0], inference=False) if fake_default_tensor is not None else None
            fake_raw_img = tensor2image(fake_raw_tensor[0], inference=False) if fake_raw_tensor is not None else None
            fake_ema_img = tensor2image(fake_ema_tensor[0], inference=False) if fake_ema_tensor is not None else None

            self.save_image_from_array(input_img, f"input_image_ep{epoch}_it{step}", self.temp_pic_dir)
            self.save_image_from_array(target_img, f"real_image_ep{epoch}_it{step}", self.temp_pic_dir)
            if fake_img is not None:
                self.save_image_from_array(fake_img, f"fake_image_ep{epoch}_it{step}", self.temp_pic_dir)

            # 更新Web可视化数据
            self.data_store.update_images('train', input_img, target_img, fake_img, fake_raw=fake_raw_img, fake_ema=fake_ema_img)

        # 验证图像保存
        if package.get("inference"):
            index = int(package.get("index", 0))
            if self.val_image_save_every_n > 0 and index % self.val_image_save_every_n == 0:
                # 优先使用预转换的图像
                input_img = package.get('input_image')
                target_img = package.get('target_image')
                fake_img = package.get('fake_image')
                fake_raw_img = package.get('fake_image_raw')
                fake_ema_img = package.get('fake_image_ema')

                # 如果没有预转换，则从tensor转换
                if input_img is None and "input_tensor" in package:
                    input_img = tensor2image(package["input_tensor"], inference=True)
                if target_img is None and "target_tensor" in package:
                    target_img = tensor2image(package["target_tensor"], inference=True)
                if fake_img is None and "fake_tensor" in package:
                    fake_img = tensor2image(package["fake_tensor"], inference=True)

                sample_stem = self._safe_stem(package.get("sample_name")) if isinstance(package.get("sample_name"), str) else ""
                tag_suffix = sample_stem if sample_stem else str(index).zfill(4)

                if self.val_save_input_real_once:
                    if target_img is not None:
                        self.save_image_from_array_if_missing(target_img, f"real_{tag_suffix}", self.val_pic_dir)
                    if input_img is not None:
                        self.save_image_from_array_if_missing(input_img, f"input_{tag_suffix}", self.val_pic_dir)
                else:
                    if self.val_image_save_until_epoch <= 0 or epoch < self.val_image_save_until_epoch:
                        if target_img is not None:
                            self.save_image_from_array(target_img, f"real_ep{epoch}_{tag_suffix}", self.val_pic_dir)
                        if input_img is not None:
                            self.save_image_from_array(input_img, f"input_ep{epoch}_{tag_suffix}", self.val_pic_dir)
                if fake_img is not None:
                    self.save_image_from_array(fake_img, f"fake_ep{epoch}_{tag_suffix}", self.val_pic_dir)
                # 可选：同时保存 raw/ema 的 fake 便于对比
                if fake_raw_img is not None:
                    self.save_image_from_array(fake_raw_img, f"fake_raw_ep{epoch}_{tag_suffix}", self.val_pic_dir)
                if fake_ema_img is not None:
                    self.save_image_from_array(fake_ema_img, f"fake_ema_ep{epoch}_{tag_suffix}", self.val_pic_dir)

                # 更新Web可视化数据（只更新第一个验证样本）
                if index == 0:
                    self.data_store.update_images('val', input_img, target_img, fake_img, fake_raw=fake_raw_img, fake_ema=fake_ema_img)
                    # 同时存储到 epoch 历史，便于后续回溯
                    self.data_store.update_val_fake_for_epoch(epoch, fake_img)


def log2txt(log_path, content):
    """写入日志到文本文件"""
    with open(log_path, "a") as f:
        f.write(content + "\n")


def save_checkpoint(tag, package, save_path):
    """保存模型checkpoint"""
    checkpoints = {}
    for key, value in package.items():
        if tag in key:
            checkpoints[key] = value
            checkpoints["epoch"] = package["Epoch"]
            checkpoints["iter"] = package["current_step"]
            checkpoints["L1_weight"] = package.get("L1_weight")
    torch.save(checkpoints, save_path)
