"""训练引擎"""

import os
import time
from contextlib import nullcontext
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import build_generator, build_discriminator
from losses import LossModule, LossScheduler, LOSS_SCHEDULE_TEMPLATES
from data import create_dataloader
from utils import (
    set_random_seed, set_save_dirs, log2txt,
    TrainingManager, tensor2image, get_ssim, get_psnr,
    ModelEMA, LossGuard,
)
from utils.distributed import (
    setup_distributed, cleanup_distributed,
    is_main_process, get_rank, get_world_size, get_local_rank,
    wrap_model_ddp, unwrap_model, save_on_main_only, print_on_main_only,
    DistributedSamplerWrapper
)


class Trainer:
    """GAN训练器

    支持两种训练模式:
    1. 单卡训练: 默认模式
    2. DDP 多卡训练: 通过 --gpus 参数或配置文件指定多个 GPU
    """

    def __init__(self, config):
        self.cfg = config
        self._setup_distributed()  # 必须最先初始化分布式环境
        self._setup_paths()
        self._setup_devices()
        self._setup_models()
        self._setup_ema()
        self._setup_optimizers()
        self._setup_loss_scheduler()
        self._setup_loss()
        self._setup_data()
        self._setup_manager()
        self._setup_loss_guard()

    def _setup_distributed(self):
        """初始化分布式训练环境"""
        # 检查是否在 DDP 环境中（由 train.py 的 mp.spawn 启动）
        if 'RANK' in os.environ:
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            self.is_ddp = True
            self.is_main = (int(os.environ.get('RANK', 0)) == 0)
        else:
            self.local_rank = 0
            self.world_size = 1
            self.is_ddp = False
            self.is_main = True

    def _setup_paths(self):
        """设置路径"""
        self.project_name = self.cfg['project_name']
        self.base = self.cfg['base']
        self.data_root = self.cfg['data_root']
        self.val_data_root = self.cfg['val_data_root']

        log_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # DDP 模式下在日志文件名中添加 rank 以避免冲突
        if self.is_ddp and not self.is_main:
            log_name = f"{log_name}_rank{os.environ.get('RANK', 0)}"
        self.log_path = os.path.join(self.base, "log", self.project_name, f"{log_name}.txt")

        (self.model_save_path, self.draw_path, self.summary_log_dir,
         self.temp_pic_dir, self.val_pic_dir) = set_save_dirs(self.base, self.project_name)

        # Debug模式
        self.debug = self.project_name == "debug"
        if self.debug:
            self.cfg['training']['print_freq'] = 5
            self.cfg['training']['val_freq'] = 1
            self.cfg['training']['total_epoch'] = 2

    def _setup_devices(self):
        """设置设备"""
        # DDP 模式下使用 local_rank 对应的 GPU
        if self.is_ddp:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            multi_gpu_cfg = self.cfg.get('multi_gpu', {})
            use_gpus = multi_gpu_cfg.get('use_gpus', [0])
            gpu_id = use_gpus[0] if use_gpus else 0
            self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else 'cpu')

    def _setup_models(self):
        """初始化模型"""
        set_random_seed(self.cfg['training']['random_seed'])

        self.image_channels = self.cfg.get('image_channels', 1)
        self.generator = build_generator(
            self.cfg['g_name'],
            self.device,
            in_channels=self.image_channels,
            out_channels=self.image_channels
        )
        self.discriminator = build_discriminator(
            self.cfg['d_name'],
            self.device,
            input_nc=self.image_channels * 2
        )

        # DDP 模式下包装模型
        if self.is_ddp:
            self.generator = wrap_model_ddp(self.generator, find_unused_parameters=False)
            self.discriminator = wrap_model_ddp(self.discriminator, find_unused_parameters=False)
            if self.is_main:
                print(f"Using DDP with {self.world_size} GPUs")

    def _setup_ema(self):
        """设置生成器 EMA（可选）"""
        ema_cfg = (self.cfg.get('training', {}) or {}).get('ema', {}) or {}
        self.ema_enabled = bool(ema_cfg.get('enabled', False))
        self.ema_use_for_val = bool(ema_cfg.get('use_for_val', True))
        self.ema_use_for_train_vis = bool(ema_cfg.get('use_for_train_vis', False))
        self.model_ema = None
        self.generator_ema = None

        if not self.ema_enabled:
            return

        decay = float(ema_cfg.get('decay', 0.999))
        update_every = int(ema_cfg.get('update_every', 1))
        update_buffers = bool(ema_cfg.get('update_buffers', True))

        gen_module = unwrap_model(self.generator)
        self.model_ema = ModelEMA(
            gen_module,
            decay=decay,
            update_every=update_every,
            device=self.device,
            update_buffers=update_buffers,
        )
        self.generator_ema = self.model_ema.ema

        if self.is_main:
            print(
                "Generator EMA enabled: "
                f"decay={self.model_ema.decay}, "
                f"update_every={self.model_ema.update_every}, "
                f"use_for_val={self.ema_use_for_val}, "
                f"use_for_train_vis={self.ema_use_for_train_vis}"
            )

    def _setup_optimizers(self):
        """设置优化器和调度器"""
        lr = self.cfg['training']['lr_ini']
        beta1 = self.cfg['training'].get('beta1', 0.5)
        beta2 = self.cfg['training'].get('beta2', 0.999)

        self.G_optim = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.D_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

        total_epoch = self.cfg['training']['total_epoch']
        start_epoch = self.cfg['training']['start_epoch']
        self.scheduler_G = CosineAnnealingLR(self.G_optim, T_max=total_epoch,
                                              last_epoch=start_epoch if start_epoch != 0 else -1)
        self.scheduler_D = CosineAnnealingLR(self.D_optim, T_max=total_epoch,
                                              last_epoch=start_epoch if start_epoch != 0 else -1)

        self.start_epoch = start_epoch

        # === 交替训练控制参数（优化：更清晰的命名和向后兼容） ===
        training_cfg = self.cfg['training']

        # 新参数（推荐）
        self.d_update_interval = int(training_cfg.get('d_update_interval', 1))
        self.g_update_interval = int(training_cfg.get('g_update_interval', 1))

        # 向后兼容：旧参数 discriminator_train_ratio
        # 旧含义：discriminator_train_ratio=N 表示每N次迭代更新1次D
        # 转换为新参数：g_update_interval=N, d_update_interval=1
        if 'discriminator_train_ratio' in training_cfg:
            old_ratio = int(training_cfg['discriminator_train_ratio'])
            if 'd_update_interval' not in training_cfg and 'g_update_interval' not in training_cfg:
                # 只有旧参数，进行转换
                self.g_update_interval = old_ratio
                self.d_update_interval = 1
                if self.is_main:
                    print(f"[Warning] 'discriminator_train_ratio' is deprecated. "
                          f"Converted to: d_update_interval=1, g_update_interval={old_ratio}")
                    print(f"  Recommended: use 'd_update_interval' and 'g_update_interval' in config")

        if self.is_main:
            print(f"Training schedule: D updates every {self.d_update_interval} iters, "
                  f"G updates every {self.g_update_interval} iters")

    def _setup_loss_scheduler(self):
        """设置损失权重调度器"""
        total_epoch = self.cfg['training']['total_epoch']

        if 'loss_schedule' not in self.cfg:
            raise ValueError(
                "Configuration must include 'loss_schedule' section. "
                "Please define loss weights in your config file.\n"
                "Example:\n"
                "loss_schedule:\n"
                "  L1:\n"
                "    type: 'constant'\n"
                "    value: 100\n"
                "  GAN:\n"
                "    type: 'linear'\n"
                "    start: 0.1\n"
                "    end: 1.0\n"
                "  ...\n"
                f"Or use a template: {list(LOSS_SCHEDULE_TEMPLATES.keys())}"
            )

        self.loss_scheduler = LossScheduler(total_epoch, self.cfg['loss_schedule'])
        print(self.loss_scheduler.get_summary())

    def _setup_loss(self):
        """设置损失模块"""
        vgg_cfg = self.cfg.get('vgg', {}) or {}
        # DDP 模式下使用 local_rank 作为设备
        use_gpus = [self.local_rank] if self.is_ddp else self.cfg.get('multi_gpu', {}).get('use_gpus', [0])
        self.loss_module = LossModule(
            use_gpus=use_gpus,
            vgg_use_imagenet_norm=bool(vgg_cfg.get('use_imagenet_norm', False)),
            vgg_clamp_to_01=bool(vgg_cfg.get('clamp_to_01', True)),
            vgg_weights=vgg_cfg.get('weights', None),
        )

    def _setup_data(self):
        """设置数据加载器"""
        global_batch_size = int(self.cfg['training']['batch_size'])
        if self.is_ddp:
            if global_batch_size % self.world_size != 0:
                raise ValueError(
                    f"DDP 模式下 training.batch_size 视为全局 batch_size，但当前 batch_size={global_batch_size} "
                    f"不能被 world_size={self.world_size} 整除。"
                    f"请改成可整除的值（例如 {self.world_size * (global_batch_size // self.world_size)} 或 "
                    f"{self.world_size * (global_batch_size // self.world_size + 1)}），"
                    f"或切换为单卡/减少 GPU 数。"
                )
            batch_size = global_batch_size // self.world_size
            if self.is_main:
                print(f"DDP effective batch_size: global={global_batch_size}, per_gpu={batch_size}, world_size={self.world_size}")
        else:
            batch_size = global_batch_size
        source_dirname = self.cfg.get('source_dir', 'source')
        cta_dirname = self.cfg.get('cta_dir', 'CTA')
        train_cache_dir = self.cfg.get('cache_dir')
        val_cache_dir = self.cfg.get('val_cache_dir')
        aug_cfg = self.cfg.get('augmentation', {}) or {}

        self.train_loader, self.train_sampler = create_dataloader(
            self.data_root,
            batch_size=batch_size,
            shuffle=not self.is_ddp,  # DDP 模式下由 sampler 控制 shuffle
            num_workers=min(batch_size + 2, 8),  # 限制最大 workers 数
            source_dirname=source_dirname,
            cta_dirname=cta_dirname,
            cache_dir=train_cache_dir,
            random_rotate=bool(aug_cfg.get('random_rotate', False)),
            rotate_degrees=float(aug_cfg.get('rotate_degrees', 0.0)),
            rotate_prob=float(aug_cfg.get('rotate_prob', 0.0)),
            distributed=self.is_ddp,
            return_sampler=True,
        )
        self.val_loader, self.val_sampler = create_dataloader(
            self.val_data_root,
            batch_size=batch_size,
            shuffle=False,
            source_dirname=source_dirname,
            cta_dirname=cta_dirname,
            cache_dir=val_cache_dir,
            distributed=self.is_ddp,  # DDP 下分布式验证以加速
            return_sampler=True,
            drop_last=False,
        )

    def _setup_manager(self):
        """设置训练管理器"""
        ckpt_cfg = self.cfg.get('checkpoint', {}) or {}
        vis_cfg = self.cfg.get('visualization', {}) or {}
        self.train_image_save_every_iters = int(vis_cfg.get('train_every_iters', 10))
        self.manager = TrainingManager(
            self.summary_log_dir,
            self.temp_pic_dir,
            self.val_pic_dir,
            self.model_save_path,
            checkpoint_save_every_epochs=int(ckpt_cfg.get('save_every_epochs', 10)),
            train_image_save_every_iters=self.train_image_save_every_iters,
            val_image_save_every_n=int(vis_cfg.get('val_every_n', 10)),
            val_image_save_until_epoch=int(vis_cfg.get('val_until_epoch', 40)),
            val_save_input_real_once=bool(vis_cfg.get('val_save_input_real_once', False)),
            enabled=self.is_main,  # DDP 下只让 rank0 写日志/写盘/推送可视化
        )

    def _setup_loss_guard(self):
        """设置 loss 异常检测器"""
        guard_cfg = (self.cfg.get('training', {}) or {}).get('loss_guard', {}) or {}
        self.loss_guard = LossGuard.from_config(guard_cfg)
        if self.is_main and self.loss_guard.mode != 'off':
            print(f"LossGuard enabled: mode={self.loss_guard.mode}, "
                  f"window={self.loss_guard.window_size}, warmup={self.loss_guard.warmup_iters}")

        # 优化：预创建DDP同步用的flag tensor，避免每次迭代创建（减少显存碎片）
        if self.is_ddp:
            self.ddp_sync_flag = torch.zeros(1, device=self.device, dtype=torch.int32)

    @staticmethod
    def _set_requires_grad(model, flag: bool):
        """Enable/disable gradients for a model (e.g., freeze D during G update)."""
        for p in model.parameters():
            p.requires_grad_(flag)

    def load_checkpoint(self, checkpoint_dir=None):
        """加载checkpoint"""
        if checkpoint_dir is None:
            checkpoint_dir = self.model_save_path

        if self.is_main:
            print("Loading checkpoints...")

        checkpoint_D = torch.load(os.path.join(checkpoint_dir, 'net_2D_latest.pth'), map_location=self.device)
        checkpoint_G = torch.load(os.path.join(checkpoint_dir, 'net_G_latest.pth'), map_location=self.device)

        # 获取原始模型（从 DDP/DP 包装中提取）用于加载权重
        disc_module = unwrap_model(self.discriminator)
        gen_module = unwrap_model(self.generator)

        # 处理 state_dict 键名（移除可能的 'module.' 前缀）
        def clean_state_dict(state_dict):
            """移除 state_dict 中的 'module.' 前缀（如果有）"""
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            return new_state_dict

        disc_module.load_state_dict(clean_state_dict(checkpoint_D['2D_state_dict']))
        gen_module.load_state_dict(clean_state_dict(checkpoint_G['G_state_dict']))

        # Restore generator EMA if enabled.
        if self.model_ema is not None and self.generator_ema is not None:
            ema_meta = checkpoint_G.get('ema')
            ema_state = checkpoint_G.get('G_ema_state_dict')

            if isinstance(ema_meta, dict):
                prev = self.model_ema.get_meta()
                self.model_ema.load_meta(ema_meta)
                if self.is_main:
                    if (
                        float(prev.get('decay', self.model_ema.decay)) != float(self.model_ema.decay)
                        or int(prev.get('update_every', self.model_ema.update_every)) != int(self.model_ema.update_every)
                    ):
                        print(
                            "Loaded EMA meta from checkpoint: "
                            f"decay {prev.get('decay')} -> {self.model_ema.decay}, "
                            f"update_every {prev.get('update_every')} -> {self.model_ema.update_every}"
                        )

            if ema_state is not None:
                self.generator_ema.load_state_dict(clean_state_dict(ema_state))
                if self.is_main:
                    print(f"Loaded generator EMA weights from checkpoint (num_updates={self.model_ema.num_updates})")
            else:
                # Backward compatible: checkpoint may not include EMA, initialize from current generator.
                self.generator_ema.load_state_dict(gen_module.state_dict())
                if self.is_main:
                    print("No EMA weights found in checkpoint; initialized EMA from current generator weights.")

        self.D_optim.load_state_dict(checkpoint_D['optimizer_2D_state'])
        self.G_optim.load_state_dict(checkpoint_G['optimizer_G_state'])

        if checkpoint_D.get('scheduler_2D_state'):
            self.scheduler_D.load_state_dict(checkpoint_D['scheduler_2D_state'])
        if checkpoint_G.get('scheduler_G_state'):
            self.scheduler_G.load_state_dict(checkpoint_G['scheduler_G_state'])

        epoch_val = int(checkpoint_D['epoch'])
        if not checkpoint_D.get('epoch_is_next', False):
            epoch_val += 1
        self.start_epoch = epoch_val
        self.iter = checkpoint_D['iter']

        if self.is_main:
            print(f"Resumed from epoch {self.start_epoch}, iter {self.iter}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.generator.train()
        self.discriminator.train()

        # DDP 模式下设置 sampler 的 epoch 以确保每个 epoch 的 shuffle 不同
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        clip_gradient_norm = self.cfg['training']['clip_gradient_norm']
        print_freq = self.cfg['training']['print_freq']
        # 注意：d_update_interval 和 g_update_interval 已在 _setup_optimizers 中设置

        # 获取当前epoch的所有损失权重
        current_weights = self.loss_scheduler.get_all_weights(epoch)

        # 每个epoch开始时打印当前权重（只在主进程打印）
        weights_str = ", ".join([f"{k}={v:.3f}" for k, v in current_weights.items() if v > 0])
        if self.is_main:
            print(f"  Loss weights: {weights_str}")
            log2txt(self.log_path, f"epoch_{epoch} weights: {weights_str}")

        # DDP 模式下只在主进程显示进度条
        loader_iter = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                          desc=f"epoch_{str(epoch).zfill(3)}", disable=not self.is_main)

        for i, input_dic in loader_iter:
            self.iter += 1

            # ============================================================
            # 标准 GAN 训练流程：先更新 D，再更新 G（单次 G forward 复用）
            # ============================================================
            #
            # 重构改进：
            # 1. 只做一次 G forward，避免重复前向导致 BN 统计漂移（DDP 下尤其明显）
            # 2. 更新 D 时：使用 fake.detach()，不影响 G 的梯度
            # 3. 更新 G 时：冻结 D 参数，D 作为 fixed network 提供梯度信号
            # 4. 不再需要 retain_graph=True，避免显存泄漏和 inplace operation 错误
            # 5. 符合标准 GAN 理论：判别器先学习区分，生成器再基于反馈改进
            # ============================================================

            # 更新频率控制
            update_d = (self.iter % self.d_update_interval == 0)
            update_g = (self.iter % self.g_update_interval == 0)

            # 将输入移动到设备
            input_2D = input_dic['source'].to(self.device)
            target_2D = input_dic['cta'].to(self.device)

            # 仅做一次 G forward（若本轮不更新 G，则不建立计算图）
            if update_g:
                fake_2D = self.generator(input_2D)
            else:
                with torch.no_grad():
                    fake_2D = self.generator(input_2D)

            # === Step 1: 更新判别器 D ===
            if update_d:
                # D loss（fake 内部会 detach）
                loss_D, pred_real, pred_fake = self.loss_module.compute_discriminator_loss(
                    self.discriminator, input_2D, target_2D, fake_2D)
                d_scores = self.loss_module.get_discriminator_scores(pred_real, pred_fake)

                # 智能异常 loss 检测（D 阶段）
                if self.loss_guard.mode != 'off':
                    with torch.no_grad():
                        losses_to_check = {'loss_D': float(loss_D.item())}
                        sample_names = input_dic.get('name', None)
                        if sample_names is not None:
                            if isinstance(sample_names, (list, tuple)):
                                sample_names = list(sample_names)
                            else:
                                sample_names = [sample_names]

                        local_skip, reasons = self.loss_guard.check(losses_to_check, sample_names=sample_names)

                        # DDP 下同步决策
                        global_skip = local_skip
                        if self.is_ddp:
                            import torch.distributed as dist
                            self.ddp_sync_flag[0] = 1 if local_skip else 0
                            dist.all_reduce(self.ddp_sync_flag, op=dist.ReduceOp.MAX)
                            global_skip = bool(int(self.ddp_sync_flag[0].item()) > 0)

                        if global_skip:
                            if self.is_main:
                                reason_str = ",".join(reasons) if reasons else "triggered_on_other_rank"
                                msg = f"[LossGuard] skip iter={self.iter} epoch={epoch} (D phase) reasons={reason_str} loss_D={losses_to_check['loss_D']:.6f}"
                                print(msg)
                                log2txt(self.log_path, msg)
                            continue  # 跳过整个 iteration

                # Backward and optimize D
                self.D_optim.zero_grad()
                loss_D.backward()
                if clip_gradient_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                                  clip_gradient_norm, norm_type='inf')
                self.D_optim.step()
            else:
                # 不更新 D 时，仍需获取 d_scores 用于监控
                with torch.no_grad():
                    _, pred_real, pred_fake = self.loss_module.compute_discriminator_loss(
                        self.discriminator, input_2D, target_2D, fake_2D)
                    d_scores = self.loss_module.get_discriminator_scores(pred_real, pred_fake)
                loss_D = torch.tensor(0.0, device=self.device)  # 占位

            # === Step 2: 更新生成器 G ===
            if update_g:
                # 单卡时冻结 D 参数；DDP 下用 no_sync 避免无用 all-reduce
                freeze_d = not self.is_ddp
                if freeze_d:
                    self._set_requires_grad(self.discriminator, False)

                # 计算 pred_real（用于 feature matching loss）
                pred_real = None
                if current_weights.get('feat', 0) > 0:
                    with torch.no_grad():
                        pred_real = self.discriminator(torch.cat((input_2D, target_2D), dim=1))

                # Forward pass for G（fake 不 detach）
                loss_G, loss_dict = self.loss_module.compute_generator_loss(
                    self.discriminator, input_2D, target_2D, fake_2D,
                    pred_real, weights=current_weights)

                # 智能异常 loss 检测（G 阶段）
                if self.loss_guard.mode != 'off':
                    with torch.no_grad():
                        losses_to_check = {
                            'loss_G': float(loss_G.item()),
                            'loss_GAN': float(loss_dict.get('GAN', torch.tensor(0.0)).item()),
                            'loss_feat': float(loss_dict.get('feat', torch.tensor(0.0)).item()),
                            'loss_vgg': float(loss_dict.get('VGG', torch.tensor(0.0)).item()),
                            'loss_L1': float(loss_dict.get('L1', torch.tensor(0.0)).item()),
                        }
                        sample_names = input_dic.get('name', None)
                        if sample_names is not None:
                            if isinstance(sample_names, (list, tuple)):
                                sample_names = list(sample_names)
                            else:
                                sample_names = [sample_names]

                        local_skip, reasons = self.loss_guard.check(losses_to_check, sample_names=sample_names)

                        # DDP 下同步决策
                        global_skip = local_skip
                        if self.is_ddp:
                            import torch.distributed as dist
                            self.ddp_sync_flag[0] = 1 if local_skip else 0
                            dist.all_reduce(self.ddp_sync_flag, op=dist.ReduceOp.MAX)
                            global_skip = bool(int(self.ddp_sync_flag[0].item()) > 0)

                        if global_skip:
                            if self.is_main:
                                reason_str = ",".join(reasons) if reasons else "triggered_on_other_rank"
                                msg = f"[LossGuard] skip iter={self.iter} epoch={epoch} (G phase) reasons={reason_str} loss_G={losses_to_check['loss_G']:.6f}"
                                print(msg)
                                log2txt(self.log_path, msg)
                                # 记录 None 以避免可视化被尖峰拉伸
                                import time as _time
                                package = {
                                    'Epoch': epoch,
                                    'current_step': self.iter,
                                    'total_step': self.total_step,
                                    'loss_2D': None, 'loss_G': None, 'loss_GAN': None,
                                    'loss_feat': None, 'loss_vgg': None, 'loss_L1': None,
                                    'loss_edge': None, 'loss_laplacian': None,
                                    'D_real': d_scores.get('D_real'),
                                    'D_fake': d_scores.get('D_fake'),
                                    'inference': False,
                                    'event': {
                                        'type': 'loss_guard_skip',
                                        'epoch': epoch,
                                        'iter': self.iter,
                                        'reasons': reasons,
                                        'losses': losses_to_check,
                                        'timestamp': _time.time(),
                                    },
                                }
                                self.manager(package, save_image=False)
                            # 恢复 D 参数梯度
                            self._set_requires_grad(self.discriminator, True)
                            continue

                # Backward and optimize G
                self.G_optim.zero_grad()
                sync_ctx = self.discriminator.no_sync() if self.is_ddp else nullcontext()
                with sync_ctx:
                    loss_G.backward()
                if clip_gradient_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(),
                                                  clip_gradient_norm, norm_type='inf')
                self.G_optim.step()

                # EMA 更新（仅在 G 更新时执行）
                if self.model_ema is not None:
                    self.model_ema.update(unwrap_model(self.generator))

                # 恢复 D 参数梯度
                if freeze_d:
                    self._set_requires_grad(self.discriminator, True)
            else:
                # 不更新 G 时，使用占位值
                loss_G = torch.tensor(0.0, device=self.device)
                loss_dict = {}

            # === Logging and Visualization ===
            save_image = (self.train_image_save_every_iters > 0 and
                         self.iter % self.train_image_save_every_iters == 0)

            # 准备可视化图像
            fake_2D = fake_2D.detach()  # detach 用于保存

            # 生成 EMA 版本的 fake（如果启用）
            fake_raw = fake_2D
            fake_ema = None
            if self.is_main and self.generator_ema is not None and save_image:
                with torch.no_grad():
                    self.generator_ema.eval()
                    fake_ema = self.generator_ema(input_2D).detach()

            # 选择用于可视化的 fake
            fake_vis = fake_raw
            if self.is_main and self.ema_use_for_train_vis and fake_ema is not None and save_image:
                fake_vis = fake_ema

            # 准备日志数据
            with torch.no_grad():
                package = {
                    'Epoch': epoch,
                    'current_step': self.iter,
                    'total_step': self.total_step,
                    'loss_2D': loss_D.item() if isinstance(loss_D, torch.Tensor) else 0.0,
                    'loss_G': loss_G.item() if isinstance(loss_G, torch.Tensor) else 0.0,
                    'loss_GAN': float(loss_dict.get('GAN', torch.tensor(0.0)).item()),
                    'loss_feat': float(loss_dict.get('feat', torch.tensor(0.0)).item()),
                    'loss_vgg': float(loss_dict.get('VGG', torch.tensor(0.0)).item()),
                    'loss_L1': float(loss_dict.get('L1', torch.tensor(0.0)).item()),
                    'loss_edge': float(loss_dict.get('edge', torch.tensor(0.0)).item()),
                    'loss_laplacian': float(loss_dict.get('laplacian', torch.tensor(0.0)).item()),
                    # 判别器监控分数
                    'D_real': d_scores['D_real'],
                    'D_fake': d_scores['D_fake'],
                    # 图像张量（clone 避免长期持有引用）
                    'input_tensor': input_2D.clone().detach(),
                    'target_tensor': target_2D.clone().detach(),
                    'fake_tensor': fake_vis.clone().detach(),
                    'fake_tensor_raw': fake_raw.clone().detach(),
                    'fake_tensor_ema': fake_ema.clone().detach() if fake_ema is not None else None,
                    'inference': False
                }

            # DDP 下仅主进程做可视化/写盘
            if self.is_main:
                self.manager(package, save_image=save_image)

            if (i + 1) % print_freq == 0 and self.is_main:
                edge_str = f", edge:{loss_dict.get('edge', 0):.4f}" if loss_dict.get('edge', 0) > 0 else ""
                lap_str = f", lap:{loss_dict.get('laplacian', 0):.4f}" if loss_dict.get('laplacian', 0) > 0 else ""
                d_str = f", D_real:{d_scores['D_real']:.3f}, D_fake:{d_scores['D_fake']:.3f}"
                print(f"  [{i + 1}/{len(self.train_loader)}] "
                      f"D:{loss_D.item() if isinstance(loss_D, torch.Tensor) else 0:.4f}, "
                      f"G:{loss_G.item() if isinstance(loss_G, torch.Tensor) else 0:.4f} "
                      f"(GAN:{loss_dict.get('GAN', 0):.4f}, "
                      f"feat:{loss_dict.get('feat', 0):.4f}, "
                      f"vgg:{loss_dict.get('VGG', 0):.4f}, "
                      f"L1:{loss_dict.get('L1', 0):.4f}{edge_str}{lap_str}){d_str}")
                log2txt(self.log_path, f"iter_{self.iter}: D:{loss_D.item() if isinstance(loss_D, torch.Tensor) else 0:.4f}, "
                        f"G:{loss_G.item() if isinstance(loss_G, torch.Tensor) else 0:.4f}, "
                        f"GAN:{loss_dict.get('GAN', 0):.4f}, feat:{loss_dict.get('feat', 0):.4f}, "
                        f"VGG:{loss_dict.get('VGG', 0):.4f}, L1:{loss_dict.get('L1', 0):.4f}{edge_str}{lap_str}{d_str}")

        return

    @torch.no_grad()
    def validate(self, epoch):
        """验证（DDP 下使用分布式验证）"""

        self.generator.eval()
        self.discriminator.eval()
        gen_for_val = self.generator
        if self.ema_use_for_val and self.generator_ema is not None:
            gen_for_val = self.generator_ema
            gen_for_val.eval()

        # DDP 模式下也设置 sampler 的 epoch（保持行为一致，尽管 shuffle=False）
        if getattr(self, "val_sampler", None) is not None:
            try:
                self.val_sampler.set_epoch(epoch)
            except Exception:
                pass

        local_ssim_sum = 0.0
        local_psnr_sum = 0.0
        local_count = 0

        loader_iter = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc="val",
            disable=not self.is_main,
        )

        for batch_idx, val_input in loader_iter:
            fake_b, input_b, target_b = self.loss_module.inference(gen_for_val, val_input, first_only=False)
            batch_names = val_input.get('name') if isinstance(val_input, dict) else None

            bsz = int(fake_b.shape[0])
            for bi in range(bsz):
                fake_img = tensor2image(fake_b[bi], inference=True)
                target_img = tensor2image(target_b[bi], inference=True)
                ssim_val = float(get_ssim(fake_img, target_img))
                psnr_val = float(get_psnr(fake_img, target_img))

                local_ssim_sum += ssim_val
                local_psnr_sum += psnr_val
            local_count += bsz

            # DDP下所有rank都需要参与forward以避免集体通信死锁
            # 但只有主进程保存图像到磁盘
            should_save_images = (self.manager.val_image_save_every_n > 0 and
                                  batch_idx % self.manager.val_image_save_every_n == 0)

            fake_img_raw = None
            fake_img_ema = None
            if should_save_images and self.generator_ema is not None:
                # 所有rank都参与DDP模型的forward（避免死锁）
                try:
                    raw_fake_0 = self.generator(input_b[:1])[0]  # DDP-wrapped，所有rank必须同时调用
                    ema_fake_0 = self.generator_ema(input_b[:1])[0]  # EMA未被DDP包装，无影响
                    # 只在主进程转换为图像（节省显存和计算）
                    if self.is_main:
                        fake_img_raw = tensor2image(raw_fake_0, inference=True)
                        fake_img_ema = tensor2image(ema_fake_0, inference=True)
                except Exception:
                    fake_img_raw = None
                    fake_img_ema = None

            # 只在主进程做可视化/落盘
            if self.is_main and should_save_images:
                # 仅取本 batch 第一个样本用于展示
                fake_0 = fake_b[0]
                input_0 = input_b[0]
                target_0 = target_b[0]
                sample_name = None
                if isinstance(batch_names, (list, tuple)) and len(batch_names) > 0:
                    sample_name = batch_names[0]
                # 统一验证可视化为 inference=True（使用窗宽窗位）
                fake_img = tensor2image(fake_0, inference=True)
                input_img = tensor2image(input_0, inference=True)
                target_img = tensor2image(target_0, inference=True)
                package = {
                    'Epoch': epoch,
                    'current_step': epoch,
                    'index': batch_idx,
                    'eva_ssim': (local_ssim_sum / max(local_count, 1)),
                    'eva_psnr': (local_psnr_sum / max(local_count, 1)),
                    'input_tensor': input_0,
                    'fake_tensor': fake_0.detach(),
                    'target_tensor': target_0,
                    'inference': True,
                    'sample_name': sample_name,
                    'input_image': input_img,
                    'target_image': target_img,
                    'fake_image': fake_img,
                    # optional: raw vs EMA compare for web/disk
                    'fake_image_raw': fake_img_raw,
                    'fake_image_ema': fake_img_ema,
                }
                self.manager(package, save_image=False)

        # DEBUG: 打印本地统计信息
        if self.is_main:
            local_mean_ssim = local_ssim_sum / max(local_count, 1)
            print(f"[DEBUG] Rank {self.local_rank}: local_ssim_sum={local_ssim_sum:.4f}, "
                  f"local_count={local_count}, local_mean={local_mean_ssim:.4f}")
            if debug_ssim_values:
                print(f"[DEBUG] SSIM range: min={min(debug_ssim_values):.4f}, "
                      f"max={max(debug_ssim_values):.4f}, samples={len(debug_ssim_values)}")

        # 汇总所有进程的指标（DDP 下 all_reduce）
        total_ssim_sum = local_ssim_sum
        total_psnr_sum = local_psnr_sum
        total_count = local_count
        if self.is_ddp:
            import torch.distributed as dist
            t = torch.tensor([local_ssim_sum, local_psnr_sum, float(local_count)],
                             device=self.device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_ssim_sum, total_psnr_sum, total_count = float(t[0].item()), float(t[1].item()), float(t[2].item())

            # DEBUG: 打印汇总后的统计信息
            if self.is_main:
                print(f"[DEBUG] After all_reduce: total_ssim_sum={total_ssim_sum:.4f}, "
                      f"total_count={total_count}, world_size={self.world_size}")

        if self.is_main:
            mean_ssim = total_ssim_sum / max(total_count, 1.0)
            mean_psnr = total_psnr_sum / max(total_count, 1.0)
            print(f"epoch_{str(epoch).zfill(3)}, SSIM: {mean_ssim:.4f}, PSNR: {mean_psnr:.4f}")
            log2txt(self.log_path, f"epoch_{str(epoch).zfill(3)}: SSIM: {mean_ssim:.4f}, PSNR: {mean_psnr:.4f}")

    def train(self, continue_train=False):
        """完整训练流程"""
        self.iter = 0
        if continue_train:
            self.load_checkpoint()

        total_epoch = self.cfg['training']['total_epoch']
        val_freq = self.cfg['training']['val_freq']
        will_val = self.cfg.get('will_val', True)
        lr_decay = self.cfg.get('lr_decay', 0)

        self.total_step = len(self.train_loader) * total_epoch

        try:
            for epoch in range(self.start_epoch, total_epoch):
                self.train_epoch(epoch)

                # Validation（只在主进程运行）
                if will_val and (epoch + 1) % val_freq == 0:
                    self.validate(epoch)

                # Learning rate decay
                if epoch > lr_decay - 1:
                    self.scheduler_G.step()
                    self.scheduler_D.step()

                # Checkpoint 保存（只在主进程保存）
                if self.is_main:
                    epoch_next = epoch + 1
                    # 获取原始模型（从 DDP/DP 包装中提取）
                    gen_to_save = unwrap_model(self.generator)
                    disc_to_save = unwrap_model(self.discriminator)
                    gen_ema_to_save = None
                    ema_meta = None
                    if self.model_ema is not None and self.generator_ema is not None:
                        gen_ema_to_save = unwrap_model(self.generator_ema)
                        ema_meta = self.model_ema.get_meta()

                    self.manager.save_latest_checkpoint(
                        discriminator=disc_to_save,
                        generator=gen_to_save,
                        generator_ema=gen_ema_to_save,
                        D_optim=self.D_optim,
                        G_optim=self.G_optim,
                        scheduler_D=self.scheduler_D,
                        scheduler_G=self.scheduler_G,
                        epoch_next=epoch_next,
                        iter_step=self.iter,
                        ema_meta=ema_meta,
                    )
                    self.manager.save_periodic_checkpoint(
                        discriminator=disc_to_save,
                        generator=gen_to_save,
                        generator_ema=gen_ema_to_save,
                        D_optim=self.D_optim,
                        G_optim=self.G_optim,
                        scheduler_D=self.scheduler_D,
                        scheduler_G=self.scheduler_G,
                        epoch_next=epoch_next,
                        iter_step=self.iter,
                        ema_meta=ema_meta,
                    )

                # DDP 模式下同步所有进程
                if self.is_ddp:
                    import torch.distributed as dist
                    dist.barrier()

        finally:
            pass  # DDP 清理由 train.py 中的 ddp_worker 处理
