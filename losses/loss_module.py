"""统一的损失计算模块

设计原则：
1. 可扩展：通过 register_loss() 动态添加新的 loss 函数
2. 统一接口：所有 loss 权重由外部 LossScheduler 提供
3. 内置常用 loss：GAN, L1, VGG, feat, edge, laplacian
"""

from typing import Dict, Callable, Optional, List, Union
import torch
import torch.nn as nn

from .gan_loss import GANLoss
from .vgg_loss import VGGLoss, EdgeLoss, LaplacianLoss


def add_3channel(input_tensor):
    """将单通道图像复制为3通道 (用于VGG)"""
    return torch.cat((input_tensor, input_tensor, input_tensor), 1)


def focal_loss_weighted_loss(loss1, loss2, gamma=2.0):
    """根据Focal Loss思想自适应融合两个损失"""
    w1 = (1 - torch.exp(-loss1)) ** gamma
    w2 = (1 - torch.exp(-loss2)) ** gamma
    w1_normalized = w1 / (w1 + w2)
    w2_normalized = w2 / (w1 + w2)
    total_loss = w1_normalized * loss1 + w2_normalized * loss2
    return total_loss


class LossModule(nn.Module):
    """统一的损失计算模块

    可扩展设计：
    - 内置 GAN, L1, VGG, feat, edge, laplacian 损失
    - 通过 register_loss() 动态添加新的损失函数
    - 所有权重由外部 LossScheduler 提供

    Args:
        use_gpus: GPU列表
        num_D: 判别器数量
        n_layers_D: 判别器层数
        vgg_use_imagenet_norm: VGG是否使用ImageNet归一化
        vgg_clamp_to_01: VGG输入是否clamp到[0,1]
        vgg_weights: VGG层权重配置 (None, str预设, 或list)

    Example:
        # 添加自定义 loss
        def my_loss_fn(fake, target, **ctx):
            return F.mse_loss(fake, target)

        loss_module.register_loss('my_loss', my_loss_fn)

        # 在 loss_schedule 配置中添加:
        # my_loss:
        #   type: "constant"
        #   value: 5.0
    """

    def __init__(self, use_gpus=[0], num_D=3, n_layers_D=3,
                 vgg_use_imagenet_norm: bool = False, vgg_clamp_to_01: bool = True,
                 vgg_weights: Optional[Union[str, List[float]]] = None):
        super().__init__()
        self.num_D = num_D
        self.n_layers_D = n_layers_D
        self.device = torch.device(f'cuda:{use_gpus[0]}' if torch.cuda.is_available() else 'cpu')

        # 基础损失函数
        self.criterionL1 = nn.L1Loss()
        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.criterionVGG = VGGLoss(
            self.device,
            use_imagenet_norm=vgg_use_imagenet_norm,
            clamp_to_01=vgg_clamp_to_01,
            weights=vgg_weights
        )

        # 细节增强损失
        self.criterionEdge = EdgeLoss().to(self.device)
        self.criterionLaplacian = LaplacianLoss().to(self.device)

        # 预创建零tensor，避免频繁分配（显存碎片优化）
        self.register_buffer('zero_loss', torch.tensor(0.0, dtype=torch.float32), persistent=False)

        # 自定义损失函数注册表
        # 签名: fn(fake, target, **context) -> torch.Tensor
        self._custom_losses: Dict[str, Callable] = {}

    def register_loss(self, name: str, loss_fn: Callable):
        """注册自定义损失函数

        Args:
            name: 损失函数名称（需与 loss_schedule 中的名称一致）
            loss_fn: 损失函数，签名为 (fake, target, **context) -> torch.Tensor
                     context 包含: input, pred_real, pred_fake, discriminator

        Example:
            def ssim_loss(fake, target, **ctx):
                from pytorch_msssim import ssim
                return 1 - ssim(fake, target, data_range=2.0)

            loss_module.register_loss('ssim', ssim_loss)
        """
        self._custom_losses[name] = loss_fn

    def get_registered_losses(self) -> list:
        """获取所有已注册的自定义损失名称"""
        return list(self._custom_losses.keys())

    def compute_discriminator_loss(self, discriminator, input_2D, target_2D, fake_2D):
        """计算判别器损失

        注意：fake_2D 必须已经 detach，由调用者确保
        """
        pred_real = discriminator(torch.cat((input_2D, target_2D), dim=1))
        pred_fake = discriminator(torch.cat((input_2D, fake_2D.detach()), dim=1))
        loss_D = (self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False)) * 0.5
        return loss_D, pred_real, pred_fake

    @staticmethod
    def get_discriminator_scores(pred_real, pred_fake) -> dict:
        """计算判别器对真实和生成图像的平均分数

        针对 PatchGAN 多尺度判别器的输出结构：
        - pred: List[List[Tensor]] 结构
        - pred[scale][layer]，最后一层是 patch-wise logits

        对于 LSGAN，输出没有经过 sigmoid，是原始 logits。
        我们对所有 patch 取平均，再对所有尺度取平均。

        Returns:
            dict: {
                'D_real': 判别器对真实图像的平均分数,
                'D_fake': 判别器对生成图像的平均分数,
                'D_real_std': 真实图像分数的标准差,
                'D_fake_std': 生成图像分数的标准差,
            }

        健康状态参考（LSGAN, 目标 real=1, fake=0）:
        - 健康: D_real ≈ 0.7-1.0, D_fake ≈ 0.0-0.3
        - D过强: D_real ≈ 1.0, D_fake ≈ 0.0 (梯度消失)
        - D过弱: D_real ≈ D_fake ≈ 0.5 (失去区分能力)
        """
        def extract_scores(pred):
            """从多尺度判别器输出中提取分数"""
            scores = []
            if isinstance(pred[0], list):
                # 多尺度判别器: pred[scale][layer]
                for scale_pred in pred:
                    # 取最后一层的输出 (logits)
                    logits = scale_pred[-1]
                    # 对每个 patch 取平均
                    scores.append(logits.mean().item())
            else:
                # 单尺度: pred[layer]
                logits = pred[-1]
                scores.append(logits.mean().item())
            return scores

        real_scores = extract_scores(pred_real)
        fake_scores = extract_scores(pred_fake)

        return {
            'D_real': float(sum(real_scores) / len(real_scores)),
            'D_fake': float(sum(fake_scores) / len(fake_scores)),
            'D_real_std': float(torch.tensor(real_scores).std().item()) if len(real_scores) > 1 else 0.0,
            'D_fake_std': float(torch.tensor(fake_scores).std().item()) if len(fake_scores) > 1 else 0.0,
        }

    def compute_generator_loss(self, discriminator, input_2D, target_2D, fake_2D,
                                pred_real, weights: Dict[str, float]):
        """计算生成器损失

        Args:
            discriminator: 判别器模型
            input_2D: 输入图像
            target_2D: 目标图像
            fake_2D: 生成图像
            pred_real: 判别器对真实图像的预测
            weights: 损失权重字典 (由 LossScheduler.get_all_weights() 提供)

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细字典
        """
        loss_dict = {}

        # 获取 pred_fake 用于 GAN 和 feat loss
        pred_fake = discriminator(torch.cat((input_2D, fake_2D), dim=1))

        # === 内置损失 ===

        # GAN loss
        w_gan = weights.get('GAN', 0)
        if w_gan > 0:
            loss_gan_raw = self.criterionGAN(pred_fake, True)
            loss_dict['GAN'] = loss_gan_raw * w_gan
        else:
            loss_dict['GAN'] = self.zero_loss

        # Feature matching loss
        w_feat = weights.get('feat', 0)
        if w_feat > 0:
            # 优化：使用列表收集后再sum，避免Python标量累加产生的显存碎片
            feat_losses = []
            feat_weights = 4.0 / (self.n_layers_D + 1)
            D_weights = 1.0 / self.num_D
            for i in range(self.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    feat_losses.append(D_weights * feat_weights *
                                     self.criterionL1(pred_fake[i][j], pred_real[i][j].detach()))
            loss_dict['feat'] = sum(feat_losses) * w_feat if feat_losses else self.zero_loss
        else:
            loss_dict['feat'] = self.zero_loss

        # VGG loss
        w_vgg = weights.get('VGG', 0)
        if w_vgg > 0:
            channels = fake_2D.shape[1]
            if channels == 1:
                loss_vgg_raw = self.criterionVGG(add_3channel(fake_2D), add_3channel(target_2D))
            else:
                # 保持原有逻辑：对每个2.5D通道分别计算VGG（用户要求不修改）
                vgg_loss = 0
                for c in range(channels):
                    fake_c = add_3channel(fake_2D[:, c:c + 1])
                    target_c = add_3channel(target_2D[:, c:c + 1])
                    vgg_loss += self.criterionVGG(fake_c, target_c)
                loss_vgg_raw = vgg_loss
            loss_dict['VGG'] = loss_vgg_raw * w_vgg
        else:
            loss_dict['VGG'] = self.zero_loss

        # L1 loss
        w_l1 = weights.get('L1', 0)
        if w_l1 > 0:
            loss_dict['L1'] = self.criterionL1(fake_2D, target_2D) * w_l1
        else:
            loss_dict['L1'] = self.zero_loss

        # Edge loss
        w_edge = weights.get('edge', 0)
        if w_edge > 0:
            loss_dict['edge'] = self.criterionEdge(fake_2D, target_2D) * w_edge
        else:
            loss_dict['edge'] = self.zero_loss

        # Laplacian loss
        w_lap = weights.get('laplacian', 0)
        if w_lap > 0:
            loss_dict['laplacian'] = self.criterionLaplacian(fake_2D, target_2D) * w_lap
        else:
            loss_dict['laplacian'] = self.zero_loss

        # === 自定义损失 ===
        context = {
            'input': input_2D,
            'pred_real': pred_real,
            'pred_fake': pred_fake,
            'discriminator': discriminator,
            'device': self.device,
        }

        for loss_name, loss_fn in self._custom_losses.items():
            w = weights.get(loss_name, 0)
            if w > 0:
                loss_dict[loss_name] = loss_fn(fake_2D, target_2D, **context) * w
            else:
                loss_dict[loss_name] = self.zero_loss

        # 计算总损失
        total_loss = sum(loss_dict.values())

        return total_loss, loss_dict

    def forward_D(self, discriminator, generator, input_dic, return_images: bool = True):
        """判别器的 forward pass（标准 GAN 训练：fake 会自动 detach）

        这是推荐的训练方法，遵循标准 GAN 训练流程。

        Args:
            discriminator: 判别器模型
            generator: 生成器模型
            input_dic: 训练数据字典 {'source': input, 'cta': target}
            return_images: 是否返回图像张量（用于可视化）

        Returns:
            loss_D: 判别器损失
            d_scores: 判别器分数字典 {'D_real': ..., 'D_fake': ...}
            images (可选): {'input': ..., 'target': ..., 'fake': ...}
        """
        input_2D = input_dic['source'].to(self.device)
        target_2D = input_dic['cta'].to(self.device)

        # 生成 fake 图像并 detach（关键：不影响 G 的梯度）
        with torch.no_grad():
            fake_2D = generator(input_2D)
        fake_2D = fake_2D.detach()  # 显式 detach 确保安全

        # 计算判别器损失
        loss_D, pred_real, pred_fake = self.compute_discriminator_loss(
            discriminator, input_2D, target_2D, fake_2D)

        # 计算判别器分数（用于监控）
        d_scores = self.get_discriminator_scores(pred_real, pred_fake)

        if return_images:
            images = {
                'input': input_2D,
                'target': target_2D,
                'fake': fake_2D
            }
            return loss_D, d_scores, images

        return loss_D, d_scores

    def forward_G(self, discriminator, generator, input_dic, weights: Dict[str, float],
                  return_images: bool = True, compute_pred_real: bool = True):
        """生成器的 forward pass（标准 GAN 训练：fake 不 detach）

        这是推荐的训练方法，遵循标准 GAN 训练流程。

        Args:
            discriminator: 判别器模型（作为 fixed network）
            generator: 生成器模型
            input_dic: 训练数据字典 {'source': input, 'cta': target}
            weights: 损失权重字典 (由 LossScheduler.get_all_weights() 提供)
            return_images: 是否返回图像张量（用于可视化）
            compute_pred_real: 是否计算 pred_real（用于 feature matching loss）

        Returns:
            loss_G: 生成器总损失
            loss_dict: 各项损失的详细字典 {'GAN': ..., 'feat': ..., 'VGG': ..., ...}
            images (可选): {'input': ..., 'target': ..., 'fake': ...}
        """
        input_2D = input_dic['source'].to(self.device)
        target_2D = input_dic['cta'].to(self.device)

        # 生成 fake 图像（不 detach，包含 G 的梯度）
        fake_2D = generator(input_2D)

        # 计算 pred_real（用于 feature matching loss）
        pred_real = None
        if compute_pred_real and weights.get('feat', 0) > 0:
            with torch.no_grad():
                pred_real = discriminator(torch.cat((input_2D, target_2D), dim=1))

        # 计算生成器损失
        loss_G, loss_dict = self.compute_generator_loss(
            discriminator, input_2D, target_2D, fake_2D, pred_real, weights=weights)

        if return_images:
            images = {
                'input': input_2D,
                'target': target_2D,
                'fake': fake_2D
            }
            return loss_G, loss_dict, images

        return loss_G, loss_dict

    def __call__(self, discriminator, generator, train_dic, weights: Dict[str, float],
                 return_d_scores: bool = False):
        """完整的前向传播（DEPRECATED：为向后兼容保留）

        警告：这个方法在一次 forward 中调用判别器两次（一次 detached，一次非 detached），
        可能导致计算图冲突。推荐使用 forward_D() 和 forward_G() 分别训练。

        Args:
            discriminator: 判别器模型
            generator: 生成器模型
            train_dic: 训练数据字典
            weights: 损失权重 (由 LossScheduler.get_all_weights() 提供)
            return_d_scores: 是否返回判别器分数用于监控

        Returns:
            loss_D: 判别器损失
            loss_G_tuple: (total_G_loss, GAN, feat, VGG, L1, edge, laplacian, ...)
            input_2D: 输入图像
            target_2D: 目标图像
            fake_2D: 生成图像
            d_scores (可选): 判别器分数字典，仅当 return_d_scores=True 时返回
        """
        im = train_dic['source'].to(self.device)
        target = train_dic['cta'].to(self.device)

        # 生成假图像
        fake = generator(im)

        # 计算损失
        loss_D, pred_real, pred_fake = self.compute_discriminator_loss(
            discriminator, im, target, fake)

        loss_G, loss_dict = self.compute_generator_loss(
            discriminator, im, target, fake, pred_real, weights=weights)

        # 构建返回的 tuple（保持向后兼容）
        # 基础顺序: GAN, feat, VGG, L1, edge, laplacian
        base_losses = (
            loss_G,
            loss_dict.get('GAN', torch.tensor(0.0)),
            loss_dict.get('feat', torch.tensor(0.0)),
            loss_dict.get('VGG', torch.tensor(0.0)),
            loss_dict.get('L1', torch.tensor(0.0)),
            loss_dict.get('edge', torch.tensor(0.0)),
            loss_dict.get('laplacian', torch.tensor(0.0)),
        )

        # 添加自定义损失
        custom_losses = tuple(
            loss_dict.get(name, torch.tensor(0.0))
            for name in self._custom_losses.keys()
        )

        if return_d_scores:
            d_scores = self.get_discriminator_scores(pred_real, pred_fake)
            return loss_D, base_losses + custom_losses, im, target, fake, d_scores

        return loss_D, base_losses + custom_losses, im, target, fake

    def get_loss_dict(self, discriminator, generator, train_dic, weights: Dict[str, float],
                      return_d_scores: bool = False):
        """获取所有损失的字典形式（推荐使用）

        Returns:
            loss_D: 判别器损失
            loss_G: 生成器总损失
            loss_dict: 各项损失的详细字典
            im, target, fake: 图像张量
            d_scores (可选): 判别器分数字典
        """
        im = train_dic['source'].to(self.device)
        target = train_dic['cta'].to(self.device)

        fake = generator(im)

        loss_D, pred_real, pred_fake = self.compute_discriminator_loss(
            discriminator, im, target, fake)

        loss_G, loss_dict = self.compute_generator_loss(
            discriminator, im, target, fake, pred_real, weights=weights)

        if return_d_scores:
            d_scores = self.get_discriminator_scores(pred_real, pred_fake)
            return loss_D, loss_G, loss_dict, im, target, fake, d_scores

        return loss_D, loss_G, loss_dict, im, target, fake

    def inference(self, generator, val_dic, first_only: bool = True):
        """推理模式

        Args:
            generator: 生成器模型
            val_dic: dataloader 输出字典
            first_only: True 时仅返回 batch 中第 0 个样本（向后兼容）；False 时返回整个 batch
        """
        target = val_dic.get('cta')
        im = val_dic['source']
        input_2D = im.to(self.device)
        fake_2D = generator(input_2D)

        if target is not None:
            target_2D = target.to(self.device)
            if first_only:
                return fake_2D[0], input_2D[0], target_2D[0]
            return fake_2D, input_2D, target_2D
        else:
            if first_only:
                return fake_2D[0], input_2D[0]
            return fake_2D, input_2D


# 移除旧的 LossWeights 类，统一使用 Dict[str, float]
