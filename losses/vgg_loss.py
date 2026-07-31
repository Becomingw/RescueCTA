"""VGG感知损失"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights


class Vgg19(nn.Module):
    """VGG19特征提取网络"""

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


# 预定义的权重配置
VGG_WEIGHT_PRESETS = {
    # 原始配置：偏向深层语义
    'default': [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0],
    # 细节增强：强调浅层特征（边缘、纹理）
    'detail': [1.0, 1.0/2, 1.0/4, 1.0/8, 1.0/16],
    # 平衡配置：各层均衡
    'balanced': [1.0/5, 1.0/5, 1.0/5, 1.0/5, 1.0/5],
    # 细节优先但保留语义
    'detail_semantic': [1.0, 0.5, 0.25, 0.25, 0.5],
    # 只用浅层（最强调细节）
    'shallow_only': [1.0, 0.5, 0.1, 0.0, 0.0],
}


class VGGLoss(nn.Module):
    """VGG感知损失

    Args:
        device: 计算设备
        use_imagenet_norm: 是否使用ImageNet归一化
        clamp_to_01: 是否将输入clamp到[0,1]
        weights: 各层权重，可以是:
            - None: 使用默认权重
            - str: 预设名称 ('default', 'detail', 'balanced', 'detail_semantic', 'shallow_only')
            - list: 自定义5个权重值
    """

    def __init__(self, device, use_imagenet_norm: bool = False, clamp_to_01: bool = True,
                 weights=None):
        super(VGGLoss, self).__init__()
        self.device = torch.device(device)
        self.vgg = Vgg19().to(self.device)
        self.criterion = nn.L1Loss()

        # 设置层权重
        if weights is None:
            self.weights = VGG_WEIGHT_PRESETS['default']
        elif isinstance(weights, str):
            if weights in VGG_WEIGHT_PRESETS:
                self.weights = VGG_WEIGHT_PRESETS[weights]
            else:
                raise ValueError(f"Unknown weight preset: {weights}. "
                               f"Available: {list(VGG_WEIGHT_PRESETS.keys())}")
        else:
            assert len(weights) == 5, "weights must have 5 elements"
            self.weights = list(weights)

        self.use_imagenet_norm = use_imagenet_norm
        self.clamp_to_01 = clamp_to_01

        # Keep buffers on the same device as the module to avoid cuda/cpu mismatch.
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.register_buffer('imagenet_mean', mean, persistent=False)
        self.register_buffer('imagenet_std', std, persistent=False)

    def forward(self, x, y):
        if self.use_imagenet_norm:
            # model tensors are in [-1, 1]; VGG expects ImageNet-normalized [0, 1]
            x = (x + 1) * 0.5
            y = (y + 1) * 0.5
            if self.clamp_to_01:
                x = x.clamp(0.0, 1.0)
                y = y.clamp(0.0, 1.0)
            mean = self.imagenet_mean
            std = self.imagenet_std
            if mean.dtype != x.dtype:
                mean = mean.to(dtype=x.dtype)
                std = std.to(dtype=x.dtype)
            x = (x - mean) / std
            y = (y - mean) / std

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # 优化：使用列表收集后再sum，避免Python标量累加产生的显存碎片
        losses = []
        for i in range(len(x_vgg)):
            if self.weights[i] > 0:
                losses.append(self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()))
        return sum(losses) if losses else torch.tensor(0.0, device=x.device, dtype=x.dtype)


class EdgeLoss(nn.Module):
    """边缘/梯度损失 - 增强细节生成

    使用Sobel算子提取边缘，计算边缘的L1损失
    """

    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        self.criterion = nn.L1Loss()

    def get_edges(self, x):
        """提取边缘"""
        # 处理多通道
        if x.shape[1] > 1:
            # 优化：使用stack代替list+cat，减少显存碎片
            batch_size, channels, h, w = x.shape
            # 预分配输出tensor
            edges = torch.empty(batch_size, channels, h, w, device=x.device, dtype=x.dtype)
            for c in range(channels):
                xc = x[:, c:c+1]
                gx = F.conv2d(xc, self.sobel_x, padding=1)
                gy = F.conv2d(xc, self.sobel_y, padding=1)
                edges[:, c:c+1] = torch.sqrt(gx**2 + gy**2 + 1e-8)
            return edges
        else:
            gx = F.conv2d(x, self.sobel_x, padding=1)
            gy = F.conv2d(x, self.sobel_y, padding=1)
            return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def forward(self, fake, real):
        fake_edges = self.get_edges(fake)
        real_edges = self.get_edges(real)
        return self.criterion(fake_edges, real_edges)


class LaplacianLoss(nn.Module):
    """拉普拉斯损失 - 增强高频细节

    使用拉普拉斯算子提取高频信息
    """

    def __init__(self):
        super(LaplacianLoss, self).__init__()
        # 拉普拉斯算子
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))
        self.criterion = nn.L1Loss()

    def get_laplacian(self, x):
        """提取高频信息"""
        if x.shape[1] > 1:
            # 优化：预分配tensor代替list+cat，减少显存碎片
            batch_size, channels, h, w = x.shape
            results = torch.empty(batch_size, channels, h, w, device=x.device, dtype=x.dtype)
            for c in range(channels):
                xc = x[:, c:c+1]
                results[:, c:c+1] = F.conv2d(xc, self.laplacian, padding=1)
            return results
        else:
            return F.conv2d(x, self.laplacian, padding=1)

    def forward(self, fake, real):
        fake_lap = self.get_laplacian(fake)
        real_lap = self.get_laplacian(real)
        return self.criterion(fake_lap, real_lap)
