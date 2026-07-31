"""RescueCTA generator with hierarchical high-resolution feature fusion.

Original HRNet: Copyright (c) Microsoft, Licensed under MIT License
Modifications to ImageEnhancementNetwork: Copyright (c) Intel, Licensed under MIT License
RescueCTA adaptations: Mengfei Wang and collaborators.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_norm_func(norm_type, **kwargs):
    """获取归一化层"""
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, **kwargs)
    elif norm_type == 'group':
        return functools.partial(nn.GroupNorm, **kwargs)
    elif norm_type == 'ins':
        return functools.partial(nn.InstanceNorm2d, **kwargs)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=True)
    )


def make_conv_layer(dims, strides=1, leaky_relu=True, spectral=False, norm_factory=None,
                    skip_final_relu=False, kernel=3):
    """构建简单卷积网络"""
    if type(strides) == int:
        strides = [strides] + [1] * (len(dims) - 2)

    c = nn.Conv2d(dims[0], dims[1], kernel, stride=strides[0], bias=spectral)
    m = [] if kernel == 1 else [nn.ReplicationPad2d(kernel // 2)]
    m += [c if not spectral else torch.nn.utils.spectral_norm(c)]

    if norm_factory:
        m += [norm_factory(dims[1])]

    m += [nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)]

    num_convs = len(dims) - 2
    for i, di in enumerate(dims[2:]):
        c = nn.Conv2d(dims[i + 1], di, 3, stride=strides[i + 1], bias=spectral)

        if kernel > 1:
            m += [nn.ReplicationPad2d(kernel // 2)]
        m += [c if not spectral else torch.nn.utils.spectral_norm(c)]

        if norm_factory:
            m += [norm_factory(di)]

        if i == num_convs - 1 and skip_final_relu:
            continue
        else:
            m += [nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)]

    return nn.Sequential(*m)


def make_blocks_dict(nor_type="ins"):
    """创建block字典"""
    return {
        'BASIC': (lambda *args, **kwargs: BasicBlock(*args, **{'norm_func': get_norm_func(nor_type), **kwargs}),
                  BasicBlock.expansion),
        'BOTTLENECK': (lambda *args, **kwargs: Bottleneck(*args, **{'norm_func': get_norm_func(nor_type), **kwargs}),
                       Bottleneck.expansion)
    }


class BasicBlock(nn.Module):
    """基础残差块"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_func=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_func(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Sequential(nn.ReLU(inplace=True), conv3x3(planes, planes))
        self.bn2 = norm_func(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        r = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = 0.1 * x + r
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    """瓶颈残差块"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_func=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_func(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_func(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_func(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class HighResolutionModule(nn.Module):
    """高分辨率模块"""
    def __init__(self, num_branches, blocks, block_expansion, num_blocks, num_inchannels,
                 num_channels, fuse_method, norm, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.Norm2d = norm
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, block_expansion, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            raise ValueError(f'NUM_BRANCHES({num_branches}) <> NUM_BLOCKS({len(num_blocks)})')
        if num_branches != len(num_channels):
            raise ValueError(f'NUM_BRANCHES({num_branches}) <> NUM_CHANNELS({len(num_channels)})')
        if num_branches != len(num_inchannels):
            raise ValueError(f'NUM_BRANCHES({num_branches}) <> NUM_INCHANNELS({len(num_inchannels)})')

    def _make_one_branch(self, branch_index, block, block_expansion, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block_expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block_expansion,
                          kernel_size=1, stride=stride, bias=True),
                self.Norm2d(num_channels[branch_index] * block_expansion)
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block_expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, block_expansion, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, block_expansion, num_blocks, num_channels, 1))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=True),
                        self.Norm2d(num_inchannels[i])
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.ReplicationPad2d(1),
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 0, bias=True),
                                self.Norm2d(num_outchannels_conv3x3)
                            ))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.ReplicationPad2d(1),
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 0, bias=True),
                                self.Norm2d(num_outchannels_conv3x3),
                                nn.ReLU(inplace=False)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0]([x[0]])[0]]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class RescueCTAGenerator(nn.Module):
    """Hierarchical 2.5D generator used by RescueCTA."""

    def __init__(self, hr_config, ie_config):
        extra = hr_config
        super().__init__()

        self._log = logging.getLogger(self.__class__.__name__)

        self.norm_type = ie_config.get('my_norm', 'ins')
        self._other_norm = ie_config.get('other_norm', 'ins')
        self._stem_norm = ie_config.get('stem_norm', 'ins')
        self._num_classes = ie_config.get('num_classes', 1)
        self._num_stages = int(ie_config.get('num_stages', 6))

        blocks_dict = make_blocks_dict(self.norm_type)
        self.Norm2d = get_norm_func('batch')

        dim_in = ie_config.get('in_channels', 1)  # 支持多通道输入
        dim_out = ie_config.get('out_channels', dim_in)

        self.stem = make_conv_layer([dim_in, 16, 16], 1, False, False, norm_factory=get_norm_func('batch'))

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block, block_exp = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, block_exp, 16, num_channels, num_blocks, 1)
        pre_stage_channels = [block_exp * num_channels]

        stage_cfgs = []
        stages = []
        transitions = []
        for si in range(2, self._num_stages + 1):
            stage_cfg = extra[f'STAGE{si}']
            num_channels = stage_cfg['NUM_CHANNELS']
            block, block_exp = blocks_dict[stage_cfg['BLOCK']]
            num_channels = [num_channels[i] * block_exp for i in range(len(num_channels))]
            transitions.append(self._make_transition_layer(pre_stage_channels, num_channels))
            stage, pre_stage_channels = self._make_stage(block, block_exp, stage_cfg, num_channels)
            stages.append(stage)
            stage_cfgs.append(stage_cfg)

        self.transitions = nn.ModuleList(transitions)
        self.stages = nn.ModuleList(stages)
        self.stage_cfgs = stage_cfgs

        out_channels = pre_stage_channels[::-1]

        last_layers = []
        for i, (ci, co) in enumerate(zip(out_channels[:-1], out_channels[1:])):
            m = [nn.ReplicationPad2d(1), nn.Conv2d(ci + co, co, 3), nn.LeakyReLU(0.2, True)]
            if i == self._num_stages - 2:
                m += [nn.ReplicationPad2d(1), nn.Conv2d(co, dim_out, 3)]
            last_layers.append(nn.Sequential(*m))

        self.up_layers = nn.ModuleList(last_layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.ReplicationPad2d(1),
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 0, bias=True),
                        self.Norm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=False)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.ReplicationPad2d(1),
                        nn.Conv2d(inchannels, outchannels, 3, 2, 0, bias=True),
                        self.Norm2d(outchannels),
                        nn.ReLU(inplace=False)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, block_expansion, inplanes, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or inplanes != planes * block_expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block_expansion, kernel_size=1, stride=stride, bias=True),
                self.Norm2d(planes * block_expansion)
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block_expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, block, block_expansion, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches, block, block_expansion, num_blocks,
                                     num_inchannels, num_channels, fuse_method, self.Norm2d,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        x_list = [x if self.transitions[0][i] is None else self.transitions[0][i](x)
                  for i in range(self.stage_cfgs[0]['NUM_BRANCHES'])]

        for j in range(self._num_stages - 2):
            y_list = self.stages[j](x_list)
            x_list = []
            for i in range(self.stage_cfgs[j + 1]['NUM_BRANCHES']):
                if self.transitions[j + 1][i] is None:
                    x_list.append(y_list[i])
                else:
                    x_list.append(
                        self.transitions[j + 1][i](y_list[i if i < self.stage_cfgs[j]['NUM_BRANCHES'] else -1]))

        x = self.stages[-1](x_list)
        del y_list
        del x_list

        x = x[::-1]
        y = x[0]
        for i, xi in enumerate(x[1:]):
            y = F.interpolate(y, size=(xi.shape[-2], xi.shape[-1]), mode='bilinear', align_corners=False)
            y = self.up_layers[i](torch.cat((y, xi), 1))
        return torch.tanh(y)  # 限制输出范围到[-1, 1]，确保与CT图像归一化范围一致

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_generator_config(num_stages):
    """Create the hierarchical high-resolution stage configuration."""
    hrnet_cfg = {}
    for i in range(1, num_stages + 1):
        cfg = {}
        cfg['NUM_MODULES'] = 1
        cfg['NUM_BRANCHES'] = i
        cfg['NUM_BLOCKS'] = [3] * i
        cfg['NUM_CHANNELS'] = [16 * 2 ** j for j in range(0, i)]
        cfg['BLOCK'] = 'BASIC'
        cfg['FUSE_METHOD'] = 'SUM'
        hrnet_cfg[f'STAGE{i}'] = cfg
    return hrnet_cfg


def build_rescuecta_generator(num_stages=6, **ie_config):
    """Build the RescueCTA generator."""
    hrnet_config = make_generator_config(num_stages)
    ie_config['num_stages'] = num_stages
    return RescueCTAGenerator(hrnet_config, ie_config)
