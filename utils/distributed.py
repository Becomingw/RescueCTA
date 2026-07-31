"""分布式训练工具模块"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_dist_available():
    """检查分布式训练是否可用"""
    return dist.is_available()


def is_initialized():
    """检查分布式训练是否已初始化"""
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    """获取总进程数"""
    if not is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """获取当前进程的全局 rank"""
    if not is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    """获取当前进程的本地 rank（同一节点内）"""
    if not is_initialized():
        return 0
    # 优先从环境变量获取
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return dist.get_rank()


def is_main_process():
    """检查是否是主进程（rank=0）"""
    return get_rank() == 0


def setup_distributed(backend='nccl', init_method='env://'):
    """初始化分布式训练环境

    支持两种启动方式：
    1. torchrun / torch.distributed.launch: 自动设置环境变量
    2. 手动设置: 需要设置 MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE

    Args:
        backend: 通信后端，GPU 训练推荐 'nccl'
        init_method: 初始化方法

    Returns:
        local_rank: 本地 GPU 编号
        world_size: 总进程数
    """
    if is_initialized():
        return get_local_rank(), get_world_size()

    # 检查是否通过 torchrun 启动
    if 'RANK' not in os.environ:
        print("Not running in distributed mode, using single GPU")
        return 0, 1

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank))

    # 设置当前进程使用的 GPU
    torch.cuda.set_device(local_rank)

    # 初始化进程组
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )

    # 同步所有进程
    dist.barrier()

    if is_main_process():
        print(f"Distributed training initialized: world_size={world_size}, backend={backend}")

    return local_rank, world_size


def cleanup_distributed():
    """清理分布式训练环境"""
    if is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor, world_size=None, reduce_op=dist.ReduceOp.SUM):
    """在所有进程间 reduce tensor

    Args:
        tensor: 要 reduce 的 tensor
        world_size: 进程数，用于求平均
        reduce_op: reduce 操作类型

    Returns:
        reduced tensor
    """
    if not is_initialized():
        return tensor

    if world_size is None:
        world_size = get_world_size()

    rt = tensor.clone()
    dist.all_reduce(rt, op=reduce_op)

    if reduce_op == dist.ReduceOp.SUM:
        rt /= world_size

    return rt


def gather_tensor(tensor):
    """从所有进程收集 tensor 到主进程

    Args:
        tensor: 要收集的 tensor

    Returns:
        List of tensors (only on rank 0), None on other ranks
    """
    if not is_initialized():
        return [tensor]

    world_size = get_world_size()

    if is_main_process():
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, gather_list=gathered, dst=0)
        return gathered
    else:
        dist.gather(tensor, dst=0)
        return None


def broadcast_tensor(tensor, src=0):
    """从源进程广播 tensor 到所有进程

    Args:
        tensor: 要广播的 tensor
        src: 源进程 rank
    """
    if not is_initialized():
        return tensor

    dist.broadcast(tensor, src=src)
    return tensor


def wrap_model_ddp(model, device_ids=None, find_unused_parameters=False):
    """将模型包装为 DDP

    Args:
        model: PyTorch 模型
        device_ids: GPU 设备 ID 列表
        find_unused_parameters: 是否查找未使用的参数（某些模型需要）

    Returns:
        DDP 包装的模型
    """
    if not is_initialized():
        return model

    local_rank = get_local_rank()

    if device_ids is None:
        device_ids = [local_rank]

    return DDP(
        model,
        device_ids=device_ids,
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters
    )


def unwrap_model(model):
    """从 DDP/DataParallel 包装中获取原始模型

    Args:
        model: 可能被包装的模型

    Returns:
        原始模型
    """
    if isinstance(model, (DDP, torch.nn.DataParallel)):
        return model.module
    return model


def save_on_main_only(state_dict, path):
    """只在主进程保存 checkpoint

    Args:
        state_dict: 要保存的状态字典
        path: 保存路径
    """
    if is_main_process():
        torch.save(state_dict, path)

    # 确保所有进程等待保存完成
    if is_initialized():
        dist.barrier()


def print_on_main_only(*args, **kwargs):
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)


class DistributedSamplerWrapper:
    """分布式采样器的简单封装，用于兼容非分布式场景"""

    def __init__(self, dataset, shuffle=True, seed=42):
        from torch.utils.data import DistributedSampler

        self.distributed = is_initialized()

        if self.distributed:
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=shuffle,
                seed=seed
            )
        else:
            self.sampler = None

    def set_epoch(self, epoch):
        """设置当前 epoch（用于 shuffle）"""
        if self.distributed and self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def get_sampler(self):
        """获取采样器（非分布式时返回 None）"""
        return self.sampler
