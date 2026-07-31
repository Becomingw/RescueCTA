#!/usr/bin/env python
"""RescueCTA training entry point.

支持两种训练模式：
1. 单卡训练:
   python train.py --config configs/xxx.yaml

2. DDP 多卡训练:
   python train.py --config configs/xxx.yaml --gpus 0,1,2,3
   # 或在配置文件中设置 multi_gpu.use_gpus: [0, 1, 2, 3]
"""

import os
import argparse
import socket
from datetime import timedelta


def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Best-effort check whether a TCP port is available for listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, int(port)))
        except OSError:
            return False
        return True


def _pick_free_port() -> int:
    """Pick an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

def main():
    parser = argparse.ArgumentParser(description="RescueCTA training")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--gpus', type=str, default=None,
                        help='GPU IDs to use (e.g., "0,1,2,3"). Overrides config file.')
    # DDP 内部参数（由子进程自动设置）
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DDP (internal use)')
    args = parser.parse_args()

    # 加载配置
    from utils import load_config
    config = load_config(args.config)

    # 解析 GPU 列表
    if args.gpus is not None:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        gpu_ids = config.get('multi_gpu', {}).get('use_gpus', [0])

    # 如果是子进程（已经在 DDP 环境中），直接运行训练
    if 'RANK' in os.environ:
        run_training(config, args)
        return

    # 主进程：根据 GPU 数量决定启动方式
    if len(gpu_ids) > 1:
        # 多卡：使用 multiprocessing 启动 DDP
        launch_ddp(args, gpu_ids)
    else:
        # 单卡：直接训练
        if gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
        run_training(config, args)


def launch_ddp(args, gpu_ids):
    """使用 torch.multiprocessing 启动 DDP 训练"""
    import torch.multiprocessing as mp

    world_size = len(gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')

    # Pick a free rendezvous port to avoid EADDRINUSE.
    if 'MASTER_PORT' not in os.environ:
        default_port = 29500
        os.environ['MASTER_PORT'] = str(default_port if _is_port_free(default_port) else _pick_free_port())

    print(f"Launching DDP training on GPUs: {gpu_ids} (MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']})")
    mp.spawn(
        ddp_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


def ddp_worker(local_rank, world_size, args):
    """DDP 子进程工作函数"""
    import torch
    import torch.distributed as dist

    # 设置环境变量
    os.environ['RANK'] = str(local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')

    # 设置当前进程使用的 GPU
    torch.cuda.set_device(local_rank)

    # 初始化进程组
    # Increase timeout to tolerate long validation/checkpoint on rank0.
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=timedelta(minutes=60),
        device_id=torch.device(f'cuda:{local_rank}')  # 消除barrier警告
    )

    # 加载配置并运行训练
    from utils import load_config
    config = load_config(args.config)
    run_training(config, args)

    # 清理
    dist.destroy_process_group()


def run_training(config, args):
    """运行训练"""
    from engine import Trainer

    # 检查是否是主进程
    is_main = os.environ.get('RANK', '0') == '0'

    if is_main:
        print(f"配置参数: {config}")

    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train(continue_train=args.resume)


if __name__ == '__main__':
    main()
