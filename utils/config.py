"""配置加载和管理"""

import yaml
import os


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config, save_path):
    """保存配置到YAML文件"""
    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def make_dirs(*args):
    """创建多个目录"""
    for dir_path in args:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def set_save_dirs(base, name):
    """设置并创建保存目录"""
    model_save_path = os.path.join(base, 'weights', name)
    draw_path = os.path.join(base, 'draw', name)
    summary_log_dir = os.path.join(base, 'log', name)
    temp_pic_dir = os.path.join(base, 'temp_pic', name)
    val_pic_dir = os.path.join(base, 'val_pic', name)
    # draw_path is kept for backward compatibility but is intentionally not created here.
    # (This project doesn't currently use it, and creating it causes confusing "unused" folders.)
    make_dirs(model_save_path, summary_log_dir, temp_pic_dir, val_pic_dir)
    return model_save_path, draw_path, summary_log_dir, temp_pic_dir, val_pic_dir
