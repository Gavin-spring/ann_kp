# src/utils/logger.py
import logging
import sys
import os
from datetime import datetime

# 我们不再从这里导入配置，因为配置应该作为参数传入
# 这样可以保持工具函数的独立性

def setup_logger(run_name: str, log_dir: str, level=logging.INFO):
    """
    Configures the root logger for the entire application.
    This should be called only ONCE at the application's entry point.
    """
    logger = logging.getLogger() # Get the root logger

    # --- 防止重复配置 ---
    # 如果 root logger 已经有处理器了，说明已经配置过，直接返回
    if logger.hasHandlers():
        return

    logger.setLevel(level)

    # 1. 创建文件处理器 (File Handler)
    # 使用传入的 run_name 和 log_dir 来创建唯一的日志文件
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{run_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_handler.setLevel(logging.DEBUG) # 文件日志记录所有DEBUG及以上级别
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 2. 创建控制台处理器 (Console Handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # 控制台只显示INFO及以上级别
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. All subsequent logs will be saved to: {log_filepath}")

# 为了方便，我们可以保留一个 get_logger 的辅助函数
# 但标准的 logging.getLogger(__name__) 也是完全可以的
def get_logger(name: str) -> logging.Logger:
    """A helper to get a logger instance."""
    return logging.getLogger(name)