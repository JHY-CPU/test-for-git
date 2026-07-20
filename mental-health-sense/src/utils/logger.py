"""
日志配置模块

使用loguru进行结构化日志记录，输出到控制台和文件。
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_dir: str | Path | None = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """
    配置全局日志。

    Args:
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_dir: 日志文件目录，为None时仅输出到控制台
        rotation: 日志文件轮转大小
        retention: 日志保留时间
    """
    # 移除默认handler
    logger.remove()

    # 控制台输出（彩色）
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
    )

    # 文件输出
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 全量日志
        logger.add(
            log_dir / "mental_health_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )

        # 错误日志单独存储
        logger.add(
            log_dir / "error_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )

    logger.info(f"Logger initialized at level {log_level}")


def get_logger(name: str = __name__):
    """
    获取模块级别的logger（通过bind添加模块名）。

    Usage:
        logger = get_logger(__name__)
        logger.info("Processing elder {}", elder_id)
    """
    return logger.bind(name=name)
