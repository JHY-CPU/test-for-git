"""
冷启动训练脚本（单人系统）

对被监测的老人执行 Day14 冷启动训练。

Usage:
    python scripts/train_all_baselines.py
    python scripts/train_all_baselines.py --elder E001
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline.trainer import train_initial_baseline
from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)

# 被监测老人的默认 ID（单人系统）
DEFAULT_ELDER_ID = "E001"


def main():
    parser = argparse.ArgumentParser(description="冷启动训练")
    parser.add_argument(
        "--elder", type=str, default=DEFAULT_ELDER_ID,
        help=f"老人ID（默认: {DEFAULT_ELDER_ID}）",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logger(log_level=args.log_level)

    elder_id = args.elder

    logger.info(f"{'=' * 40}")
    logger.info(f"冷启动训练: {elder_id}")
    logger.info(f"{'=' * 40}")

    try:
        model, scaler, stats, ewma = train_initial_baseline(elder_id)

        logger.info(f"  ✅ {elder_id} 训练成功")
        logger.info(f"    模型参数: {model.count_parameters()}")
        logger.info(f"    残差均值: {stats['mean'].mean():.4f}")
        logger.info(f"    残差标准差: {stats['std'].mean():.4f}")
        logger.info(f"    EWMA n={ewma.n}, mean={ewma.mean:.4f}")
    except Exception as e:
        logger.error(f"  ❌ {elder_id} 训练失败: {e}")
        raise


if __name__ == "__main__":
    main()
