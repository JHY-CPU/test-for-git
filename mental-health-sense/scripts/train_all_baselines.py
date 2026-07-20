"""
批量冷启动训练脚本

对模拟数据中所有老人执行 Day14 冷启动训练。

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


def main():
    parser = argparse.ArgumentParser(description="批量冷启动训练")
    parser.add_argument("--elder", type=str, help="指定老人ID（不指定则全部训练）")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logger(log_level=args.log_level)

    # 默认训练全部5位模拟老人
    elder_ids = ["E001", "E002", "E003", "E004", "E005"]
    if args.elder:
        elder_ids = [args.elder]

    logger.info(f"开始冷启动训练: {len(elder_ids)}位老人")

    success_count = 0
    fail_count = 0

    for elder_id in elder_ids:
        try:
            logger.info(f"\n{'=' * 40}")
            logger.info(f"训练: {elder_id}")
            logger.info(f"{'=' * 40}")

            model, scaler, stats, ewma = train_initial_baseline(elder_id)

            logger.info(f"  ✅ {elder_id} 训练成功")
            logger.info(f"    模型参数: {model.count_parameters()}")
            logger.info(f"    残差均值: {stats['mean'].mean():.4f}")
            logger.info(f"    残差标准差: {stats['std'].mean():.4f}")
            logger.info(f"    EWMA n={ewma.n}, mean={ewma.mean:.4f}")

            success_count += 1

        except Exception as e:
            logger.error(f"  ❌ {elder_id} 训练失败: {e}")
            fail_count += 1

    logger.info(f"\n{'=' * 40}")
    logger.info(f"训练完成: 成功{success_count}, 失败{fail_count}")
    logger.info(f"{'=' * 40}")


if __name__ == "__main__":
    main()
