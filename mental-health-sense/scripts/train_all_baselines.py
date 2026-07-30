"""
双轨冷启动训练脚本（单人系统）

对被监测的老人执行两轨建档。一轨失败不影响另一轨。

Usage:
    python scripts/train_all_baselines.py
    python scripts/train_all_baselines.py --elder E001
    python scripts/train_all_baselines.py --track sleep     # 只训一轨
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline.scaler_utils import TRACKS
from src.baseline.trainer import train_initial_baseline
from src.utils.logger import get_logger, setup_logger

logger = get_logger(__name__)

DEFAULT_ELDER_ID = "E001"


def main():
    parser = argparse.ArgumentParser(description="双轨冷启动训练")
    parser.add_argument(
        "--elder", type=str, default=DEFAULT_ELDER_ID,
        help=f"老人ID（默认: {DEFAULT_ELDER_ID}）",
    )
    parser.add_argument(
        "--track", type=str, choices=TRACKS, default=None,
        help="只训指定轨；不传则两轨都训",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logger(log_level=args.log_level)

    elder_id = args.elder
    tracks = (args.track,) if args.track else TRACKS

    logger.info("=" * 46)
    logger.info(f"双轨冷启动训练: {elder_id}  轨道={list(tracks)}")
    logger.info("=" * 46)

    failures = []
    for track in tracks:
        try:
            model, scaler, stats, ewma = train_initial_baseline(elder_id, track)
            logger.info(f"  [{track}] 训练成功")
            logger.info(f"    模型参数: {model.count_parameters()}")
            logger.info(
                f"    留出段 signed_std: {stats['signed']['std'].mean():.4f}"
                f"  abs_mean: {stats['abs']['mean'].mean():.4f}"
            )
            logger.info(f"    EWMA: {ewma}")
        except Exception as e:
            logger.error(f"  [{track}] 训练失败: {e}")
            failures.append((track, str(e)))

    logger.info("=" * 46)
    if failures:
        for track, err in failures:
            logger.error(f"失败: [{track}] {err}")
        # 两轨全失败才算脚本失败；单轨失败是可接受的降级
        if len(failures) == len(tracks):
            sys.exit(1)
        logger.warning(f"部分轨道建档失败（{len(failures)}/{len(tracks)}），系统可降级运行")
    else:
        logger.info("两轨建档全部完成")


if __name__ == "__main__":
    main()
