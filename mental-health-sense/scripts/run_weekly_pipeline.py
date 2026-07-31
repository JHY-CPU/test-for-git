"""
每周趋势轨脚本（单人系统）

手动或由 cron/systemd 触发：双轨微调 + 生成周报。

此前 run_weekly_pipeline 零调用方——src/、scripts/、tests/ 里都没有，
而 docs/TRAINING.md 写着"每周日 03:00（weekly_job）"。整条周轨事实上
没有入口，只能靠手动 import 模块才能跑。

Usage:
    python scripts/run_weekly_pipeline.py
    python scripts/run_weekly_pipeline.py --elder E001
    python scripts/run_weekly_pipeline.py --no-retrain   # 只出周报，不动模型
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scheduler.weekly_job import run_weekly_pipeline
from src.utils.logger import get_logger, setup_logger

logger = get_logger(__name__)

# 被监测老人的默认 ID（单人系统）
DEFAULT_ELDER_ID = "E001"


def main():
    parser = argparse.ArgumentParser(description="每周趋势轨（微调 + 周报）")
    parser.add_argument(
        "--elder", type=str, default=DEFAULT_ELDER_ID,
        help=f"老人ID（默认: {DEFAULT_ELDER_ID}）",
    )
    parser.add_argument(
        "--no-retrain", action="store_true",
        help="跳过 GRU 微调，只生成周报（排查周报问题时用，避免动基线）",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logger(log_level=args.log_level)

    logger.info("=" * 46)
    logger.info(f"每周流程: {args.elder}"
                f"{'（跳过微调）' if args.no_retrain else ''}")
    logger.info("=" * 46)

    try:
        result = run_weekly_pipeline(
            elder_id=args.elder, retrain=not args.no_retrain
        )

        logger.info(f"  周期: {result['week_start']} ~ {result['week_end']}")
        logger.info(f"  微调: {result['retrain_status']}")
        logger.info(f"  风险等级: {result['risk_label']}")
        if result.get("report_path"):
            logger.info(f"  周报: {result['report_path']}")
        else:
            logger.warning("  周报未生成（见上方错误）")

    except Exception as e:
        logger.error(f"  {args.elder} 每周流程失败: {e}")
        raise

    logger.info("=" * 46)
    logger.info("每周流程完成")


if __name__ == "__main__":
    main()
