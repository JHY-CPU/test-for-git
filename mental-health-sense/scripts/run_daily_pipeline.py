"""
每日推理脚本（单人系统）

手动触发被监测老人的每日推理管道。

Usage:
    python scripts/run_daily_pipeline.py
    python scripts/run_daily_pipeline.py --date 2026-08-01
    python scripts/run_daily_pipeline.py --elder E001
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scheduler.daily_job import run_daily_pipeline
from src.risk.judge import quick_judge
from src.risk.alert import trigger_alert
from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)

# 被监测老人的默认 ID（单人系统）
DEFAULT_ELDER_ID = "E001"


def main():
    parser = argparse.ArgumentParser(description="每日推理管道")
    parser.add_argument("--date", type=str, help="日期 (YYYY-MM-DD)，默认昨天")
    parser.add_argument(
        "--elder", type=str, default=DEFAULT_ELDER_ID,
        help=f"老人ID（默认: {DEFAULT_ELDER_ID}）",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logger(log_level=args.log_level)

    elder_id = args.elder
    date_str = args.date

    logger.info(f"{'=' * 40}")
    logger.info(f"每日推理: {elder_id}, 日期={date_str or '昨天'}")
    logger.info(f"{'=' * 40}")

    try:
        result = run_daily_pipeline(
            elder_id=elder_id,
            date_str=date_str,
        )

        logger.info(f"  数据质量: {result['data_quality']}")
        logger.info(f"  管道状态: {result['status']}")

        if result.get("inference_result"):
            inf = result["inference_result"]
            logger.info(f"  异常分: {inf['anomaly_score']:.4f}")
            logger.info(f"  阈值: {inf['dynamic_threshold']:.4f}")
            logger.info(f"  偏离: {inf['is_deviation']}")

        if result.get("risk_result"):
            risk = result["risk_result"]
            logger.info(f"  风险等级: {risk['risk_level']}({risk['risk_label']})")
            if risk.get("recommendation"):
                logger.info(f"  建议: {risk['recommendation']}")

    except Exception as e:
        logger.error(f"  ❌ {elder_id} 推理失败: {e}")
        raise

    logger.info(f"\n{'=' * 40}")
    logger.info("每日推理完成")


if __name__ == "__main__":
    main()
