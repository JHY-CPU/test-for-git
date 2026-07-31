"""
抑郁评估脚本（MPDD 旁路通道）

★ 刻意**不挂进每日管道**。抑郁评估是事件驱动的——只有拿到"有正脸 + 有连续语音"
  的录像片段时才有意义，而独居老人这种时段稀少（可能子女来访才有）。挂进 03:00
  的日批处理只会每天产生一条"今天没视频"的噪音，还要在主链路里处理一堆与睡眠、
  社交无关的失败分支。日管道的职责边界不该被这个稀疏事件撑开。

Usage:
    # 设备到货前：手工传录像
    python scripts/run_depression_assessment.py --elder E001 --date 2026-08-12 \
        --video /path/to/footage.mp4

    # 设备到货后：自动取当天录像（StreamClipSource，尚未实现）
    python scripts/run_depression_assessment.py --elder E001 --date 2026-08-12

前置条件：
    data/depression/{elder_id}/description.txt   个人介绍（英文，冻结不改）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.depression.runner import get_description_path, run_depression_assessment
from src.utils.logger import get_logger, setup_logger

logger = get_logger(__name__)

DEFAULT_ELDER_ID = "E001"


def main():
    parser = argparse.ArgumentParser(description="抑郁评估（MPDD 群体基线，旁路通道）")
    parser.add_argument("--date", type=str, help="day_key (YYYY-MM-DD)，默认今天")
    parser.add_argument(
        "--elder", type=str, default=DEFAULT_ELDER_ID,
        help=f"老人ID（默认: {DEFAULT_ELDER_ID}）",
    )
    parser.add_argument("--video", type=str, help="录像文件路径（设备到货前必填）")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logger(log_level=args.log_level)

    logger.info("=" * 46)
    logger.info(f"抑郁评估: {args.elder}, day_key={args.date or '今天'}")
    logger.info("=" * 46)

    description_file = get_description_path(args.elder)
    if not description_file.is_file():
        logger.error(
            f"缺少个人介绍文本: {description_file}\n"
            f"  MPDD 的 A-V+P 模型需要它作为第三路模态输入（编码成 1024 维 RoBERTa 嵌入）。\n"
            f"  建议用简洁英文描述老人的日常状态与表达方式。\n"
            f"  ★ 一旦确定必须冻结——改一个字分数就会变，而与老人实际状态无关。"
        )
        sys.exit(1)

    result = run_depression_assessment(
        elder_id=args.elder, day_key=args.date, video=args.video
    )

    if not result:
        logger.warning("抑郁评估未启用（config: depression.enabled=false）")
        sys.exit(0)

    logger.info(f"  状态: {result['status']}")
    outcome = result.get("result") or {}
    if outcome:
        logger.info(f"  等级: {outcome.get('level')}")
        logger.info(
            f"  PHQ-9: {outcome.get('phq9_median')} "
            f"(段间极差 {outcome.get('phq9_spread')})"
        )
        logger.info(f"  类别概率: {outcome.get('class_probs')}")
    for reason in result.get("low_confidence_reasons", []):
        logger.warning(f"  证据说明: {reason}")

    ev = result.get("evidence", {})
    logger.info(f"  片段: 采用 {ev.get('n_clips', 0)}，丢弃 {ev.get('n_rejected', 0)}")
    logger.info(f"  有效期至: {result.get('valid_until')}")
    logger.warning(f"  ⚠️ {result['calibration']['warning']}")

    logger.info("=" * 46)

    # 失败状态以非零码退出，便于 cron / 上层脚本发现。
    # 这与 daily_job 那条"status 恒 success 掩盖了推理失败"的教训是同一个道理：
    # 全绿的退出码不该盖住真实故障。
    if result["status"] in ("failed",):
        sys.exit(1)


if __name__ == "__main__":
    main()
