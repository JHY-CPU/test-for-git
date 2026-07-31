"""
预警确认脚本：把当前活跃事件标记为「已知晓」

确认之后该事件转入**静默追踪**：同级持续不再通知，但升级与出现新风险类型
**仍会破静默**。语义是"我知道了，别再提醒我这件事"，而不是"关掉这个老人的
所有告警"——后者会让系统在真正恶化时也保持沉默。

Usage:
    python scripts/ack_alert.py --elder E001
    python scripts/ack_alert.py --elder E001 --channel depression
    python scripts/ack_alert.py --elder E001 --undo      # 撤销确认
    python scripts/ack_alert.py --elder E001 --show      # 只看状态，不改

> 真实推送通道接入后，App 侧的"已知晓"按钮回写的就是这同一个状态，
> 本脚本是它的手工等价物。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.risk.alert_state import (
    CHANNELS,
    CHANNEL_BASELINE,
    acknowledge,
    event_age_days,
    load_state,
)
from src.utils.logger import get_logger, setup_logger

logger = get_logger(__name__)

DEFAULT_ELDER_ID = "E001"

_CHANNEL_LABELS = {
    "baseline": "睡眠 / 社会连接（GRU 个人基线）",
    "depression": "情绪状态（MPDD 群体基线）",
}


def _print_state(elder_id: str) -> None:
    state = load_state(elder_id)
    logger.info(f"预警事件状态: {elder_id}")
    for name in CHANNELS:
        ch = state["channels"][name]
        label = _CHANNEL_LABELS.get(name, name)
        if not ch.get("active"):
            logger.info(f"  [{label}] 无活跃事件")
            continue
        age = event_age_days(ch, ch.get("last_seen_day") or "")
        logger.info(
            f"  [{label}] L{ch['level']} 持续中"
            + (f"，第 {age} 天" if age else "")
            + f"（{ch.get('started_day')} 起）"
        )
        logger.info(
            f"      风险类型: {ch.get('risk_keys') or '—'}；"
            f"已通知 {ch.get('notify_count', 0)} 次"
            f"（最近 {ch.get('last_notified_day') or '—'}）；"
            f"已知晓: {'是' if ch.get('acknowledged') else '否'}"
        )


def main():
    parser = argparse.ArgumentParser(description="确认预警事件（转静默追踪）")
    parser.add_argument(
        "--elder", type=str, default=DEFAULT_ELDER_ID,
        help=f"老人ID（默认: {DEFAULT_ELDER_ID}）",
    )
    parser.add_argument(
        "--channel", type=str, default=CHANNEL_BASELINE, choices=list(CHANNELS),
        help="事件流：baseline（睡眠/社会连接）或 depression（情绪状态）",
    )
    parser.add_argument("--undo", action="store_true", help="撤销确认，恢复正常通知")
    parser.add_argument("--show", action="store_true", help="只显示状态，不做修改")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logger(log_level=args.log_level)

    if args.show:
        _print_state(args.elder)
        return

    result = acknowledge(args.elder, channel=args.channel, undo=args.undo)

    if result is None:
        logger.warning(
            f"{args.elder} 的 [{_CHANNEL_LABELS.get(args.channel, args.channel)}] "
            f"当前没有活跃事件，无需确认"
        )
        sys.exit(0)

    if args.undo:
        logger.info(f"已撤销确认：{args.elder} / {args.channel}，恢复正常通知")
    else:
        logger.info(
            f"已确认：{args.elder} / {args.channel} 的当前事件转入静默追踪。\n"
            f"  同级持续不再通知；**升级与新风险类型仍会破静默**。"
        )

    _print_state(args.elder)


if __name__ == "__main__":
    main()
