"""预警事件状态：把「每日状态」换成「事件」

★ 本模块解决的问题

  判定层每天都会给出一个 risk_level，而 `alert.trigger_alert` 原本对每一个
  >= L1 的日子都无条件发一次通知。它没有任何记忆——不知道昨天发过没有，
  连"上次"这个概念都不存在。于是**把"状态持续"当成了"每天都是新事件"**。

  实测代价（重建后的 E001 60 天数据）：
      子女 App 推送 12 次，而真实事件只有 4 个 —— 三分之二是重复
  极端情况（PERM_step，老人永久性衰退）：
      连续 91 天每天判 L3 = 91 次短信 + 91 次强制响铃 + 91 次惊动网格员

  后果不是"吵"，是**系统最终变哑**：家属一周内关掉通知，此后真正的新变化
  也收不到了。方向闸门、幅度门槛、持续性统计——前面所有为压低误报做的工作，
  在通知被关掉的那一刻全部失去出口。这比漏报更快摧毁信任。

★ 检测层是对的，缺口在策略层

  老人确实一直处于低效睡眠状态，每天判 L3 在事实层面没错。错在通知层把
  "判定"与"通知"当成了同一件事。**判定要每天做，通知不该每天发。**
  所以本模块不碰 judge.py / rules.py 一行。

★ 事件模型

  事件 = (elder_id, channel) 上的一段连续风险期。
  channel ∈ {"baseline"(GRU 双轨), "depression"(MPDD)}，两条流**独立计数、
  独立冷却、互不压制**——理由同"绝不跨轨比较绝对分"：它们不是同一件事，
  不该共用一个计数器。

  只在**状态发生变化**时通知：开始 / 升级 / 出现新风险类型 / 缓解。
  同级持续不重复通知，改由周报的"预警回执"承担告知义务。

★ 最关键的一条不变量：升级必须穿透一切抑制

  纯粹按"同一等级 X 天内最多一次"做冷却会出现：
      D1 L2 → 发通知，进入冷却
      D3 L2 → 冷却中，不发   ✓ 对
      D5 L3 → 冷却中，不发   ✗ 恶化被冷却窗吃掉
  这是拿误报换漏报，比修复前更糟。`decide()` 里升级分支排在所有抑制判据
  **之前**，且无视 acknowledged。tests/test_alert_events.py 有专门的守门员用例。

★ 为什么状态必须落盘

  这是每日批处理，进程每天起一次就退，内存态活不过今天。与 EWMA 的
  `freeze_streak` / `last_day_key` 需要持久化是完全同一个理由
  （见 ewma.py 的 `_state_filename`）。沿用那边的四条容错不变量：
  状态另存小文件 / 原子写 / 缺文件按默认起算 / 损坏则重置该 channel 并继续。
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.io import atomic_write_json, get_log_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = "1.0.0"
LOG_TYPE = "alert_state"
DATE_FMT = "%Y-%m-%d"

# 两条独立的事件流
CHANNEL_BASELINE = "baseline"      # GRU 双轨（睡眠障碍 / 孤独）
CHANNEL_DEPRESSION = "depression"  # MPDD 群体基线（抑郁）
CHANNELS = (CHANNEL_BASELINE, CHANNEL_DEPRESSION)

# 状态迁移类型。前四个会通知，后四个不会。
TRANSITION_STARTED = "started"          # 事件开始
TRANSITION_ESCALATED = "escalated"      # 等级上升 —— 穿透一切抑制
TRANSITION_NEW_TYPE = "new_type"        # 出现新的风险类型
TRANSITION_RESOLVED = "resolved"        # 缓解（回到 L0）
TRANSITION_REPEAT = "repeat"            # 同级持续，超过 repeat_days 的重提

TRANSITION_IMPROVED = "improved"        # 等级下降但仍 >=1，不通知
TRANSITION_COOLDOWN = "cooldown"        # 同级持续，冷却中
TRANSITION_ACKNOWLEDGED = "acknowledged"  # 已人工确认，转静默追踪
TRANSITION_NONE = "none"                # 无事件、无变化

NOTIFYING_TRANSITIONS = frozenset({
    TRANSITION_STARTED,
    TRANSITION_ESCALATED,
    TRANSITION_NEW_TYPE,
    TRANSITION_RESOLVED,
    TRANSITION_REPEAT,
})


def get_alert_state_dir() -> Path:
    """状态目录。复用 io.get_log_dir，不另造路径函数——路径推导必须只有一处。

    测试通过 monkeypatch 本函数把状态重定向到 tmp（见 tests/conftest.py 的
    autouse fixture）。不隔离的话，跑一次 pytest 就会往真实 data/logs/ 写状态
    并**跨会话残留**，第二次跑测试时状态已存在，结果不可复现——
    本仓"不可复现的绿色比红色更危险"。
    """
    return get_log_dir(LOG_TYPE)


def _empty_channel() -> dict:
    return {
        "active": False,
        "level": 0,
        "risk_keys": [],
        "started_day": None,
        "last_seen_day": None,
        "last_notified_day": None,
        "last_notified_level": 0,
        "notify_count": 0,
        "acknowledged": False,
        "acknowledged_at": None,
    }


def _empty_state(elder_id: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "elder_id": elder_id,
        "channels": {c: _empty_channel() for c in CHANNELS},
    }


def load_state(elder_id: str) -> dict:
    """读状态。缺文件按默认起算；损坏则重置并继续，绝不抛。

    容错策略照抄 ewma.TrackEWMAPools.load：预警状态坏掉的后果是"退回到没有
    冷却的行为"（多发几条通知），而抛异常的后果是**整条日管道当天失败**、
    老人零监测。两害相权，降级明显更轻。
    """
    path = get_alert_state_dir() / f"{elder_id}.json"
    if not path.exists():
        return _empty_state(elder_id)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"预警状态损坏，按默认重置: {path.name} ({e})")
        return _empty_state(elder_id)

    state = _empty_state(elder_id)
    if not isinstance(raw, dict):
        return state

    channels = raw.get("channels")
    if isinstance(channels, dict):
        for name in CHANNELS:
            ch = channels.get(name)
            if isinstance(ch, dict):
                # 逐键合并而非整体替换：老状态文件缺新字段时按默认补齐，
                # 向后兼容（同 ewma.from_dict 的 data.get(..., 默认) 写法）
                merged = _empty_channel()
                merged.update({k: v for k, v in ch.items() if k in merged})
                state["channels"][name] = merged
            elif ch is not None:
                logger.error(f"预警状态的 {name} 通道结构异常，已重置该通道")
    return state


def save_state(state: dict) -> Path:
    """原子写。状态坏一半比没有更糟——下次读到半截 JSON 会整个重置。"""
    path = get_alert_state_dir() / f"{state['elder_id']}.json"
    atomic_write_json(path, state)
    return path


def _days_between(later: str | None, earlier: str | None) -> int | None:
    """两个 day_key 相差几个自然日。任一不可解析时返回 None。"""
    if not later or not earlier:
        return None
    try:
        return (datetime.strptime(later, DATE_FMT)
                - datetime.strptime(earlier, DATE_FMT)).days
    except ValueError:
        return None


def decide(
    channel_state: dict,
    day_key: str,
    risk_level: int,
    risk_keys: list[str],
    events_cfg: dict,
    repeat_days: int,
    max_skip_days: int = 3,
) -> tuple[str, bool]:
    """给定当前事件状态与今天的判定，决定发不发通知。

    ★ 纯函数：不读文件、不写文件、不看时钟。这样它能被穷举测试，
      而"要不要发通知"这件事的全部逻辑都集中在一处可审查的地方。

    Returns:
        (transition, notify)

    判定顺序即优先级，**升级排在所有抑制判据之前**（见模块 docstring）。
    """
    active = bool(channel_state.get("active"))
    prev_level = int(channel_state.get("level", 0))
    prev_keys = set(channel_state.get("risk_keys") or [])
    cur_keys = set(risk_keys or [])

    notify_on_resolve = events_cfg.get("notify_on_resolve", True)
    new_type_cooldown = int(events_cfg.get("new_type_cooldown_days", 3))

    gap = _days_between(day_key, channel_state.get("last_seen_day"))

    # 0. ★ 补算**早于**当前事件已观测区间的历史日：一律不动状态、不通知。
    #
    #    这条必须排在最前面，连"回到正常"都要让位。理由：事件状态描述的是
    #    "此刻这个老人处于什么状况"，而一条更早的历史判定不携带任何关于"此刻"
    #    的信息——它既不能宣布缓解，也不能宣布恶化。
    #
    #    实测（补算一个 7 月的正常日，而 8 月正有活跃 L3）：
    #        2026-08-10 L3 → started (通知)
    #        2026-08-11/12 L3 → cooldown
    #        补算 2026-07-15 L0 → resolved，active=False
    #        状态被清空：started_day=null、notify_count=0
    #        2026-08-13 L3 → started (**又发一次通知**)
    #    一次"补算上个月漏掉的某天"就把正在持续的 L3 事件整个抹掉：周报的
    #    「预警回执」不再显示它、事件年龄归零、次日重发"开始"通知。
    #
    #    反方向同样坏：补算一个更早的高等级日会被判成 escalated 并**真的发出
    #    强制响铃 + 惊动网格员**，而事件的 started_day 仍指向 8 月，
    #    落盘后 last_seen_day(7/21) < started_day(8/10)，状态自相矛盾。
    #
    #    `decide` 的第 3 条说"同一天重算出更高等级是真实的升级"——那个论证只对
    #    gap == 0 成立（同一天的判定结果变了），gap < 0 是搭了便车。
    #    这与 ewma.update 用 `day_key <= last_day_key` 拒收乱序补算是同一条原则：
    #    "往回补一天会把早已过去的分当成最新观测"。
    if active and gap is not None and gap < 0:
        return TRANSITION_NONE, False

    # 1. 回到正常：事件结束
    if risk_level <= 0:
        if active:
            return TRANSITION_RESOLVED, bool(notify_on_resolve)
        return TRANSITION_NONE, False

    # 2. 事件开始：没有活跃事件，或与上次可信观测断开太久
    #
    #    断开判据复用 max_skip_days（与 continuity.walk_back_days 一致）：
    #    设备离线一段时间后，不该把两段无关的风险期粘成同一个事件——
    #    那正是"5 条日志跨 22 个日历日仍数出 consecutive=5"的同型错误。
    if not active or gap is None or gap > max_skip_days + 1:
        return TRANSITION_STARTED, True

    # 3. ★ 升级：穿透冷却、穿透 acknowledged、也穿透下面的同日幂等分支
    #
    #    这是整个模块最重要的一条。纯冷却方案会在 L2 的冷却窗内吃掉 L3，
    #    等于拿误报换漏报。恶化永远是新信息。
    #
    #    必须排在同日幂等（gap <= 0）**之前**：同一天重算出更高的等级，
    #    说明判定结果变了（修了 bug、补了数据），那是真实的升级，
    #    不是"重复的同一件事"。放在后面会让重算把恶化静默掉。
    if risk_level > prev_level:
        return TRANSITION_ESCALATED, True

    # 4. 补算/重跑同一天且等级未上升：幂等，不产生新通知也不推进计数。
    if gap <= 0:
        return TRANSITION_NONE, False

    # 5. 出现新的风险类型：也是新信息，穿透 acknowledged，但受一个短冷却约束
    #    （防止类型在门槛附近抖动时反复响）
    if cur_keys - prev_keys:
        since_notify = _days_between(day_key, channel_state.get("last_notified_day"))
        if since_notify is None or since_notify >= new_type_cooldown:
            return TRANSITION_NEW_TYPE, True
        return TRANSITION_COOLDOWN, False

    # 6. 等级下降但仍在风险中：不通知。
    #    只有回到 L0 才算"缓解"——从 L3 掉到 L2 仍然需要关注，
    #    这时发一条"好转了"的通知会让家属误以为可以放松。
    if risk_level < prev_level:
        return TRANSITION_IMPROVED, False

    # 7~9. 同级持续
    if channel_state.get("acknowledged"):
        return TRANSITION_ACKNOWLEDGED, False

    if repeat_days <= 0:
        # 不重复通知。告知义务由周报的"预警回执"承担——彻底静默会让
        # "系统安静"与"系统挂了"无法区分，那是另一种失效。
        return TRANSITION_COOLDOWN, False

    since_notify = _days_between(day_key, channel_state.get("last_notified_day"))
    if since_notify is None or since_notify >= repeat_days:
        return TRANSITION_REPEAT, True
    return TRANSITION_COOLDOWN, False


def apply(
    channel_state: dict,
    day_key: str,
    risk_level: int,
    risk_keys: list[str],
    transition: str,
    notified: bool,
) -> dict:
    """把本次判定的结果落到事件状态上，返回新的 channel 状态（不就地改）。"""
    new = dict(channel_state)

    if transition == TRANSITION_NONE and not new.get("active"):
        return new
    if transition == TRANSITION_NONE and new.get("active"):
        # 补算/重跑同一天：状态原样不动（幂等）
        return new

    if transition == TRANSITION_RESOLVED:
        # 事件关闭。保留 acknowledged=False 以便下一次事件从干净状态开始。
        return _empty_channel()

    if transition == TRANSITION_STARTED:
        new = _empty_channel()
        new["started_day"] = day_key

    new["active"] = True
    new["level"] = int(risk_level)
    # 风险类型取并集：事件期内出现过的都记着，用于判断"有没有新类型"。
    # 用并集而非覆盖，是因为类型可能在门槛附近抖动，用覆盖会让同一个类型
    # 反复被当成"新出现"。
    new["risk_keys"] = sorted(set(new.get("risk_keys") or []) | set(risk_keys or []))
    new["last_seen_day"] = day_key

    if notified:
        new["last_notified_day"] = day_key
        new["last_notified_level"] = int(risk_level)
        new["notify_count"] = int(new.get("notify_count", 0)) + 1

    if transition == TRANSITION_ESCALATED:
        # 升级视为新的告知起点：之前的"已知晓"是针对更低等级的，
        # 不能让它把更严重的情况一并静默掉。
        new["acknowledged"] = False
        new["acknowledged_at"] = None

    return new


def acknowledge(elder_id: str, channel: str = CHANNEL_BASELINE,
                undo: bool = False) -> dict | None:
    """把当前活跃事件置为"已知晓"（或撤销）。

    确认之后该事件转入静默追踪：同级持续不再通知，但**升级与新风险类型
    仍会破静默**（见 decide 的第 3、4 条）。这是"我知道了，别再提醒我这件事"
    而不是"关掉这个老人的所有告警"。

    Returns:
        更新后的 channel 状态；没有活跃事件时返回 None。
    """
    if channel not in CHANNELS:
        raise ValueError(f"Unknown channel: {channel!r}. Expected one of {list(CHANNELS)}")

    state = load_state(elder_id)
    ch = state["channels"][channel]
    if not ch.get("active"):
        return None

    ch["acknowledged"] = not undo
    ch["acknowledged_at"] = None if undo else datetime.now().isoformat(timespec="seconds")
    save_state(state)
    return ch


def active_events(elder_id: str) -> dict[str, dict]:
    """当前所有活跃事件（供周报回执使用）。"""
    state = load_state(elder_id)
    return {
        name: ch for name, ch in state["channels"].items()
        if ch.get("active")
    }


def event_age_days(channel_state: dict, as_of: str) -> int | None:
    """事件已持续几天（含首日）。"""
    gap = _days_between(as_of, channel_state.get("started_day"))
    return None if gap is None else gap + 1
