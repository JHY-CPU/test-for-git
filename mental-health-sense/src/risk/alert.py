"""
三级预警推送模块（按事件去重）

预警等级（全部基于趋势检测）：
    一级「关注」→ 写入周报，不打扰老人（单日偏离）
    二级「提醒」→ 子女App推送 + 周报重点标注（连续3天偏离）
    三级「严重」→ 子女强提醒 + 社区网格员介入（连续5天偏离）

★ 通知按**事件**而非按天发出

  判定层每天都给一个 risk_level，但一段连续的 L2/L3 是**一个事件**，不是 N 个。
  只在状态变化时通知：开始 / 升级 / 出现新风险类型 / 缓解。同级持续不重复，
  改由周报的"预警回执"承担告知义务。

  修复前实测：E001 60 天内 12 次推送对应 4 个真实事件；PERM_step（永久性衰退）
  是连续 91 天每天报 L3 = 91 次短信 + 强制响铃 + 惊动网格员。后果不是"吵"，
  是家属关掉通知后**真正的新变化也收不到了**。

  事件模型与判定逻辑在 src/risk/alert_state.py，本模块只负责"决定之后做什么"。
  ★ 最关键的一条不变量：**升级穿透一切抑制**（含冷却与人工确认）。
"""

from datetime import datetime
from enum import IntEnum

from src.risk import alert_state
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlertLevel(IntEnum):
    NORMAL = 0
    ATTENTION = 1   # 关注
    WARNING = 2     # 提醒
    SEVERE = 3      # 严重


# 预警动作的**默认值**。运行时以 config/settings.yaml 的 alert 段为准，
# 本表只在配置缺失或某一级没配时兜底（见 _actions_for）。
#
# 此前 settings.yaml 的整个 alert 段无人读取：trigger_alert 收了 config 形参
# 却从未在函数体里用过，动作表完全硬编码在这里。改配置不生效，且两边还漂了——
# 配置里 level_1.notify 写的是 false（布尔），代码里是 []（列表）。
ALERT_ACTIONS = {
    AlertLevel.NORMAL: {
        # 正常的一天什么都不做——这是对的，不要改。缓解通知走的是下面的
        # RESOLVE_ACTIONS，因为它是**状态迁移**的通知，不是"L0 这个等级"的动作。
        "action": "none",
        "notify": [],
        "log": True,
    },
    AlertLevel.ATTENTION: {
        "action": "log_only",
        "notify": [],
        "log": True,
        "include_in_report": True,
    },
    AlertLevel.WARNING: {
        "action": "push_notification",
        "notify": ["children"],
        "log": True,
        "include_in_report": True,
        "highlight_in_report": True,
    },
    AlertLevel.SEVERE: {
        "action": "force_notification",
        "notify": ["children", "community_worker"],
        "log": True,
        "include_in_report": True,
        "highlight_in_report": True,
    },
}


# 缓解（回到 L0）的通知动作。
#
# ★ 为什么不能沿用 ALERT_ACTIONS[NORMAL]
#
#   `notify_on_resolve: true` 配了却**完全无效**：decide 正确返回
#   (RESOLVED, True)，但 trigger_alert 拿 AlertLevel.NORMAL 去查动作表，
#   得到 action="none"，`_execute_alert_actions` 第一行就 return []；
#   `_build_alert_message` 对 NORMAL 也直接返回 ""。实测：
#       D1 L3 → started   alerted=True  actions=[log_alert, push_to_children,
#                                                push_to_community_worker, force_ring]
#       D5 L0 → resolved  alerted=False actions=[]  message=''
#   三层各自都"对"，合起来这个配置项静默失效——而它承担的是"家属同样想知道
#   '过去了'"这件事。test_alert_events 只断言了 transition == RESOLVED，
#   没断言 alerted，所以测不到（又一次"断言打错了层"）。
#
#   根因是把"等级"与"状态迁移"混成了一件事：缓解不是"L0 这个等级要做什么"，
#   而是"从有事件变成没事件要通知谁"。所以它需要自己的动作表。
#
#   刻意用 push_notification 而非 force_notification：缓解是好消息，
#   不该强制响铃、也不该惊动社区网格员。收件人取"曾经被通知过的那些人"，
#   见 _resolve_actions_for。
RESOLVE_ACTIONS = {
    "action": "push_notification",
    "notify": ["children"],
    "log": True,
    "include_in_report": True,
}


def _normalize_notify(value) -> list[str]:
    """
    把配置里的 notify 归一成收件人列表。

    YAML 里写 `notify: false` 表示"不通知"（settings.yaml 的 level_1 就是这么写的），
    但代码一路当列表用。不归一的话 `for r in False` 会直接抛异常。
    """
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _actions_for(level: AlertLevel, config: dict | None) -> dict:
    """
    取该等级的动作配置：以 settings.yaml 的 alert 段为准，缺项回落到 ALERT_ACTIONS。

    回落而非报错：alert 段是可选的运营配置，缺了应当按内置默认继续发预警，
    而不是让整条每日管道因为一段配置没写就中断。
    """
    defaults = ALERT_ACTIONS.get(level, ALERT_ACTIONS[AlertLevel.NORMAL])
    if not config:
        return defaults

    section = (config.get("alert") or {}).get(f"level_{int(level)}")
    if not isinstance(section, dict):
        return defaults

    merged = dict(defaults)
    if "action" in section:
        merged["action"] = section["action"]
    if "notify" in section:
        merged["notify"] = _normalize_notify(section["notify"])
    # ★ 这是白名单式合并：不在这个元组里的键会被**静默丢弃**。
    #   新增配置项时必须同步加进来——settings.yaml 的 alert 段注释里记着的
    #   "这段配置此前根本没人读"就是同一类缺陷的上一次发作。
    for key in ("log", "include_in_report", "highlight_in_report", "repeat_days"):
        if key in section:
            merged[key] = section[key]
    return merged


def _resolve_actions_for(config: dict | None, prev_level: int) -> dict:
    """缓解通知的动作配置。

    收件人跟着**事件曾经达到过的最高等级**走：只有 L3 惊动过社区网格员，
    那就也该告诉他们"过去了"；只推送过子女的事件不必因为缓解去打扰网格员。
    否则会出现"网格员被叫来过但没人告诉他结束了"，或反过来"从没参与的人
    收到一条莫名其妙的缓解通知"。

    配置入口是 `alert.events.resolve`（可选）；不写则用 RESOLVE_ACTIONS。
    """
    merged = dict(RESOLVE_ACTIONS)

    if prev_level >= int(AlertLevel.SEVERE):
        # 沿用 L3 的收件人名单，但**不**沿用 force_notification 与 force_ring
        severe = ALERT_ACTIONS[AlertLevel.SEVERE]
        merged["notify"] = _normalize_notify(severe.get("notify", []))

    section = ((config or {}).get("alert") or {}).get("events") or {}
    resolve_cfg = section.get("resolve")
    if isinstance(resolve_cfg, dict):
        if "action" in resolve_cfg:
            merged["action"] = resolve_cfg["action"]
        if "notify" in resolve_cfg:
            merged["notify"] = _normalize_notify(resolve_cfg["notify"])
        # 白名单式合并，同 _actions_for：新增键必须加进这个元组
        for key in ("log", "include_in_report"):
            if key in resolve_cfg:
                merged[key] = resolve_cfg[key]
    return merged


def _event_gap_days(config: dict | None, channel: str) -> int:
    """事件断段窗口（隔多久没观测就算两段无关的事件），**按 channel 取**。

    ★ 两条流的观测节奏差一个数量级，不能共用一个窗口。

      baseline 是每日批处理，复用 `risk.continuity.max_skip_days`——与
      `continuity.walk_back_days` 同一套断段语义，"超过三天没有可信数据，
      就不该再假装这是同一段状态"。

      depression 是**稀疏事件驱动**：要"有正脸 + 有连续语音"的片段，独居老人
      可能几周才有一次（见 run_depression_assessment.py 的模块 docstring）。
      套用 3 天的话，任何两次相邻评估都 > 3+1 天 → 每次都判 TRANSITION_STARTED
      → **去重完全失效**，而去重正是 alert_state 存在的理由。实测两次间隔 7 天
      的"重度"评估各推送一次、notify_count 各自归 1。

      改用评估有效期 `depression.assessment.valid_days`：评估还在有效期内，
      就是"它仍代表当前状况"，那两次评估描述的就是同一段事件。语义自洽，
      也不必再引入一个会与它漂开的第二个数字。
    """
    cfg = config or {}
    if channel == alert_state.CHANNEL_DEPRESSION:
        return int(
            ((cfg.get("depression") or {}).get("assessment") or {}).get("valid_days", 30)
        )
    return int(((cfg.get("risk") or {}).get("continuity") or {}).get("max_skip_days", 3))


def _risk_keys_of(risk_types: list[dict]) -> list[str]:
    """取稳定的机器键，不用中文展示名。

    `risk_type`（"睡眠稳定性偏离"）是文案，改一次措辞事件身份就断了；
    `risk_key`（sleep_stability）是 build_risk_rules 的 dict key，稳定。
    """
    keys = []
    for r in risk_types or []:
        key = r.get("risk_key") or r.get("risk_type")
        if key:
            keys.append(str(key))
    return keys


def trigger_alert(
    elder_id: str,
    risk_level: int,
    risk_types: list[dict] | None = None,
    config: dict | None = None,
    day_key: str | None = None,
    channel: str = alert_state.CHANNEL_BASELINE,
) -> dict:
    """
    根据风险等级触发对应的预警动作，**按事件去重**。

    ★ 这不再是一个纯函数。它读写 data/logs/alert_state/{elder_id}.json，
      因为"上次发过没有"这件事跨天，而每日批处理的进程活不过今天。

    Args:
        elder_id: 老人ID
        risk_level: 风险等级 (0/1/2/3)
        risk_types: 风险类型列表（用其中的 risk_key 做事件身份）
        config: 全局配置。给了就以其 alert 段为准，不给则用内置默认。
        day_key: 判定所属自然日。**补算历史日时必须给**——不给会退回
            datetime.now()，把"今天"的日期盖到一条历史判定上，冷却窗口
            与事件边界全算错。这与 judge_risk_level 被迫加 today_key
            是同一类问题。
        channel: 事件流。baseline（GRU 双轨）与 depression（MPDD）各自独立
            计数、独立冷却、互不压制——理由同"绝不跨轨比较绝对分"。

    Returns:
        {
            "alerted": bool,        # 本次实际发出了通知吗（被抑制则 False）
            "suppressed": bool,     # 是否因事件去重被抑制
            "transition": str,      # started/escalated/new_type/resolved/
                                    # repeat/improved/cooldown/acknowledged/none
            "event": dict,          # 事件快照（持续天数、已通知次数等）
            "level": str, "label": str, "actions": [...], "message": str,
        }
    """
    if risk_types is None:
        risk_types = []

    try:
        level_enum = AlertLevel(risk_level)
    except ValueError:
        logger.error(f"无效的风险等级: {risk_level}")
        level_enum = AlertLevel.NORMAL

    if day_key is None:
        day_key = datetime.now().strftime(alert_state.DATE_FMT)
        logger.warning(
            "trigger_alert 未收到 day_key，退回使用今天。补算历史日时这会让"
            "冷却窗口与事件边界算在错误的日期上。"
        )

    actions_config = _actions_for(level_enum, config)
    normalized_level = int(level_enum)

    # 1. 事件决策
    cfg = config or {}
    alert_cfg = cfg.get("alert") or {}
    events_cfg = alert_cfg.get("events") or {}
    event_gap_days = _event_gap_days(cfg, channel)
    repeat_days = int(actions_config.get("repeat_days", 0) or 0)

    state = alert_state.load_state(elder_id)
    channel_state = state["channels"].get(channel) or {}
    risk_keys = _risk_keys_of(risk_types)

    # 事件曾达到过的最高等级，用于决定缓解通知发给谁（见 _resolve_actions_for）。
    # 必须在 decide/apply 之前读：apply 对 RESOLVED 会把 channel 清空。
    prev_level = int(channel_state.get("last_notified_level") or 0)

    transition, should_notify = alert_state.decide(
        channel_state, day_key, normalized_level, risk_keys,
        events_cfg, repeat_days, event_gap_days,
    )

    # ★ 缓解走独立的动作表：它是**状态迁移**的通知，不是"L0 这个等级"的动作。
    #   沿用 ALERT_ACTIONS[NORMAL] 会让 action="none" 把整条通知吃掉，
    #   notify_on_resolve 这个配置项静默失效。见 RESOLVE_ACTIONS 的说明。
    is_resolve = transition == alert_state.TRANSITION_RESOLVED
    if is_resolve:
        actions_config = _resolve_actions_for(config, prev_level)

    # 2. 只在需要通知时才真正执行动作
    #
    #    被抑制时仍然记一条 log_alert：日志是排查用的，不该跟着通知一起消失。
    #    消失的只有推送与响铃。
    if should_notify:
        # 缓解按 WARNING 的强度执行（推送、不响铃），而不是按 level_enum
        # ——此刻 level_enum 是 NORMAL，会一个动作都不执行。
        actions = _execute_alert_actions(
            elder_id,
            AlertLevel.WARNING if is_resolve else level_enum,
            actions_config,
            risk_types,
        )
    else:
        actions = ["log_alert"] if actions_config.get("action") != "none" else []

    # 3. 落盘事件状态
    state["channels"][channel] = alert_state.apply(
        channel_state, day_key, normalized_level, risk_keys, transition, should_notify,
    )
    try:
        alert_state.save_state(state)
    except OSError as e:
        # 状态写不出去只该少一次去重（退回到多发通知），不该让日管道失败。
        logger.error(f"预警状态落盘失败（本次仍按判定执行）: {e}")

    event = state["channels"][channel]
    suppressed = bool(not should_notify and normalized_level >= AlertLevel.WARNING)

    log_entry = {
        "elder_id": elder_id,
        "day_key": day_key,
        "channel": channel,
        "risk_level": normalized_level,
        "risk_label": level_enum.name,
        "action": actions_config["action"] if should_notify else "suppressed",
        "transition": transition,
        "risk_types": [r.get("risk_type", "") for r in risk_types],
    }

    alert_result = {
        # 用归一化后的 level_enum，不用原始整数：trigger_alert("E001", 99)
        # 会被上面的 try 兜成 NORMAL，但 `99 >= WARNING` 仍为真，于是返回
        # {"alerted": True, "level": "NORMAL", ...}——上层据此以为发过预警，
        # 实际一个动作都没执行。
        # 现在还要再与 should_notify 取与：被事件去重抑制时没有真的发出去。
        #
        # 缓解单独放行：它的 level_enum 是 NORMAL（< WARNING），但确实推送了。
        # 不放行的话 alerted 恒 False，上层（run_daily_pipeline 的打印、
        # 周报回执、测试）都会以为缓解通知没发出去。
        "alerted": bool(
            should_notify
            and (level_enum >= AlertLevel.WARNING or is_resolve)
        ),
        "suppressed": suppressed,
        "transition": transition,
        "event": {
            "channel": channel,
            "active": event.get("active", False),
            "level": event.get("level", 0),
            "risk_keys": event.get("risk_keys", []),
            "started_day": event.get("started_day"),
            "age_days": alert_state.event_age_days(event, day_key),
            "notify_count": event.get("notify_count", 0),
            "last_notified_day": event.get("last_notified_day"),
            "acknowledged": event.get("acknowledged", False),
        },
        "level": level_enum.name,
        "label": _get_level_label(risk_level),
        "actions": actions,
        # 缓解要有自己的文案：_build_alert_message 对 NORMAL 返回 ""，
        # 于是家属会收到一条空推送。
        "message": (
            _build_resolve_message(elder_id, event, prev_level)
            if is_resolve
            else _build_alert_message(elder_id, level_enum, risk_types)
        ),
    }

    # 记录日志
    if actions_config.get("log", True):
        if suppressed:
            logger.info(f"预警抑制({transition}): {log_entry}")
        elif level_enum >= AlertLevel.WARNING:
            logger.warning(f"预警触发({transition}): {log_entry}")
        else:
            logger.info(f"预警记录({transition}): {log_entry}")

    return alert_result


def _get_level_label(risk_level: int) -> str:
    """获取风险等级中文标签"""
    labels = {0: "正常", 1: "关注", 2: "提醒", 3: "严重"}
    return labels.get(risk_level, "未知")


def _build_alert_message(
    elder_id: str,
    level: AlertLevel,
    risk_types: list[dict],
) -> str:
    """构建预警消息文本"""
    if level == AlertLevel.NORMAL:
        return ""

    risk_names = [r.get("risk_type", "") for r in risk_types] if risk_types else ["多项指标"]

    templates = {
        AlertLevel.ATTENTION: (
            f"【关注】{elder_id}老人{'、'.join(risk_names)}指标出现轻微波动，"
            f"系统将持续监测"
        ),
        AlertLevel.WARNING: (
            f"【提醒】{elder_id}老人{'、'.join(risk_names)}指标连续偏离常态，"
            f"建议您主动联系老人了解近况"
        ),
        AlertLevel.SEVERE: (
            f"【严重】{elder_id}老人{'、'.join(risk_names)}指标严重异常，"
            f"建议尽快安排探访或就医咨询"
        ),
    }

    return templates.get(level, "")


def _build_resolve_message(elder_id: str, event: dict, prev_level: int) -> str:
    """缓解通知的文案。

    措辞约束同 judge._generate_recommendation：只说行为观察，不下诊断，
    也不说"已康复"——回到个人基线不等于问题解决，只是当前不再偏离。
    """
    label = _get_level_label(prev_level) if prev_level else "异常"
    return (
        f"【缓解】{elder_id}老人此前的{label}状态已回到个人常态范围，"
        f"系统将继续日常监测"
    )


def _execute_alert_actions(
    elder_id: str,
    level: AlertLevel,
    actions_config: dict,
    risk_types: list[dict],
) -> list[str]:
    """
    执行预警动作（当前为模拟接口）。

    Returns:
        已执行的动作列表
    """
    executed = []

    action = actions_config.get("action", "none")

    if action == "none":
        return executed

    # 日志记录（始终执行）
    executed.append("log_alert")

    if action == "log_only":
        return executed

    # App推送（模拟）
    if action in ("push_notification", "force_notification"):
        notify_list = _normalize_notify(actions_config.get("notify", []))
        for recipient in notify_list:
            executed.append(f"push_to_{recipient}")
        logger.info(
            f"  └─ 推送通知: elder={elder_id}, "
            f"recipients={notify_list}, level={level.name}"
        )

    # 强提醒（模拟）
    if level == AlertLevel.SEVERE:
        executed.append("force_ring")
        logger.warning(f"  └─ 强提醒: elder={elder_id}")

    return executed
