"""
三级预警推送模块

预警等级（全部基于趋势检测）：
    一级「关注」→ 写入周报，不打扰老人（单日偏离）
    二级「提醒」→ 子女App推送 + 周报重点标注（连续3天偏离）
    三级「严重」→ 子女强提醒 + 社区网格员介入（连续5天偏离）
"""

from datetime import datetime
from enum import IntEnum

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
    for key in ("log", "include_in_report", "highlight_in_report"):
        if key in section:
            merged[key] = section[key]
    return merged


def trigger_alert(
    elder_id: str,
    risk_level: int,
    risk_types: list[dict] | None = None,
    config: dict | None = None,
) -> dict:
    """
    根据风险等级触发对应的预警动作。

    Args:
        elder_id: 老人ID
        risk_level: 风险等级 (0/1/2/3)
        risk_types: 风险类型列表
        config: 全局配置。给了就以其 alert 段为准，不给则用内置默认。

    Returns:
        {"alerted": bool, "level": str, "actions": [...], "message": str}
    """
    if risk_types is None:
        risk_types = []

    try:
        level_enum = AlertLevel(risk_level)
    except ValueError:
        logger.error(f"无效的风险等级: {risk_level}")
        level_enum = AlertLevel.NORMAL

    actions_config = _actions_for(level_enum, config)

    log_entry = {
        "elder_id": elder_id,
        "timestamp": datetime.now().isoformat(),
        "risk_level": risk_level,
        "risk_label": level_enum.name,
        "action": actions_config["action"],
        "risk_types": [r.get("risk_type", "") for r in risk_types],
    }

    # 执行推送（当前为模拟，实际对接推送服务）
    alert_result = {
        "alerted": risk_level >= AlertLevel.WARNING,
        "level": level_enum.name,
        "label": _get_level_label(risk_level),
        "actions": _execute_alert_actions(elder_id, level_enum, actions_config, risk_types),
        "message": _build_alert_message(elder_id, level_enum, risk_types),
    }

    # 记录日志
    if actions_config.get("log", True):
        if level_enum >= AlertLevel.WARNING:
            logger.warning(f"预警触发: {log_entry}")
        else:
            logger.info(f"预警记录: {log_entry}")

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
