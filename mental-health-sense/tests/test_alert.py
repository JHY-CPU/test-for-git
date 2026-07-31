"""
预警推送单元测试
"""

import pytest

from src.risk.alert import (
    trigger_alert,
    AlertLevel,
    ALERT_ACTIONS,
    _build_alert_message,
    _get_level_label,
)


class TestAlertLevel:
    """测试预警等级枚举"""

    def test_level_values(self):
        assert AlertLevel.NORMAL == 0
        assert AlertLevel.ATTENTION == 1
        assert AlertLevel.WARNING == 2
        assert AlertLevel.SEVERE == 3

    def test_level_from_int(self):
        assert AlertLevel(0) == AlertLevel.NORMAL
        assert AlertLevel(3) == AlertLevel.SEVERE

    def test_invalid_level(self):
        with pytest.raises(ValueError):
            AlertLevel(99)


class TestAlertActions:
    """测试预警动作配置"""

    def test_normal_has_no_notify(self):
        assert ALERT_ACTIONS[AlertLevel.NORMAL]["notify"] == []

    def test_warning_notifies_children(self):
        assert "children" in ALERT_ACTIONS[AlertLevel.WARNING]["notify"]

    def test_severe_notifies_all(self):
        notify = ALERT_ACTIONS[AlertLevel.SEVERE]["notify"]
        assert "children" in notify
        assert "community_worker" in notify


class TestTriggerAlert:
    """测试预警触发"""

    def test_normal_no_alert(self):
        result = trigger_alert("E001", 0)
        assert not result["alerted"]
        assert result["level"] == "NORMAL"

    def test_attention_logs_only(self):
        result = trigger_alert("E001", 1)
        assert not result["alerted"]  # 一级不推送
        assert "log_alert" in result["actions"]

    def test_warning_pushes_notification(self):
        result = trigger_alert(
            "E001",
            2,
            risk_types=[{"risk_type": "情绪低落", "risk_key": "mood_low"}],
        )
        assert result["alerted"]
        assert "push_to_children" in result["actions"]

    def test_severe_force_notification(self):
        result = trigger_alert("E001", 3)
        assert result["alerted"]
        assert "push_to_community_worker" in result["actions"]
        assert "force_ring" in result["actions"]

    def test_invalid_level_defaults_to_normal(self):
        result = trigger_alert("E001", 99)
        assert result["level"] == "NORMAL"


class TestAlertConfigWiring:
    """★ 回归：settings.yaml 的 alert 段必须真的被读到。

    历史缺陷：trigger_alert 收了 config 形参却从未在函数体里用过，动作表
    完全硬编码在 ALERT_ACTIONS。改配置不生效，两边还漂了——配置里
    level_1.notify 是布尔 false，代码里是列表 []。
    """

    def test_config_overrides_recipients(self):
        config = {"alert": {"level_2": {
            "action": "push_notification",
            "notify": ["children", "family_doctor"],
        }}}
        result = trigger_alert("E001", 2, config=config)
        assert "push_to_family_doctor" in result["actions"]

    def test_config_can_downgrade_action(self):
        """配置把二级改成只记日志 → 不该再推送"""
        config = {"alert": {"level_2": {"action": "log_only", "notify": []}}}
        result = trigger_alert("E001", 2, config=config)
        assert result["actions"] == ["log_alert"]

    def test_boolean_notify_normalized(self):
        """YAML 里 `notify: false` 不能让 `for r in False` 炸掉"""
        config = {"alert": {"level_2": {
            "action": "push_notification", "notify": False,
        }}}
        result = trigger_alert("E001", 2, config=config)
        assert not any(a.startswith("push_to_") for a in result["actions"])

    def test_missing_section_falls_back_to_defaults(self):
        """alert 段缺失时按内置默认继续发预警，而不是让每日管道中断"""
        result = trigger_alert("E001", 3, config={"risk": {}})
        assert "push_to_community_worker" in result["actions"]
        assert "force_ring" in result["actions"]

    def test_shipped_config_is_readable(self):
        """仓库里那份 settings.yaml 本身要能被正确消费"""
        from src.utils.io import load_config

        config = load_config()
        assert config.get("alert"), "settings.yaml 应有 alert 段"
        result = trigger_alert("E001", 3, config=config)
        assert result["alerted"]
        assert "push_to_children" in result["actions"]


class TestAlertMessages:
    """测试预警消息生成"""

    def test_attention_message(self):
        msg = _build_alert_message("E001", AlertLevel.ATTENTION, [])
        assert "关注" in msg or "E001" in msg

    def test_warning_message(self):
        msg = _build_alert_message(
            "E001", AlertLevel.WARNING,
            [{"risk_type": "情绪低落"}],
        )
        assert "提醒" in msg or "情绪低落" in msg

    def test_severe_message(self):
        msg = _build_alert_message("E001", AlertLevel.SEVERE, [])
        assert "严重" in msg

    def test_normal_no_message(self):
        msg = _build_alert_message("E001", AlertLevel.NORMAL, [])
        assert msg == ""


class TestLevelLabel:
    """测试等级标签"""

    def test_all_labels(self):
        assert _get_level_label(0) == "正常"
        assert _get_level_label(1) == "关注"
        assert _get_level_label(2) == "提醒"
        assert _get_level_label(3) == "严重"
        assert _get_level_label(99) == "未知"
