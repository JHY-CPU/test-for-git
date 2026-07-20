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
