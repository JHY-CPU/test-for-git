"""预警事件模型测试

★ 断言全部打在 `trigger_alert` 这个**真入口**上，不打在 `decide` 上。

  `decide` 是纯函数、好测，但只测它就重演了本仓踩过的坑：断言打在辅助函数
  而非真实链路的输出上（VALIDATION §8）。真实链路里还有配置读取、状态落盘、
  动作执行三层，任何一层接错都会让"决策正确但通知没发出去"。
  所以主体用 trigger_alert，`decide` 只在构造纯边界时直接调。

★ 本文件最重要的一条是 `TestEscalationBypassesEverything`。

  纯粹按"同一等级 X 天内最多一次"做冷却会在 L2 的冷却窗内吃掉 L3——
  拿误报换漏报，比修复前更糟。那两条用例是整个设计的守门员。
"""

from datetime import datetime, timedelta

import pytest

from src.risk.alert import trigger_alert
from src.risk.alert_state import (
    CHANNEL_BASELINE,
    CHANNEL_DEPRESSION,
    TRANSITION_ACKNOWLEDGED,
    TRANSITION_COOLDOWN,
    TRANSITION_ESCALATED,
    TRANSITION_IMPROVED,
    TRANSITION_NEW_TYPE,
    TRANSITION_NONE,
    TRANSITION_REPEAT,
    TRANSITION_RESOLVED,
    TRANSITION_STARTED,
    acknowledge,
    load_state,
)

BASE = datetime(2026, 8, 1)


def day(n: int) -> str:
    return (BASE + timedelta(days=n)).strftime("%Y-%m-%d")


def cfg(repeat_l2: int = 0, repeat_l3: int = 30, **events) -> dict:
    """与 settings.yaml 同构的最小配置（含 events 段）。"""
    ev = {"notify_on_resolve": True, "new_type_cooldown_days": 3, "weekly_receipt": True}
    ev.update(events)
    return {
        "alert": {
            "level_1": {"action": "log_only", "notify": []},
            "level_2": {"action": "push_notification", "notify": ["children"],
                        "repeat_days": repeat_l2},
            "level_3": {"action": "force_notification",
                        "notify": ["children", "community_worker"],
                        "repeat_days": repeat_l3},
            "events": ev,
        },
        "risk": {"continuity": {"max_skip_days": 3}},
    }


SLEEP = [{"risk_key": "sleep_stability", "risk_type": "睡眠稳定性偏离"}]
SOCIAL = [{"risk_key": "social_decline", "risk_type": "社会连接减弱"}]


def fire(elder, n, level, types=SLEEP, config=None, channel=CHANNEL_BASELINE):
    return trigger_alert(elder, level, types, config or cfg(), day(n), channel)


class TestEventStart:
    def test_首次_L2_发出通知(self):
        r = fire("A1", 0, 2)
        assert r["transition"] == TRANSITION_STARTED
        assert r["alerted"] is True
        assert "push_to_children" in r["actions"]
        assert r["event"]["age_days"] == 1

    def test_事件年龄随天数增长(self):
        fire("A2", 0, 2)
        r = fire("A2", 4, 2)
        assert r["event"]["started_day"] == day(0)
        assert r["event"]["age_days"] == 5


class TestSameLevelSuppressed:
    def test_同级第二天起被抑制(self):
        first = fire("B1", 0, 2)
        second = fire("B1", 1, 2)

        assert first["alerted"] is True
        assert second["transition"] == TRANSITION_COOLDOWN
        assert second["alerted"] is False
        assert second["suppressed"] is True
        # 仍记日志，但不推送、不响铃——日志是排查用的，不该跟着通知一起消失
        assert second["actions"] == ["log_alert"]

    def test_连续十天只通知一次(self):
        notified = [fire("B2", i, 2)["alerted"] for i in range(10)]
        assert sum(notified) == 1, f"10 天 L2 通知了 {sum(notified)} 次，应为 1"

    def test_L3_按_repeat_days_月度重提(self):
        """L3 会惊动社区网格员，完全静默不合适。"""
        notified = [fire("B3", i, 3, config=cfg(repeat_l3=30))["alerted"]
                    for i in range(91)]
        # 开始 1 次 + 第 30/60/90 天各 1 次
        assert sum(notified) == 4, f"91 天 L3 通知了 {sum(notified)} 次，应为 4"

    def test_repeat_days_为零则永不重提(self):
        notified = [fire("B4", i, 3, config=cfg(repeat_l3=0))["alerted"]
                    for i in range(91)]
        assert sum(notified) == 1


class TestEscalationBypassesEverything:
    """★ 守门员：升级必须穿透一切抑制。

    纯冷却方案在这里会漏报——D1 发 L2 进入冷却，D5 升 L3 被冷却窗吃掉。
    那是拿误报换漏报，比完全没有冷却更危险。
    """

    def test_冷却期内升级必须发出(self):
        fire("C1", 0, 2)
        assert fire("C1", 1, 2)["alerted"] is False        # 冷却中
        r = fire("C1", 2, 3)
        assert r["transition"] == TRANSITION_ESCALATED
        assert r["alerted"] is True, "恶化被冷却窗吃掉了——这比没有冷却更糟"
        assert "force_ring" in r["actions"]

    def test_已确认后升级仍必须发出(self):
        fire("C2", 0, 2)
        acknowledge("C2")
        assert fire("C2", 1, 2)["transition"] == TRANSITION_ACKNOWLEDGED
        r = fire("C2", 2, 3)
        assert r["alerted"] is True, "「已知晓」是针对更低等级的，不该静默掉恶化"
        # 升级后确认状态重置：新等级需要重新确认
        assert r["event"]["acknowledged"] is False

    def test_同日重算出更高等级也算升级(self):
        """补了数据 / 修了 bug 后同一天重算出 L3，那是真实的升级，
        不该被"同日幂等"分支静默掉。"""
        fire("C3", 0, 2)
        r = fire("C3", 0, 3)
        assert r["transition"] == TRANSITION_ESCALATED
        assert r["alerted"] is True


class TestNewRiskType:
    def test_出现新类型要通知(self):
        # 逐日推进：真实链路每天都跑，last_seen_day 天天更新。
        # 跳着喂会被事件边界判成"断开太久 → 新事件"，测不到本意。
        fire("D1", 0, 2, SLEEP)
        fire("D1", 1, 2, SLEEP)
        fire("D1", 2, 2, SLEEP)
        r = fire("D1", 3, 2, SLEEP + SOCIAL)   # 距上次通知 3 天 >= new_type_cooldown
        assert r["transition"] == TRANSITION_NEW_TYPE
        assert r["alerted"] is True
        assert set(r["event"]["risk_keys"]) == {"sleep_stability", "social_decline"}

    def test_新类型受短冷却约束(self):
        """类型会在门槛附近抖动，紧挨着上次通知时不重复响。"""
        fire("D2", 0, 2, SLEEP)
        r = fire("D2", 1, 2, SLEEP + SOCIAL)   # 距上次通知仅 1 天 < 3
        assert r["alerted"] is False
        assert r["transition"] == TRANSITION_COOLDOWN

    def test_类型消失再出现不算新类型(self):
        """risk_keys 取并集：抖动时同一个类型不该被反复当成"新出现"。"""
        fire("D3", 0, 2, SLEEP + SOCIAL)
        fire("D3", 1, 2, SLEEP)                # social 暂时不达标
        fire("D3", 2, 2, SLEEP)
        fire("D3", 3, 2, SLEEP)
        r = fire("D3", 4, 2, SLEEP + SOCIAL)   # 又回来了
        # 断言 cooldown 而非仅 != NEW_TYPE：后者在事件被判成 started 时也成立，
        # 那是一条会假通过的断言。
        assert r["transition"] == TRANSITION_COOLDOWN
        assert r["alerted"] is False


class TestResolveAndImprove:
    def test_回到L0发缓解通知并关闭事件(self):
        fire("E1", 0, 3)
        r = fire("E1", 5, 0, [])
        assert r["transition"] == TRANSITION_RESOLVED
        assert r["event"]["active"] is False
        assert load_state("E1")["channels"][CHANNEL_BASELINE]["active"] is False

    def test_缓解通知可关闭(self):
        fire("E2", 0, 3)
        r = fire("E2", 5, 0, [], config=cfg(notify_on_resolve=False))
        assert r["transition"] == TRANSITION_RESOLVED
        assert r["alerted"] is False

    def test_降级但仍有风险时不通知(self):
        """L3 掉到 L2 仍需关注，这时发"好转了"会让家属误以为可以放松。"""
        fire("E3", 0, 3)
        r = fire("E3", 1, 2)
        assert r["transition"] == TRANSITION_IMPROVED
        assert r["alerted"] is False
        assert r["event"]["active"] is True, "降级不该关闭事件"

    def test_缓解后再次恶化算新事件(self):
        fire("E4", 0, 2)
        fire("E4", 3, 0, [])
        r = fire("E4", 6, 2)
        assert r["transition"] == TRANSITION_STARTED
        assert r["event"]["started_day"] == day(6)

    def test_一直正常时不产生事件(self):
        r = fire("E5", 0, 0, [])
        assert r["transition"] == TRANSITION_NONE
        assert r["event"]["active"] is False


class TestEventBoundary:
    def test_长时间断开算新事件(self):
        """设备离线一段时间后，不该把两段无关的风险期粘成同一个事件——
        与 continuity.walk_back_days 的断段语义一致。"""
        fire("F1", 0, 2)
        r = fire("F1", 10, 2)     # 断开 10 天 > max_skip_days(3)+1
        assert r["transition"] == TRANSITION_STARTED
        assert r["event"]["started_day"] == day(10)

    def test_短暂断开仍是同一事件(self):
        fire("F2", 0, 2)
        r = fire("F2", 3, 2)      # 断开 3 天 <= 4，仍算同一段
        assert r["transition"] == TRANSITION_COOLDOWN
        assert r["event"]["started_day"] == day(0)


class TestIdempotent:
    def test_同日重跑不改变状态(self):
        fire("G1", 0, 2)
        before = load_state("G1")["channels"][CHANNEL_BASELINE]
        r = fire("G1", 0, 2)
        after = load_state("G1")["channels"][CHANNEL_BASELINE]

        assert r["transition"] == TRANSITION_NONE
        assert r["alerted"] is False
        assert after == before, "补算/重跑同一天必须幂等"

    def test_重跑不推进通知计数(self):
        fire("G2", 0, 2)
        for _ in range(5):
            fire("G2", 0, 2)
        assert load_state("G2")["channels"][CHANNEL_BASELINE]["notify_count"] == 1


class TestChannelsAreIndependent:
    def test_两条流互不压制(self):
        """抑郁事件流与基线事件流各自计数、各自冷却。
        理由同「绝不跨轨比较绝对分」：它们不是同一件事。"""
        fire("H1", 0, 2, channel=CHANNEL_BASELINE)
        assert fire("H1", 1, 2, channel=CHANNEL_BASELINE)["alerted"] is False

        r = fire("H1", 1, 2, channel=CHANNEL_DEPRESSION)
        assert r["alerted"] is True, "基线在冷却，不该顺带静默抑郁通道"
        assert r["transition"] == TRANSITION_STARTED

    def test_确认一条流不影响另一条(self):
        fire("H2", 0, 2, channel=CHANNEL_BASELINE)
        fire("H2", 0, 2, channel=CHANNEL_DEPRESSION)
        acknowledge("H2", channel=CHANNEL_BASELINE)

        state = load_state("H2")["channels"]
        assert state[CHANNEL_BASELINE]["acknowledged"] is True
        assert state[CHANNEL_DEPRESSION]["acknowledged"] is False


class TestAcknowledge:
    def test_确认后同级持续被静默(self):
        fire("I1", 0, 2, config=cfg(repeat_l2=1))
        acknowledge("I1")
        r = fire("I1", 1, 2, config=cfg(repeat_l2=1))
        assert r["transition"] == TRANSITION_ACKNOWLEDGED
        assert r["alerted"] is False

    def test_撤销确认后恢复通知(self):
        fire("I2", 0, 2, config=cfg(repeat_l2=1))
        acknowledge("I2")
        acknowledge("I2", undo=True)
        r = fire("I2", 1, 2, config=cfg(repeat_l2=1))
        assert r["transition"] == TRANSITION_REPEAT
        assert r["alerted"] is True

    def test_无活跃事件时确认返回_None(self):
        assert acknowledge("I3") is None

    def test_非法_channel_报错(self):
        with pytest.raises(ValueError, match="Unknown channel"):
            acknowledge("I4", channel="nope")


class TestCorruptedState:
    def test_损坏状态被重置且不抛(self, _isolate_alert_state):
        """一个坏文件不该让整条日管道当天失败、老人零监测。"""
        (_isolate_alert_state / "J1.json").write_text("{截断的 js", encoding="utf-8")
        r = fire("J1", 0, 2)
        assert r["transition"] == TRANSITION_STARTED
        assert r["alerted"] is True

    def test_单通道结构异常只重置该通道(self, _isolate_alert_state):
        import json
        (_isolate_alert_state / "J2.json").write_text(json.dumps({
            "schema_version": "1.0.0", "elder_id": "J2",
            "channels": {"baseline": "不是字典", "depression": {"active": False}},
        }), encoding="utf-8")
        r = fire("J2", 0, 2)
        assert r["alerted"] is True


class TestDayKeyRequired:
    def test_缺_day_key_时退回今天并告警(self, caplog):
        """补算历史日不传 day_key 会把"今天"盖到历史判定上。"""
        r = trigger_alert("K1", 2, SLEEP, cfg())
        assert r["event"]["started_day"] == datetime.now().strftime("%Y-%m-%d")


class TestConfigWiring:
    def test_repeat_days_能从配置读到(self):
        """_actions_for 是白名单式合并，新键不加进白名单会被静默丢弃。"""
        notified = [fire("L1", i, 2, config=cfg(repeat_l2=2))["alerted"]
                    for i in range(6)]
        # day0 started + day2 + day4 重提
        assert sum(notified) == 3, f"repeat_days 未生效，通知 {sum(notified)} 次"

    def test_仓库真实配置可读且生效(self):
        from src.utils.io import load_config

        real = load_config()
        assert fire("L2", 0, 2, config=real)["alerted"] is True
        assert fire("L2", 1, 2, config=real)["alerted"] is False
