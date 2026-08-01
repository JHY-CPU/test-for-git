"""2026-08-02 第三轮走查：预警事件状态机与四态数据质量的回归测试。

这一轮的 5 个缺陷全部活在「481 全绿 + 范围 1 20/20」之下，共同特征仍是
**断言打在错的层**（同 VALIDATION §8/§9）：

  - 事件状态机只测了 `trigger_alert` 的活跃期，没测"事件关闭之后"
  - `_maybe_alert` 整个函数零覆盖
  - `offline` 只在 `validate_daily_data` 这个**辅助函数**上被直接构造过，
    从没有人问过"生产链路走得到它吗"（答案是走不到）

所以本文件的断言全部落在生产入口上：`trigger_alert` / `_maybe_alert` /
`_process_track` / `infer_track` / `_apply_cold_start_fallbacks`。

预警状态由 conftest 的 autouse fixture 隔离到 tmp；需要真实 data/ 布局的用例
用 conftest 的 `sandbox`。
"""

import json
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pytest

ELDER = "R001"
START = "2026-07-01"


def day_key_of(day: int) -> str:
    """第 day 天（1-based）的 day_key，与 test_integration 同一套口径。"""
    return (datetime.strptime(START, "%Y-%m-%d")
            + timedelta(days=day - 1)).strftime("%Y-%m-%d")


# ===== 缺陷二：事件缓解后通道失忆，补算历史日会凭空推送 =====

class TestChannelCursorSurvivesResolve:
    """★ 修复前实测：

        2026-08-10 L2 → started  （推送给子女）
        2026-08-12 L0 → resolved （推送"缓解"，apply 的 _empty_channel 清空游标）
        补算 2026-07-20 L2 → **started，真的又推了一次**，started_day 锚在 7 月
        2026-08-13 L2 → started （再推一次）

    一次"补算上个月漏跑的某天"换来两条错误推送，而 README 明写"补算与重跑是安全的"。
    """

    def test_缓解后补算历史日不得开出新事件(self):
        from src.risk import alert_state
        from src.risk.alert import trigger_alert

        trigger_alert(ELDER, 2, day_key="2026-08-10")
        trigger_alert(ELDER, 2, day_key="2026-08-11")
        trigger_alert(ELDER, 0, day_key="2026-08-12")     # 事件关闭
        after_resolve = deepcopy(
            alert_state.load_state(ELDER)["channels"]["baseline"]
        )

        historical = trigger_alert(ELDER, 2, day_key="2026-07-20")

        assert historical["transition"] == "none"
        assert historical["alerted"] is False, "补算历史日不得真的推送给子女"
        assert "push_to_children" not in historical["actions"]
        assert historical["event"]["active"] is False
        assert (
            alert_state.load_state(ELDER)["channels"]["baseline"] == after_resolve
        ), "补算不得改写通道状态"

    def test_通道游标跨事件保留且不被补算回拨(self):
        from src.risk import alert_state
        from src.risk.alert import trigger_alert

        trigger_alert(ELDER, 2, day_key="2026-08-10")
        trigger_alert(ELDER, 0, day_key="2026-08-12")

        ch = alert_state.load_state(ELDER)["channels"]["baseline"]
        assert ch["active"] is False
        assert ch["last_processed_day"] == "2026-08-12", (
            "事件关闭不该把通道游标一起清掉——它描述的是通道不是这个事件"
        )

        trigger_alert(ELDER, 2, day_key="2026-07-20")
        ch = alert_state.load_state(ELDER)["channels"]["baseline"]
        assert ch["last_processed_day"] == "2026-08-12", "游标只增不减"

    def test_正常日也推进游标(self):
        """一直没有事件的日子同样要推进游标，否则"连续正常 N 天后补算"会漏过守卫。"""
        from src.risk import alert_state
        from src.risk.alert import trigger_alert

        for offset in range(5):
            trigger_alert(ELDER, 0, day_key=f"2026-08-{10 + offset:02d}")

        ch = alert_state.load_state(ELDER)["channels"]["baseline"]
        assert ch["last_processed_day"] == "2026-08-14"
        assert trigger_alert(ELDER, 3, day_key="2026-08-01")["alerted"] is False

    def test_老状态文件缺游标要用last_seen_day兜底(self, _isolate_alert_state):
        """★ 这条守的是"修复本身别变成回归"。

        2026-08-02 之前写下的状态文件没有 last_processed_day。若 load_state 只按
        默认补 None，decide 第 0 条的守卫在这些文件上会**完全失效**——而旧代码
        那版守卫读的是 last_seen_day，在同一批文件上恰恰是挡得住的。

        实测输入取自仓库里真实的 data/logs/alert_state/E001.json（active=true、
        last_seen_day=2026-08-29、acknowledged=true）：不兜底时照文档跑
        `--date 2026-08-15` 判 L3 会 escalated + 强制响铃 + 惊动网格员。
        """
        from src.risk import alert_state
        from src.risk.alert import trigger_alert

        (_isolate_alert_state / f"{ELDER}.json").write_text(json.dumps({
            "schema_version": "1.0.0", "elder_id": ELDER,
            "channels": {
                "baseline": {
                    "active": True, "level": 1,
                    "risk_keys": ["sleep_stability", "social_decline"],
                    "started_day": "2026-08-08", "last_seen_day": "2026-08-29",
                    "last_notified_day": "2026-08-23", "last_notified_level": 3,
                    "notify_count": 5, "acknowledged": True,
                    "acknowledged_at": "2026-07-31T23:45:13",
                },
            },
        }), encoding="utf-8")

        assert (
            alert_state.load_state(ELDER)["channels"]["baseline"]["last_processed_day"]
            == "2026-08-29"
        ), "老文件缺游标时应回落到 last_seen_day"

        backfill = trigger_alert(ELDER, 3, day_key="2026-08-15")

        assert backfill["transition"] == "none"
        assert backfill["alerted"] is False, "补算历史日不得强制响铃 + 惊动网格员"
        assert "force_ring" not in backfill["actions"]

        ch = alert_state.load_state(ELDER)["channels"]["baseline"]
        assert ch["last_seen_day"] == "2026-08-29", "游标/观测日不得被补算拨回"
        assert ch["acknowledged"] is True, "补算不得清掉家属已确认的状态"
        assert ch["notify_count"] == 5

    def test_脏day_key不得毒化游标(self, _isolate_alert_state):
        """不可解析的 day_key 一旦写进游标，_days_between 此后恒返回 None，
        该通道的乱序守卫会**永久静默失效**且无任何报错线索。"""
        from src.risk import alert_state

        state = alert_state._empty_channel()
        polluted = alert_state.apply(state, "not-a-date", 2, [], "started", True)
        assert polluted["last_processed_day"] is None

        state = alert_state.apply(
            alert_state._empty_channel(), "2026-08-10", 2, [], "started", True
        )
        state = alert_state.apply(state, "garbage", 2, [], "cooldown", False)
        assert state["last_processed_day"] == "2026-08-10", "脏值不得覆盖有效游标"

    def test_缓解后真正的新事件仍要通知(self):
        """守卫不能矫枉过正：晚于游标的新风险期该报还得报。"""
        from src.risk.alert import trigger_alert

        trigger_alert(ELDER, 2, day_key="2026-08-10")
        trigger_alert(ELDER, 0, day_key="2026-08-12")
        again = trigger_alert(ELDER, 2, day_key="2026-08-20")

        assert again["transition"] == "started"
        assert again["alerted"] is True
        assert again["event"]["started_day"] == "2026-08-20"


# ===== 缺陷一：抑郁通道既不关事件、也没有去重 =====

def _dep_config(alert_on: bool = True) -> tuple[dict, dict]:
    """返回 (完整 config, depression 段)，打开 alert 开关模拟 checkpoint 校准后。"""
    from src.utils.io import load_config

    config = load_config()
    dep_cfg = {**config["depression"], "alert": alert_on}
    return {**config, "depression": dep_cfg}, dep_cfg


def _assess(elder: str, day_key: str, level: str, config: dict, dep_cfg: dict) -> None:
    from src.depression.runner import _maybe_alert

    _maybe_alert(
        elder, day_key,
        {"status": "assessed", "result": {"level": level}},
        dep_cfg, config,
    )


class TestDepressionChannel:
    def test_评估回到正常要关闭事件(self):
        """修复前：`if risk_level <= 0: return` 让事件永不关闭，
        周报的「预警回执」无限期显示"情绪状态评估 持续中，第 N 天"。"""
        from src.risk.alert_state import active_events, load_state

        config, dep_cfg = _dep_config()
        _assess(ELDER, "2026-08-01", "重度", config, dep_cfg)
        assert load_state(ELDER)["channels"]["depression"]["active"] is True

        _assess(ELDER, "2026-08-15", "正常", config, dep_cfg)

        ch = load_state(ELDER)["channels"]["depression"]
        assert ch["active"] is False, "评估回到正常必须关闭事件"
        assert "depression" not in active_events(ELDER), "周报回执不该再显示它"

    def test_稀疏评估属于同一事件不重复推送(self):
        """抑郁评估几周才一次。事件断段窗口若沿用 baseline 的 max_skip_days=3，
        每次评估都会判成新事件 → 去重完全失效（实测两次各推一次）。"""
        from src.risk.alert_state import load_state

        config, dep_cfg = _dep_config()
        _assess(ELDER, "2026-08-01", "重度", config, dep_cfg)
        _assess(ELDER, "2026-08-08", "重度", config, dep_cfg)   # 间隔 7 天

        ch = load_state(ELDER)["channels"]["depression"]
        assert ch["started_day"] == "2026-08-01", "同一段事件，不该重开"
        assert ch["notify_count"] == 1, f"稀疏评估被推送了 {ch['notify_count']} 次"

    def test_超出有效期才算新事件(self):
        """valid_days=30：评估过了有效期就不再代表当前状况，那是新的一段。"""
        from src.risk.alert_state import load_state

        config, dep_cfg = _dep_config()
        _assess(ELDER, "2026-06-01", "重度", config, dep_cfg)
        _assess(ELDER, "2026-08-01", "重度", config, dep_cfg)   # 隔了 61 天

        ch = load_state(ELDER)["channels"]["depression"]
        assert ch["started_day"] == "2026-08-01"

    def test_评估失败不得关闭已有事件(self):
        """"测不到"不等于"好转"——同 judge 拿不到方向元数据时不压等级。"""
        from src.depression.runner import _maybe_alert
        from src.risk.alert_state import load_state

        config, dep_cfg = _dep_config()
        _assess(ELDER, "2026-08-01", "重度", config, dep_cfg)
        _maybe_alert(
            ELDER, "2026-08-03",
            {"status": "failed", "result": None}, dep_cfg, config,
        )
        assert load_state(ELDER)["channels"]["depression"]["active"] is True

    def test_未知等级名不得被当成缓解(self):
        """`.get(level, 0)` 的兜底在"0 == 缓解"之后语义变了：认不出的等级名
        会去关闭活跃事件并推一条「已回到个人常态范围」。level 直接取自
        checkpoint 的 probabilities 键名，换 checkpoint 就会踩到。"""
        from src.depression.runner import _maybe_alert
        from src.risk.alert_state import load_state

        config, dep_cfg = _dep_config()
        _assess(ELDER, "2026-08-01", "重度", config, dep_cfg)
        _maybe_alert(
            ELDER, "2026-08-03",
            {"status": "assessed", "result": {"level": "moderate"}},
            dep_cfg, config,
        )
        assert load_state(ELDER)["channels"]["depression"]["active"] is True

    def test_开关关闭时完全不碰事件状态(self):
        from src.risk.alert_state import load_state

        config, dep_cfg = _dep_config(alert_on=False)
        _assess(ELDER, "2026-08-01", "重度", config, dep_cfg)
        assert load_state(ELDER)["channels"]["depression"]["active"] is False

    def test_两条流的断段窗口互不影响(self):
        """baseline 仍按 max_skip_days=3 断段，不该被抑郁那条的 30 天带跑。"""
        from src.risk.alert import trigger_alert

        config, _ = _dep_config()
        trigger_alert(ELDER, 2, config=config, day_key="2026-08-01")
        later = trigger_alert(ELDER, 2, config=config, day_key="2026-08-11")
        assert later["transition"] == "started", "baseline 断开 10 天仍应是新事件"


# ===== 缺陷三：QUALITY_OFFLINE 在生产链路不可达 =====

class TestOfflineReachableInPipeline:
    """★ 修复前：`_process_track` 在聚合抛 DataInsufficientError 时直接
    return insufficient，走不到 `validate_daily_data` 的离线升级分支；而聚合已经
    把"缺 ≥3 维"整类截走，那条分支的 missing_count 恒 ≤2。于是 offline 全仓
    只有单测直接构造 missing_count=3 才产得出来。实测连跑 7 天全缺仍只记
    insufficient，四态退化成三态。
    """

    def _run(self, sandbox, days: list[tuple[str, dict | None]]) -> list[str]:
        """按序跑若干天的 sleep 轨 _process_track，返回每天的质量档。"""
        from src.data_pipeline.aggregator import aggregate_sleep_features
        from src.scheduler.daily_job import _process_track
        from src.utils.io import load_config

        config = load_config()
        return [
            _process_track(
                ELDER, day, "sleep", aggregate_sleep_features(payload), config
            )
            for day, payload in days
        ]

    def test_连续数据不足升级为offline(self, sandbox):
        from src.utils.io import load_features_csv

        qualities = self._run(sandbox, [(day_key_of(i), None) for i in range(1, 8)])

        assert qualities[0] == "insufficient", "首日无历史，不该直接判 offline"
        assert "offline" in qualities, "连续多天全缺必须升级为 offline"
        assert qualities[-1] == "offline"

        # 必须真的落进 CSV——只在返回值里对而磁盘上还是 insufficient
        # 等于什么都没修（get_quality_summary / _get_recent_quality 都读磁盘）
        df = load_features_csv(ELDER, "sleep")
        assert "offline" in set(df["data_quality"])

    def test_健康数据里偶发一天不足不得升级(self, sandbox):
        """守住反向：offline 的语义是"设备连着几天没上报"，不是"今天缺了"。"""
        good = {
            "sleep_efficiency": 0.86, "waso_min": 30.0, "sol_min": 18.0,
            "bed_exit_count": 1.0, "deep_sleep_ratio": 0.19,
            "sleep_onset_clock": 165.0, "night_hr_mean": 62.0,
            "daytime_nap_min": 25.0,
        }
        days = [(day_key_of(i), good) for i in range(1, 5)]
        days.append((day_key_of(5), None))

        qualities = self._run(sandbox, days)
        assert qualities[:4] == ["valid"] * 4
        assert qualities[4] == "insufficient", "只缺一天不该判设备离线"

    def test_两个产出点共用同一判据(self):
        """离线判据只准有一份：两份会漂的判据比没有更危险
        （同 imputer 删掉 check_offline_status 的理由）。"""
        from src.data_pipeline.validator import (
            escalate_if_offline,
            validate_daily_data,
        )

        recent = ["insufficient"] * 3
        assert escalate_if_offline("insufficient", recent) == "offline"
        assert validate_daily_data(
            np.full(8, np.nan), 8, "sleep", recent_quality=recent
        ) == "offline"
        assert escalate_if_offline("valid", recent) == "valid"
        assert escalate_if_offline("insufficient", ["valid"] * 3) == "insufficient"


# ===== 缺陷四：兜底轨阈值被复用到 GRU 轨 =====

@pytest.fixture
def trained_sleep_baseline(sandbox, monkeypatch):
    """在 sandbox 里造 60 天数据并建好睡眠轨基线。

    `generate_all_data` 的落盘 ID 取自模块级 ELDER_ID，改它才能落到本文件的
    ELDER 目录——与 scripts/validate_synthetic.py 的 `gen.ELDER_ID = VELDER`
    是同一条路数。
    """
    import scripts.generate_simulation_data as gen
    from src.baseline.trainer import train_initial_baseline

    monkeypatch.setattr(gen, "ELDER_ID", ELDER)
    gen.generate_all_data(sandbox, n_days=60, start_date=START)
    train_initial_baseline(ELDER, "sleep")
    return sandbox


def _write_prior_log(day_key: str, sleep_track: dict) -> None:
    """写一份该日的推理日志（首跑产物），供重跑复用阈值。"""
    from src.utils.io import get_log_dir

    log_dir = get_log_dir("daily_inference")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{ELDER}_{day_key}.json").write_text(
        json.dumps({"elder_id": ELDER, "day_key": day_key, "sleep": sleep_track}),
        encoding="utf-8",
    )


class TestThresholdReuseIsMethodScoped:
    def test_兜底期阈值不得被复用到GRU轨(self, trained_sleep_baseline):
        """★ 打在 infer_track 这个真入口上，不打在 _load_prior_thresholds 上。

        建档完成后补算一个建档期内的日子会同时满足两个条件：基线文件已存在
        （走 GRU 路径）、day_key <= ewma.last_day_key（判为重跑）。修复前会把
        兜底日志里的 fallback_sigma=3.0 当成 GRU 轨的 dynamic_threshold，
        而 GRU 轨正常阈值在 1.3~1.6——is_deviation 恒为 False。
        """
        from src.baseline.inference import infer_track
        from src.utils.io import load_config

        # 手工写一份"建档期那天走了冷启动兜底"的日志（真实链路会这么写）
        target = day_key_of(20)
        _write_prior_log(target, {
            "track": "sleep", "status": "cold_start_fallback",
            "anomaly_score": 1.77, "is_deviation": False,
            "static_threshold": 3.0, "ewma_threshold": 3.0,
            "dynamic_threshold": 3.0,
        })

        result = infer_track(ELDER, target, "sleep", load_config())

        # `ewma_n` 只有 GRU 路径会写，兜底路径的返回里根本没有这个键——
        # 用它确认这次真的走了 GRU，而不是断言某个具体 status
        # （重跑不喂 EWMA，inferences_since_train=0 会让 status 落在 observation，
        #  那与"走没走 GRU"无关，断死 "success" 是一条脆断言）。
        assert "ewma_n" in result, "基线已就绪，应走 GRU 路径"
        assert result["status"] not in ("cold_start", "cold_start_fallback")
        assert result["dynamic_threshold"] != pytest.approx(3.0), (
            "兜底轨的 fallback_sigma 被当成了 GRU 轨阈值——两者量纲不可比"
        )
        assert result["dynamic_threshold"] < 3.0

    def test_同源重跑仍然复用阈值(self, trained_sleep_baseline):
        """守住反向：幂等复用本身不能被这条守卫误伤。

        同日重跑必须拿到与首跑一模一样的阈值，否则擦线分数会翻转 is_deviation，
        而该字段驱动 consecutive / qualifies / 微调排除集 / 周报统计。
        """
        from src.baseline.inference import infer_track
        from src.utils.io import load_config

        target = day_key_of(20)
        _write_prior_log(target, {
            "track": "sleep", "status": "success",
            "static_threshold": 1.42, "ewma_threshold": 1.37,
            "dynamic_threshold": 1.37,
        })

        result = infer_track(ELDER, target, "sleep", load_config())
        assert result["dynamic_threshold"] == pytest.approx(1.37)


# ===== 缺陷五：兜底路径重算连续天数漏了基准日 =====

def test_兜底路径连续天数按基准日截断(monkeypatch):
    """`load_daily_results` 不传 end_day_key 会拿磁盘上**最新**几天的日志去数
    连续偏离天数——VALIDATION §9 缺陷①"补算历史日判成最新日"的残留入口。"""
    from src.scheduler import daily_job
    from src.utils import io

    captured: dict = {}

    def _spy(elder_id, n_days=7, end_day_key=None):
        captured["end_day_key"] = end_day_key
        return []

    monkeypatch.setattr(io, "load_daily_results", _spy)
    monkeypatch.setattr(io, "save_daily_result", lambda *a, **k: None)
    monkeypatch.setattr(
        daily_job, "_cold_start_fallback_track",
        lambda *a, **k: {"status": "cold_start_fallback", "is_deviation": True},
    )

    inference_result = {"sleep": {"status": "cold_start"}}
    daily_job._apply_cold_start_fallbacks(
        ELDER, "2026-08-15", inference_result, ("sleep",), {}, {"sleep": "valid"},
    )

    assert captured["end_day_key"] == "2026-08-15"
