"""2026-07-31 走查缺陷的回归测试

★ 这些断言全部打在**真实链路的输出**上，不打在辅助函数上。

  本仓的教训（VALIDATION §8）是：上一轮缺陷能长期活在"全绿"之下，原因是
  断言打错了层——验证脚本数的是 status 流转而非能否出等级；集成测试断言的是
  validator 的辅助函数而非 rules 层的实际行为；单测手搓的字典带着生产链路
  根本没写过的字段。所以这一轮每条回归都从 run_daily_pipeline / judge_risk_level /
  generate_weekly_report 这些真入口进。
"""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.baseline.scaler_utils import SLEEP_FEATURES, SOCIAL_FEATURES
from src.risk.judge import judge_risk_level
from src.risk.rules import _counts_toward_consecutive


def _day(n: int) -> str:
    return (datetime(2026, 8, 1) + timedelta(days=n)).strftime("%Y-%m-%d")


def _track_result(
    *, deviation: bool, quality: str = "valid", score: float = 2.0,
    threshold: float = 1.0, features: list[str],
) -> dict:
    """构造一条与生产链路**同构**的单轨推理结果。

    字段照着 infer_track 的真实返回写：少一个 dynamic_threshold，_severity 就会
    退回绝对分；少一个 signed_available，方向闸门就直接放行。手搓字典时漏字段
    正是上一轮"绿得没有意义"的成因。
    """
    return {
        "track": "sleep" if features is SLEEP_FEATURES else "social",
        "anomaly_score": score,
        "static_threshold": threshold,
        "ewma_threshold": threshold,
        "dynamic_threshold": threshold,
        "is_deviation": deviation,
        "signed_residuals": {n: -1.5 for n in features},
        "abs_residuals": {n: 1.5 for n in features},
        "signed_z": {n: -2.0 for n in features},
        "signed_available": True,
        "valid_features": list(features),
        "skipped_features": [],
        "data_quality": quality,
        "status": "success",
    }


def _log(day: str, *, sleep=None, social=None, quality=None) -> dict:
    out = {"elder_id": "T001", "day_key": day, "status": "success"}
    if sleep is not None:
        out["sleep"] = sleep
    if social is not None:
        out["social"] = social
    if quality is not None:
        out["track_quality"] = quality
    out["is_deviation"] = any(
        t.get("is_deviation") for t in (sleep, social) if isinstance(t, dict)
    )
    return out


class TestF3_判级必须跳过降级日:
    """degraded 日既不累加也不打断——这条不变量此前只在 rules 层生效，
    judge 层完全没有，导致同一段数据"不激活风险类型却报 L3"。"""

    def test_降级日不该凑出严重等级(self):
        # 5 天社交偏离，其中第 2、4 天 C6c 掉线判 degraded。
        # 真正可信的偏离只有 3 天（D0/D2/D4 中间被断开），不该到 L3。
        logs = []
        for i in range(5):
            quality = "degraded" if i in (1, 3) else "valid"
            logs.append(_log(
                _day(i),
                social=_track_result(
                    deviation=True, quality=quality, score=2.0, threshold=1.0,
                    features=SOCIAL_FEATURES,
                ),
            ))

        result = judge_risk_level("T001", logs, config={"risk": {}}, today_key=_day(4))
        social = result["per_track"]["social"]

        # 计入的只有 3 天（degraded 被跳过），达不到 severe 门槛 5
        assert social["consecutive"] == 3, (
            f"degraded 日被计入了：consecutive={social['consecutive']}，应为 3"
        )
        assert result["risk_level"] < 3, (
            "L3 会触发短信 + 强制响铃 + 网格员介入，不该由设计上不计入的日子凑出"
        )

    def test_降级日也不该打断真实的连续段(self):
        """反向：真实 5 天恶化里夹一个 degraded，不该让计数归零。"""
        logs = []
        for i in range(6):
            quality = "degraded" if i == 2 else "valid"
            logs.append(_log(
                _day(i),
                sleep=_track_result(
                    deviation=True, quality=quality, score=2.0, threshold=1.0,
                    features=SLEEP_FEATURES,
                ),
            ))

        result = judge_risk_level("T001", logs, config={"risk": {}}, today_key=_day(5))
        # 6 天里跳过 1 天 → 计入 5 天，全部偏离
        assert result["per_track"]["sleep"]["consecutive"] == 5


class TestF3_缺日不得粘连:
    def test_断开超过max_skip_days即打断(self):
        """5 条日志跨 22 个日历日仍数出 consecutive=5 的老问题。"""
        logs = [
            _log(_day(0), sleep=_track_result(
                deviation=True, features=SLEEP_FEATURES)),
            _log(_day(1), sleep=_track_result(
                deviation=True, features=SLEEP_FEATURES)),
            # 中间断 10 天
            _log(_day(12), sleep=_track_result(
                deviation=True, features=SLEEP_FEATURES)),
            _log(_day(13), sleep=_track_result(
                deviation=True, features=SLEEP_FEATURES)),
        ]
        result = judge_risk_level("T001", logs, config={"risk": {}}, today_key=_day(13))
        assert result["per_track"]["sleep"]["consecutive"] == 2, (
            "断开 10 天的两段偏离被粘成了一段"
        )


class TestF2_单轨掉线时的持续性判定:
    """daily_job 只把 usable 轨传给 daily_inference，不可用轨没有子字典，
    质量档只在顶层 track_quality 镜像里。"""

    def test_轨缺失时从track_quality读到真实档位(self):
        day = _log(
            _day(0),
            social=_track_result(deviation=False, features=SOCIAL_FEATURES),
            quality={"sleep": "insufficient", "social": "valid"},
        )
        # 睡眠轨没有子字典，但顶层镜像里有 insufficient → 该日不计入睡眠类规则
        assert _counts_toward_consecutive(day, frozenset({"sleep"})) is False
        # 社交轨正常 → 计入
        assert _counts_toward_consecutive(day, frozenset({"social"})) is True

    def test_跨轨规则任一轨降级即不计入(self):
        day = _log(
            _day(0),
            social=_track_result(deviation=False, features=SOCIAL_FEATURES),
            quality={"sleep": "offline", "social": "valid"},
        )
        assert _counts_toward_consecutive(day, frozenset({"sleep", "social"})) is False


class TestF1_补算历史日:
    """quick_judge 取的是"最近 N 个文件"，与被补算的那一天无关。"""

    def test_基准日之后的日志不得进入判定窗(self):
        logs = [
            _log(_day(0), sleep=_track_result(
                deviation=False, score=0.5, features=SLEEP_FEATURES)),
            _log(_day(1), sleep=_track_result(
                deviation=False, score=0.5, features=SLEEP_FEATURES)),
            # 补算基准日 = _day(1)；后面这些"未来"的偏离日不该影响它
            _log(_day(2), sleep=_track_result(
                deviation=True, score=9.0, features=SLEEP_FEATURES)),
            _log(_day(3), sleep=_track_result(
                deviation=True, score=9.0, features=SLEEP_FEATURES)),
        ]
        # 显式给 today_key：调用方知道自己在判哪一天
        result = judge_risk_level("T001", logs, config={"risk": {}}, today_key=_day(1))
        assert result["per_track"]["sleep"]["consecutive"] == 0
        assert result["risk_level"] == 0, (
            "判定窗漂到了后面的偏离日上——这正是补算历史日会拿最新那天的等级"
            "去发预警、去写契约的成因"
        )


class TestF11_日志损坏不瘫痪整条链路:
    def test_单个损坏日志被跳过而不是抛异常(self, tmp_path, monkeypatch):
        from src.utils import io

        log_dir = tmp_path / "daily_inference"
        log_dir.mkdir(parents=True)
        monkeypatch.setattr(io, "get_log_dir", lambda t="daily_inference": log_dir)

        good = _log(_day(0), sleep=_track_result(
            deviation=False, features=SLEEP_FEATURES))
        (log_dir / f"T001_{_day(0)}.json").write_text(
            json.dumps(good), encoding="utf-8")
        (log_dir / f"T001_{_day(1)}.json").write_text("{截断的 js", encoding="utf-8")

        results = io.load_daily_results("T001", n_days=10)
        assert len(results) == 1, "坏文件应被跳过，而不是让整条链路抛 JSONDecodeError"
        assert results[0]["day_key"] == _day(0)

    def test_end_day_key_截断窗口(self, tmp_path, monkeypatch):
        from src.utils import io

        log_dir = tmp_path / "daily_inference"
        log_dir.mkdir(parents=True)
        monkeypatch.setattr(io, "get_log_dir", lambda t="daily_inference": log_dir)

        for i in range(5):
            (log_dir / f"T001_{_day(i)}.json").write_text(
                json.dumps(_log(_day(i))), encoding="utf-8")

        results = io.load_daily_results("T001", n_days=10, end_day_key=_day(2))
        assert [r["day_key"] for r in results] == [_day(0), _day(1), _day(2)]


class TestF4_缺测维不得伪造偏离:
    """imputer 在原始量纲填 0 会造出 −15σ 的假偏离。"""

    def test_填不上的维保持NaN(self):
        from src.baseline.scaler_utils import TRACK_SOCIAL
        from src.data_pipeline.imputer import impute_missing

        idx = SOCIAL_FEATURES.index("copresence_min")
        current = np.array([np.nan, 90.0, 0.9, 0.3, 180.0])
        prev = np.array([120.0, 90.0, 0.9, 0.3, 180.0])

        filled, missing_count, missing_names = impute_missing(
            current, TRACK_SOCIAL, prev)

        assert np.isnan(filled[idx]), (
            "copresence_min 缺失时填 0 会产生满足 down 方向的假 z，"
            "把'摄像头掉线'变成'确实没人来'——而它是 social_decline 的必选维"
        )
        assert "copresence_min" in missing_names

    def test_缺测维不进signed_z(self):
        """推理结果里不该出现缺测维的 z——rules 的方向判定会把它当真实证据读走。"""
        result = _track_result(deviation=False, features=SOCIAL_FEATURES)
        # 生产链路的 infer_track 只把 valid_features 写进三个残差字典
        assert set(result["signed_z"]) == set(result["valid_features"])


class TestF10_默认门槛与配置一致:
    def test_social_decline_默认门槛是1点0(self):
        """代码默认值曾是 1.2，而 settings.yaml 用近 30 行论证过 1.2 是"假绿"。
        凡是传裁剪配置的调用方都会静默拿到损坏的门槛。"""
        from src.risk.rules import build_risk_rules

        assert build_risk_rules({})["social_decline"].threshold_ratio == 1.0

    def test_代码默认值与配置文件一致(self):
        from src.risk.rules import build_risk_rules
        from src.utils.io import load_config

        from_config = build_risk_rules(load_config())
        from_default = build_risk_rules({})
        for key in ("sleep_stability", "social_decline", "circadian_disruption"):
            assert (
                from_config[key].threshold_ratio == from_default[key].threshold_ratio
            ), f"{key} 的代码默认值与 settings.yaml 漂开了"


class TestMinor_预警返回值:
    def test_非法等级不得报告已预警(self):
        from src.risk.alert import trigger_alert

        result = trigger_alert("T001", 99)
        assert result["level"] == "NORMAL"
        assert result["alerted"] is False, (
            "alerted 用原始整数判断时，99 >= WARNING 为真，上层会以为发过预警，"
            "而实际一个动作都没执行"
        )


class TestMinor_EWMA空池:
    def test_空池阈值为inf而非报错(self):
        from src.baseline.ewma import CumulativeEWMABaseline

        ewma = CumulativeEWMABaseline(alpha=0.05)
        assert ewma.get_threshold(2.5) == float("inf"), (
            "没有基线时阈值应当永不触发；返回 0 会让任何分数都判偏离"
        )
        repr(ewma)   # 不得抛 TypeError
