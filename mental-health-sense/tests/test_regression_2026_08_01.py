"""2026-08-01 漏跑日与预警事件缺陷的回归测试。

断言尽量落在生产入口的最终输出上，避免辅助函数正确、整条链仍失效。
"""

from copy import deepcopy

import numpy as np
import pandas as pd


def test_feature_window按自然日补NaN(monkeypatch):
    from src.utils import io

    names = io.get_feature_names("sleep")
    rows = []
    for day, value in (("2026-08-01", 1.0), ("2026-08-03", 3.0)):
        row = {name: value for name in names}
        row.update(day_key=day, missing_count=0, data_quality="valid")
        rows.append(row)
    monkeypatch.setattr(io, "load_features_csv", lambda elder, track: pd.DataFrame(rows))

    matrix, missing = io.get_feature_window(
        "T001", "2026-08-01", "2026-08-03", "sleep"
    )

    assert matrix.shape == (3, len(names))
    assert missing == ["2026-08-02"]
    assert np.isnan(matrix[1]).all()
    assert matrix[0, 0] == 1.0 and matrix[2, 0] == 3.0


def test_recent_quality把漏跑日计为insufficient(monkeypatch):
    from src.scheduler import daily_job

    df = pd.DataFrame([
        {"day_key": "2026-08-01", "data_quality": "valid"},
        {"day_key": "2026-08-03", "data_quality": "valid"},
    ])
    monkeypatch.setattr(daily_job, "load_features_csv", lambda elder, track: df)

    quality = daily_job._get_recent_quality(
        "T001", "sleep", n_days=3, before_day_key="2026-08-04"
    )
    assert quality == ["valid", "insufficient", "valid"]


def test_有可用轨但推理无结论不得返回success(monkeypatch):
    from src.baseline import inference
    from src.scheduler import daily_job

    monkeypatch.setattr(daily_job, "aggregate_sleep_features", lambda raw: {})
    monkeypatch.setattr(daily_job, "aggregate_social_features", lambda activity, camera: {})
    monkeypatch.setattr(daily_job, "_process_track", lambda *args, **kwargs: "valid")
    monkeypatch.setattr(daily_job, "_apply_cold_start_fallbacks", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_job, "_get_recent_quality", lambda *args, **kwargs: ["valid"])
    monkeypatch.setattr(
        inference,
        "daily_inference",
        lambda *args, **kwargs: {
            "status": "data_insufficient",
            "track_statuses": {"sleep": "data_insufficient", "social": "data_insufficient"},
        },
    )

    result = daily_job.run_daily_pipeline(
        "T001", "2026-08-10", raw_data={}, config={"risk": {}}
    )

    assert result["risk_result"] is None
    assert result["status"] == "no_verdict"


def test_不可评估轨应跳过而非打断连续段():
    from src.risk.rules import _counts_toward_consecutive

    day = {
        "sleep": {
            "status": "data_insufficient",
            "data_quality": "valid",
            "is_deviation": False,
        },
        "track_quality": {"sleep": "valid"},
    }
    assert _counts_toward_consecutive(day, frozenset({"sleep"})) is False


def test_缓解通知实际发出且文案非空():
    from src.risk.alert import trigger_alert

    started = trigger_alert("T001", 3, day_key="2026-08-10")
    resolved = trigger_alert("T001", 0, day_key="2026-08-11")

    assert started["alerted"] is True
    assert resolved["transition"] == "resolved"
    assert resolved["alerted"] is True
    assert "push_to_children" in resolved["actions"]
    assert "push_to_community_worker" in resolved["actions"]
    assert "force_ring" not in resolved["actions"]
    assert "回到个人常态范围" in resolved["message"]


def test_乱序补算不得改写当前活跃事件():
    from src.risk import alert_state
    from src.risk.alert import trigger_alert

    trigger_alert("T001", 3, day_key="2026-08-10")
    trigger_alert("T001", 3, day_key="2026-08-11")
    before = deepcopy(alert_state.load_state("T001")["channels"]["baseline"])

    historical = trigger_alert("T001", 0, day_key="2026-07-15")
    after = alert_state.load_state("T001")["channels"]["baseline"]

    assert historical["transition"] == "none"
    assert historical["alerted"] is False
    assert after == before


# ===== 2026-08-02 审查补充：infer_track 的 max_gap 守卫此前从未被真执行 =====
#
# 最高影响 bug（漏跑一天 → 之后连续 7 天 data_insufficient、score=0、零监测）的
# 回归只测了 io.get_feature_window（helper）与把整个 daily_inference stub 掉——
# infer_track 里 `len(missing_days) > max_gap` 这个真实分支从没被测试跑到。
# 下面两个用例直接打 infer_track，且覆盖"无行缺日"与"全 NaN 占位行"两种形态。

_CONFIG = {
    "gru": {"window": 7, "num_layers": 1, "dropout": 0.2,
            "sleep": {"feature_dim": 8, "hidden_dim": 8},
            "social": {"feature_dim": 5, "hidden_dim": 8}},
    "risk": {"sigma_multiplier": 2.5, "cold_start_observation_days": 0,
             "continuity": {"max_skip_days": 3}},
    "ewma": {"alpha": 0.05, "max_freeze_days": 14},
}


def _fake_inference_ready(monkeypatch, df):
    """把 infer_track 的基建换成假对象，只留真实的窗口读取与 max_gap 守卫。
    目标日 2026-07-01，窗口 06-24~06-30。"""
    import numpy as np

    from src.baseline import inference
    from src.baseline.scaler_utils import get_feature_names
    from src.utils import io

    names = get_feature_names("sleep")
    dim = len(names)

    class _FakeScaler:
        mean_ = np.zeros(dim)
        scale_ = np.ones(dim)

        def transform(self, x):
            return x

    class _FakeEWMA:
        @classmethod
        def load(cls, *a, **k):
            return cls()

        def n_samples(self, is_weekend=False):
            return 28

        def min_samples_required(self, config, is_weekend=False):
            return 20

        def already_fed(self, day_key):
            return False

        def get_threshold(self, sigma, is_weekend=False):
            return 1.5

        def update(self, *a, **k):
            return True

        def save(self, *a, **k):
            return None

        def total_samples(self):
            return 28

    monkeypatch.setattr(io, "load_features_csv", lambda elder, track: df)
    monkeypatch.setattr("src.baseline.scaler_utils.load_scaler", lambda p: _FakeScaler())
    monkeypatch.setattr(io, "load_gru_model", lambda *a, **k: object())
    monkeypatch.setattr(inference, "load_residual_stats", lambda e, t: {
        "signed": {"mean": np.zeros(dim), "std": np.ones(dim)},
        "abs": {"mean": np.ones(dim), "std": np.ones(dim)},
        "signed_available": True,
    })
    monkeypatch.setattr(inference, "TrackEWMAPools", _FakeEWMA)
    monkeypatch.setattr(inference, "get_track_meta", lambda e, t: {"ewma_n_at_train": 28})


def test_infer_track窗口缺日超限返回data_insufficient(monkeypatch):
    """★ [18] 无行缺日形态：窗口里缺 5 天 > max_gap(3)，必须 data_insufficient。"""
    import numpy as np
    import pandas as pd

    from src.baseline import inference
    from src.baseline.scaler_utils import get_feature_names

    names = get_feature_names("sleep")
    rows = []
    for day in (24, 30):                          # 只留两行，其余 5 天无行
        r = {n: 1.0 for n in names}
        r.update(day_key=f"2026-06-{day:02d}", missing_count=0, data_quality="valid")
        rows.append(r)
    target = {n: 1.0 for n in names}
    target.update(day_key="2026-07-01", missing_count=0, data_quality="valid")
    rows.append(target)
    _fake_inference_ready(monkeypatch, pd.DataFrame(rows))

    result = inference.infer_track("T001", "2026-07-01", "sleep", config=_CONFIG)

    assert result["status"] == "data_insufficient"
    assert len(result["missing_window_days"]) > 3


def test_infer_track全NaN占位行计入缺日(monkeypatch):
    """★ [14] 全 NaN 占位行形态：daily_job 对聚合失败的日子写一行全 NaN
    （"留下痕迹比什么都不写好"），这类行不能绕过 max_gap 守卫——否则设备连续
    离线 4~7 天后 GRU 会用训练均值填满窗口、照常出分，预测退化成"平均的一天"。
    """
    import numpy as np
    import pandas as pd

    from src.baseline import inference
    from src.baseline.scaler_utils import get_feature_names

    names = get_feature_names("sleep")
    rows = []
    for day in (24, 25, 26):                      # 3 天正常
        r = {n: 1.0 for n in names}
        r.update(day_key=f"2026-06-{day:02d}", missing_count=0, data_quality="valid")
        rows.append(r)
    for day in (27, 28, 29, 30):                  # 4 天全 NaN 占位（设备离线）
        r = {n: np.nan for n in names}
        r.update(day_key=f"2026-06-{day:02d}", missing_count=8, data_quality="insufficient")
        rows.append(r)
    target = {n: 1.0 for n in names}
    target.update(day_key="2026-07-01", missing_count=0, data_quality="valid")
    rows.append(target)
    _fake_inference_ready(monkeypatch, pd.DataFrame(rows))

    result = inference.infer_track("T001", "2026-07-01", "sleep", config=_CONFIG)

    # 4 个全 NaN 占位日 > max_gap(3) → 不该静默用训练均值填满出分
    assert result["status"] == "data_insufficient"
    assert len(result["missing_window_days"]) == 4
