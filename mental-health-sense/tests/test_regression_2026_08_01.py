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
