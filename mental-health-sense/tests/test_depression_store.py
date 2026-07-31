"""抑郁评估落盘、读取与有效期测试"""

import json
from pathlib import Path

import pytest

from src.depression import store
from src.depression.contract import build_contract, compute_valid_until
from src.depression.status import (
    STATUS_ASSESSED,
    STATUS_FAILED,
    STATUS_LOW_CONFIDENCE,
    STATUS_STALE,
)


@pytest.fixture
def tmp_log_dir(tmp_path, monkeypatch):
    """把落盘目录重定向到 tmp，避免污染仓库的 data/logs/。"""
    d = tmp_path / "depression"
    d.mkdir()
    monkeypatch.setattr(store, "get_depression_dir", lambda: d)
    return d


def _make(day_key: str, status: str = STATUS_ASSESSED, level: str = "正常",
          valid_days: int = 30, elder_id: str = "E001") -> dict:
    return build_contract(
        elder_id=elder_id, day_key=day_key, status=status,
        aggregated={"level": level, "phq9_median": 2.3, "phq9_spread": 1.1,
                    "class_probs": {"正常": 0.9, "轻度": 0.06, "重度": 0.04}},
        evidence={"n_clips": 3, "n_rejected": 1,
                  "source_duration_sec": 5400, "clips": []},
        provenance={"checkpoint": "x.pth", "description_sha256": "abc",
                    "device": "cuda", "batch_size": 64,
                    "frame_sample_rate": 1, "mpdd_git_rev": "deadbee"},
        valid_days=valid_days,
    )


class TestSaveLoad:
    def test_存取往返(self, tmp_log_dir):
        payload = _make("2026-08-12")
        store.save_assessment(payload)
        loaded = store.load_assessment("E001", "2026-08-12")
        assert loaded["result"]["level"] == "正常"
        assert loaded["provenance"]["description_sha256"] == "abc"

    def test_不存在返回_None(self, tmp_log_dir):
        assert store.load_assessment("E001", "2026-01-01") is None

    def test_写入是原子的(self, tmp_log_dir):
        """写完之后目录里不该残留临时文件。"""
        store.save_assessment(_make("2026-08-12"))
        assert [p.name for p in tmp_log_dir.iterdir()] == ["E001_2026-08-12.json"]

    def test_损坏文件降级为_None_而不抛(self, tmp_log_dir):
        """一个坏文件不该让周报整体崩掉——与 load_daily_results 同样的原则。"""
        (tmp_log_dir / "E001_2026-08-12.json").write_text("{截断的 js", encoding="utf-8")
        assert store.load_assessment("E001", "2026-08-12") is None


class TestValidUntil:
    def test_有效期含当日(self):
        assert compute_valid_until("2026-08-12", 1) == "2026-08-12"
        assert compute_valid_until("2026-08-12", 30) == "2026-09-10"

    def test_跨月跨年(self):
        assert compute_valid_until("2026-12-20", 30) == "2027-01-18"

    def test_非法天数报错(self):
        with pytest.raises(ValueError):
            compute_valid_until("2026-08-12", 0)


class TestLoadLatest:
    def test_取截至基准日的最近一次(self, tmp_log_dir):
        for d in ("2026-06-01", "2026-07-01", "2026-08-01"):
            store.save_assessment(_make(d))
        assert store.load_latest("E001", as_of="2026-07-15")["day_key"] == "2026-07-01"

    def test_基准日之后的评估不可见(self, tmp_log_dir):
        """as_of 必须由调用方给出：补生成历史周报时若用 now()，
        会把几个月后才做的评估算进那一周。"""
        store.save_assessment(_make("2026-08-01"))
        assert store.load_latest("E001", as_of="2026-07-15") is None

    def test_过期改写为_stale(self, tmp_log_dir):
        store.save_assessment(_make("2026-06-01", valid_days=30))  # 有效到 06-30
        latest = store.load_latest("E001", as_of="2026-08-01")
        assert latest["status"] == STATUS_STALE
        assert latest["original_status"] == STATUS_ASSESSED

    def test_有效期内不改写(self, tmp_log_dir):
        store.save_assessment(_make("2026-06-01", valid_days=30))
        assert store.load_latest("E001", as_of="2026-06-30")["status"] == STATUS_ASSESSED


class TestSummarizeForReport:
    def test_无评估时不可展示(self, tmp_log_dir):
        view = store.summarize_for_report("E001", as_of="2026-08-01")
        assert view["displayable"] is False
        assert "尚未" in view["reason"]

    def test_过期时不可展示且不给分(self, tmp_log_dir):
        """★ 绝不拿旧分顶替今天——与"周报无本周数据不得拿别的周顶替"同型。"""
        store.save_assessment(_make("2026-06-01", valid_days=30))
        view = store.summarize_for_report("E001", as_of="2026-08-01")
        assert view["displayable"] is False

    def test_失败状态不可展示(self, tmp_log_dir):
        payload = build_contract(
            elder_id="E001", day_key="2026-08-12", status=STATUS_FAILED,
            aggregated=None,
            evidence={"n_clips": 0, "n_rejected": 0,
                      "source_duration_sec": None, "clips": []},
            provenance={}, valid_days=30,
        )
        store.save_assessment(payload)
        assert store.summarize_for_report("E001", as_of="2026-08-12")["displayable"] is False

    def test_时间线按升序且只含可展示项(self, tmp_log_dir):
        store.save_assessment(_make("2026-06-01", level="正常"))
        store.save_assessment(_make("2026-07-01", status=STATUS_LOW_CONFIDENCE, level="轻度"))
        store.save_assessment(_make("2026-08-01", level="轻度"))
        view = store.summarize_for_report("E001", as_of="2026-08-15")
        assert view["displayable"] is True
        assert [t["day_key"] for t in view["timeline"]] == [
            "2026-06-01", "2026-07-01", "2026-08-01"
        ]
        assert [t["level"] for t in view["timeline"]] == ["正常", "轻度", "轻度"]

    def test_时间线不含基准日之后的评估(self, tmp_log_dir):
        store.save_assessment(_make("2026-06-01"))
        store.save_assessment(_make("2026-09-01"))
        view = store.summarize_for_report("E001", as_of="2026-06-15")
        assert [t["day_key"] for t in view["timeline"]] == ["2026-06-01"]
