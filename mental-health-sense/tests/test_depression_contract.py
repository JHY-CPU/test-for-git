"""抑郁评估契约的组装与校验测试"""

import pytest

from src.depression.contract import (
    CONTRACT_NOTE,
    REQUIRED_TOP_LEVEL,
    SCHEMA_VERSION,
    build_contract,
    validate_contract,
)
from src.depression.status import STATUS_ASSESSED, STATUS_NO_CLIP


def _minimal(**overrides) -> dict:
    kwargs = dict(
        elder_id="E001", day_key="2026-08-12", status=STATUS_ASSESSED,
        aggregated={"level": "正常", "phq9_median": 2.3, "phq9_spread": 1.1,
                    "class_probs": {"正常": 0.99, "轻度": 0.005, "重度": 0.005}},
        evidence={"n_clips": 3, "n_rejected": 1,
                  "source_duration_sec": 5400, "clips": []},
        provenance={"checkpoint": "Track1/A-V-P/ternary/best.pth",
                    "description_sha256": "a3f1", "device": "cuda",
                    "batch_size": 64, "frame_sample_rate": 1,
                    "mpdd_git_rev": "deadbee"},
        valid_days=30,
    )
    kwargs.update(overrides)
    return build_contract(**kwargs)


class TestBuild:
    def test_包含全部必需字段(self):
        payload = _minimal()
        assert all(k in payload for k in REQUIRED_TOP_LEVEL)

    def test_assessed_at_默认取_day_key_而非_now(self):
        """跑推理的时刻可能是次日凌晨批处理、甚至几天后的补算。
        用 now() 会让"最近一次评估"的时间轴整体漂移。"""
        assert _minimal()["assessed_at"].startswith("2026-08-12")

    def test_契约自带不可合并的声明(self):
        """即使有人只拿到一个 JSON、没读代码，也该知道它不能拿去做什么。
        与 judge.build_mpdd_evidence 的 note 对称。"""
        assert _minimal()["note"] == CONTRACT_NOTE
        assert "不可比" in CONTRACT_NOTE

    def test_未校准警示恒在(self):
        """字段在，周报就必须渲染警示语——当前 checkpoint 在其验证集上
        对全部样本预测同一类别，拿它当结论会误导家属。"""
        cal = _minimal()["calibration"]
        assert cal["validated_on_site"] is False
        assert cal["warning"]

    def test_无结论时_result_为_None(self):
        payload = _minimal(status=STATUS_NO_CLIP, aggregated=None)
        assert payload["result"] is None

    def test_非法状态在组装时就报错(self):
        with pytest.raises(ValueError, match="Unknown depression status"):
            _minimal(status="nope")


class TestValidate:
    def test_合法契约通过(self):
        payload = _minimal()
        assert validate_contract(payload) is payload

    @pytest.mark.parametrize("missing", REQUIRED_TOP_LEVEL)
    def test_缺任一必需字段都报错(self, missing):
        """契约跨进程、跨仓库，写入方和读取方可能是不同版本。
        结构漂了要在读到的那一刻炸，而不是在周报渲染到一半时抛 KeyError。"""
        payload = _minimal()
        payload.pop(missing)
        with pytest.raises(ValueError, match="缺少字段"):
            validate_contract(payload)

    def test_主版本不兼容要拒绝(self):
        payload = _minimal()
        payload["schema_version"] = "2.0.0"
        with pytest.raises(ValueError, match="主版本不兼容"):
            validate_contract(payload)

    def test_同主版本的小升级放行(self):
        """新增字段应当向后兼容，不该因为小版本号不同就拒读历史契约。"""
        payload = _minimal()
        major = SCHEMA_VERSION.split(".")[0]
        payload["schema_version"] = f"{major}.9.9"
        assert validate_contract(payload)
