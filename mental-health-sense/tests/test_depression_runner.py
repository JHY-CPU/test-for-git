"""抑郁评估编排测试

分两层：
  - 解析层（parse_clip_result）：用 MPDD **真实产出**的 JSON 做输入，无需 MPDD 环境
  - 端到端（run_depression_assessment）：真调 MPDD，标 integration + slow

为什么解析层要单独测：本仓 VALIDATION §8 的教训是"断言要打在真实链路的输出上"。
手搓字典测出来的绿色，证明的只是"我想象中的契约成立"。
"""

import json
from pathlib import Path

import pytest

from src.depression.runner import parse_clip_result, run_depression_assessment
from src.depression.status import STATUS_FAILED, STATUS_NO_SOURCE

MPDD_DEMO = Path("/home/zhousenyu/project/MPDD-AVG-2026/demo")


class TestParseClipResult:
    """解析 infer_elder_depression.py 的输出。"""

    @pytest.mark.parametrize(
        "name", ["result.json", "result_derek_cole.json", "result_jimmy_carter.json"]
    )
    def test_真实输出能解析(self, name):
        path = MPDD_DEMO / name
        if not path.is_file():
            pytest.skip(f"MPDD demo 结果不存在: {path}")
        parsed = parse_clip_result(path)
        assert parsed is not None
        assert parsed["depression_level"] in ("正常", "轻度", "重度", "抑郁")
        assert isinstance(parsed["probabilities"], dict)

    def test_文件不存在返回_None(self, tmp_path):
        assert parse_clip_result(tmp_path / "nope.json") is None

    def test_损坏_JSON_返回_None_而不抛(self, tmp_path):
        """单段解析失败只该让这一段计入 rejected，不能中断整次评估。"""
        bad = tmp_path / "clip.json"
        bad.write_text("{截断", encoding="utf-8")
        assert parse_clip_result(bad) is None

    def test_缺_probabilities_返回_None(self, tmp_path):
        p = tmp_path / "clip.json"
        p.write_text(json.dumps({"depression_level": "正常"}), encoding="utf-8")
        assert parse_clip_result(p) is None

    def test_缺_depression_level_返回_None(self, tmp_path):
        p = tmp_path / "clip.json"
        p.write_text(json.dumps({"probabilities": {"正常": 1.0}}), encoding="utf-8")
        assert parse_clip_result(p) is None


class TestDisabledAndFailurePaths:
    """不需要 MPDD 环境的编排分支。"""

    def test_未启用时返回空且不落盘(self):
        assert run_depression_assessment(
            "E001", "2026-08-12", config={"depression": {"enabled": False}}
        ) == {}

    def test_缺个人介绍时产出_failed_契约(self, tmp_path, monkeypatch):
        """★ 任何失败路径都要产出一份带相应 status 的契约。

        不产出比产出一个"失败"更糟：磁盘上没有文件时，展示层分不清
        "今天没评估"和"评估崩了"，运维也就无从发现故障。
        """
        from src.depression import runner, store

        monkeypatch.setattr(store, "get_depression_dir", lambda: tmp_path)
        monkeypatch.setattr(
            runner, "get_description_path", lambda eid: tmp_path / "missing.txt"
        )
        result = run_depression_assessment(
            "E001", "2026-08-12",
            config={"depression": {"enabled": True, "assessment": {"valid_days": 30}}},
        )
        assert result["status"] == STATUS_FAILED
        assert (tmp_path / "E001_2026-08-12.json").is_file()

    def test_无录像且无实时流时产出_no_source(self, tmp_path, monkeypatch):
        from src.depression import runner, store

        monkeypatch.setattr(store, "get_depression_dir", lambda: tmp_path)
        desc = tmp_path / "description.txt"
        desc.write_text("An older adult who speaks calmly.", encoding="utf-8")
        monkeypatch.setattr(runner, "get_description_path", lambda eid: desc)

        result = run_depression_assessment(
            "E001", "2026-08-12",
            config={"depression": {"enabled": True, "assessment": {"valid_days": 30}}},
        )
        assert result["status"] == STATUS_NO_SOURCE
        # 介绍文本的指纹必须落进 provenance——它是"分数为什么跳了"的唯一线索
        assert result["provenance"]["description_sha256"]


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEnd:
    """真调 MPDD 全链路。需要 MPDD 仓 + conda 环境 + GPU，很慢（分钟级）。"""

    def test_demo_录像端到端(self, tmp_path, monkeypatch):
        from src.utils.io import load_config

        video = MPDD_DEMO / "elderly_front_derek_cole.webm"
        if not video.is_file():
            pytest.skip("MPDD demo 录像不存在")

        config = load_config()
        dep_cfg = config.get("depression", {})
        if not Path(dep_cfg.get("python_bin", "")).is_file():
            pytest.skip("MPDD 解释器不存在")

        from src.depression import runner, store

        monkeypatch.setattr(store, "get_depression_dir", lambda: tmp_path)
        desc = tmp_path / "description.txt"
        desc.write_text(
            (MPDD_DEMO / "elder_description_derek_cole.txt").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        monkeypatch.setattr(runner, "get_description_path", lambda eid: desc)

        result = run_depression_assessment(
            "E2E", "2026-08-12", video=video, config=config
        )
        assert result["status"] in ("assessed", "low_confidence", "no_clip")
        assert (tmp_path / "E2E_2026-08-12.json").is_file()
