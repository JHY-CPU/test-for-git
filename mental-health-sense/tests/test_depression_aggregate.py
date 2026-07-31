"""多片段聚合逻辑测试

★ 输入用的是 MPDD **真实产出**的 result JSON（MPDD-AVG-2026/demo/result*.json），
  不是手搓字典。

  本仓 VALIDATION §8 记过这个教训：「单测手搓的字典带着生产链路根本没写过的字段」，
  于是断言全绿而缺陷活得好好的。解析/聚合层如果也手搓输入，字段名写错、类型变了、
  上游改了结构，测试照样通过——测的是自己想象中的契约，不是真实的那个。

  只有在构造"极差过大""类别集合不一致"这类真实数据里没有的边界时才合成，
  且合成时以真实结构为模板。
"""

import json
from pathlib import Path

import pytest

from src.depression.aggregate import aggregate_clip_results
from src.depression.status import (
    STATUS_ASSESSED,
    STATUS_LOW_CONFIDENCE,
    STATUS_NO_CLIP,
)

MPDD_DEMO = Path("/home/zhousenyu/project/MPDD-AVG-2026/demo")
REAL_RESULTS = ["result.json", "result_derek_cole.json", "result_jimmy_carter.json"]


def _load_real(name: str) -> dict:
    path = MPDD_DEMO / name
    if not path.is_file():
        pytest.skip(f"MPDD demo 结果不存在（MPDD 仓未就位）: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def real_results() -> list[dict]:
    return [_load_real(name) for name in REAL_RESULTS]


class TestEmptyInput:
    def test_无片段返回_no_clip(self):
        out = aggregate_clip_results([])
        assert out["status"] == STATUS_NO_CLIP
        assert out["result"] is None


class TestRealMpddOutputs:
    def test_三段真实结果能聚合(self, real_results):
        out = aggregate_clip_results(real_results, min_clips=2, max_spread_phq9=99.0)
        assert out["status"] == STATUS_ASSESSED
        assert out["result"]["level"] in ("正常", "轻度", "重度")

    def test_phq9_取中位数(self, real_results):
        """三段实际值 4.724 / 2.346 / 2.665 → 中位数 2.665，而平均是 3.245。

        中位数抗离群：某段正好赶上老人扭头或背景电视有人说话时不会带偏结论。
        """
        values = sorted(r["estimated_phq9"] for r in real_results)
        out = aggregate_clip_results(real_results, min_clips=2, max_spread_phq9=99.0)
        assert out["result"]["phq9_median"] == pytest.approx(values[1], abs=1e-3)

    def test_必须报离散度(self, real_results):
        """段间打架本身是信息，只报中位数会把它丢掉。"""
        values = [r["estimated_phq9"] for r in real_results]
        out = aggregate_clip_results(real_results, min_clips=2, max_spread_phq9=99.0)
        assert out["result"]["phq9_spread"] == pytest.approx(
            max(values) - min(values), abs=1e-3
        )

    def test_类别用概率平均后argmax(self, real_results):
        out = aggregate_clip_results(real_results, min_clips=2, max_spread_phq9=99.0)
        probs = out["result"]["class_probs"]
        assert out["result"]["level"] == max(probs, key=probs.get)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)


class TestLowConfidence:
    def test_单段降级(self, real_results):
        out = aggregate_clip_results(real_results[:1], min_clips=2)
        assert out["status"] == STATUS_LOW_CONFIDENCE
        assert any("片段数" in r for r in out["low_confidence_reasons"])
        # 降级不等于不出结论——仍要给分，只是标注证据弱
        assert out["result"] is not None

    def test_离散度过大降级(self, real_results):
        out = aggregate_clip_results(
            real_results, min_clips=2, max_spread_phq9=0.5
        )
        assert out["status"] == STATUS_LOW_CONFIDENCE
        assert any("极差" in r for r in out["low_confidence_reasons"])

    def test_两个原因可同时命中(self, real_results):
        out = aggregate_clip_results(
            real_results[:1], min_clips=2, max_spread_phq9=0.001
        )
        # 单段时极差恒为 0，不该触发极差那条
        assert len(out["low_confidence_reasons"]) == 1


class TestMalformedInput:
    def test_类别集合不一致要报错(self, real_results):
        """binary（正常/抑郁）与 ternary（正常/轻度/重度）混在一起是配置错误。

        静默按并集处理会给出一个四类别的假分布，看上去像模像样、实际毫无意义。
        """
        mixed = [dict(real_results[0]), dict(real_results[1])]
        mixed[1] = {**mixed[1], "probabilities": {"正常": 0.6, "抑郁": 0.4}}
        with pytest.raises(ValueError, match="类别集合不一致"):
            aggregate_clip_results(mixed)

    def test_缺回归头时_phq9_为_None(self, real_results):
        """checkpoint 没有回归头时 estimated_phq9 是 None（infer 脚本的 else 分支）。

        拿 None 去算中位数会 TypeError，而那时堆栈离真正的原因已经很远。
        """
        no_phq = [{**r, "estimated_phq9": None} for r in real_results]
        out = aggregate_clip_results(no_phq, min_clips=2)
        assert out["result"]["phq9_median"] is None
        assert out["status"] == STATUS_ASSESSED

    def test_缺_probabilities_要报错(self):
        with pytest.raises(ValueError, match="probabilities"):
            aggregate_clip_results([{"depression_level": "正常"}])
