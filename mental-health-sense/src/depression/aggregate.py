"""多片段 → 单次评估结论

一次评估会截出 N 段（默认 top_k=3）分别推理，本模块把 N 份 MPDD 输出压成一个结论。

★ 纯函数、零重依赖（只用标准库 statistics）。这样它能被单测覆盖、能进 CI，
  不需要装 torch/transformers，也不需要 MPDD 环境在场。

三条聚合规则的理由：

1. **PHQ-9 取中位数而非平均**
   片段质量参差：某一段可能正好赶上老人扭头、或背景电视有人说话。中位数抗离群，
   3 段里 1 段翻车不会带偏结论；平均会。

2. **类别取"概率平均后 argmax"而非多数投票**
   2~3 段做多数投票必然出现平局（1:1、1:1:1），而平局的 tie-break 规则怎么定都
   是任意的。概率平均把每段的把握度也算进去，一段 0.95 的"正常"应当压过一段
   0.4 的"轻度"，投票做不到这件事。

3. **必须同时报离散度（max−min）**
   段间打架本身就是信息：它说明模型对这个人没把握。只报一个中位数会把这个
   信号丢掉——这与本仓库"用 abs 判方向会让好转与恶化无法区分"是同一类错误：
   合成统计量时把该保留的维度压掉了。
"""

from statistics import median

from src.depression.status import (
    STATUS_ASSESSED,
    STATUS_LOW_CONFIDENCE,
    STATUS_NO_CLIP,
)


def _clip_phq9(clip_result: dict) -> float | None:
    """取单段的 PHQ-9。

    `estimated_phq9` 在 checkpoint 没有回归头时为 None（infer_elder_depression.py
    的 else 分支）。当前所有 Track1 checkpoint 都带回归头，但不能假设它一定在——
    拿 None 去算中位数会直接 TypeError，而那时错误堆栈离真正的原因（checkpoint
    没有回归头）已经很远了。
    """
    value = clip_result.get("estimated_phq9")
    if value is None:
        return None
    return float(value)


def _average_probabilities(clip_results: list[dict]) -> dict[str, float]:
    """逐类别求概率平均。

    要求所有片段的类别集合一致——binary（正常/抑郁）与 ternary（正常/轻度/重度）
    的结果混在一起是配置错误，必须炸出来而不是静默按并集处理：并集会给出一个
    四类别的假分布，看上去像模像样，实际毫无意义。
    """
    if not clip_results:
        return {}

    key_sets = [frozenset((r.get("probabilities") or {}).keys()) for r in clip_results]
    if len(set(key_sets)) > 1:
        raise ValueError(
            f"片段间类别集合不一致，无法聚合：{[sorted(k) for k in key_sets]}。"
            "通常是 binary 与 ternary 的结果被混进了同一次评估。"
        )

    names = sorted(key_sets[0])
    n = len(clip_results)
    return {
        name: sum(float(r["probabilities"][name]) for r in clip_results) / n
        for name in names
    }


def aggregate_clip_results(
    clip_results: list[dict],
    min_clips: int = 2,
    max_spread_phq9: float = 6.0,
) -> dict:
    """把 N 份 MPDD 单片段输出聚合成一次评估结论。

    Args:
        clip_results: `infer_elder_depression.py` 的输出 JSON 列表（已解析）
        min_clips: 低于此片段数标 low_confidence
        max_spread_phq9: 段间 PHQ-9 极差超过它标 low_confidence

    Returns:
        {
            "status": str,                    # assessed / low_confidence / no_clip
            "result": dict | None,            # 无片段时为 None
            "low_confidence_reasons": list[str],
        }

    注意 status 只可能是这三个：stale 由 store 按有效期判定、no_source/failed
    由 runner 判定。本函数只回答"拿到这些片段能得出什么结论"。
    """
    if not clip_results:
        return {
            "status": STATUS_NO_CLIP,
            "result": None,
            "low_confidence_reasons": ["没有片段通过筛选"],
        }

    probs = _average_probabilities(clip_results)
    if not probs:
        raise ValueError("片段结果缺少 probabilities 字段，无法聚合")

    # 概率平均后 argmax。并列时取字典序靠前的——这种情况实际不会出现（浮点均值
    # 恰好相等的概率为零），但确定性优先：不能让结果依赖 dict 的迭代顺序。
    level = max(sorted(probs), key=lambda name: probs[name])

    phq9_values = [v for v in (_clip_phq9(r) for r in clip_results) if v is not None]
    phq9_median = round(median(phq9_values), 3) if phq9_values else None
    phq9_spread = (
        round(max(phq9_values) - min(phq9_values), 3) if len(phq9_values) >= 2 else 0.0
    )

    reasons: list[str] = []
    if len(clip_results) < min_clips:
        reasons.append(
            f"片段数 {len(clip_results)} < {min_clips}，单段结论不可靠"
        )
    if phq9_median is not None and phq9_spread > max_spread_phq9:
        reasons.append(
            f"段间 PHQ-9 极差 {phq9_spread} > {max_spread_phq9}，模型对该样本分歧较大"
        )

    return {
        "status": STATUS_LOW_CONFIDENCE if reasons else STATUS_ASSESSED,
        "result": {
            "level": level,
            "phq9_median": phq9_median,
            "phq9_spread": phq9_spread,
            "class_probs": {k: round(v, 6) for k, v in probs.items()},
        },
        "low_confidence_reasons": reasons,
    }
