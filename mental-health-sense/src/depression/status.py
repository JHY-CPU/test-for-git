"""抑郁评估的状态枚举与"可展示"判据（单一事实来源）

每次评估给出一个 status，展示层据此决定要不要显示分数。

    assessed         正常出分（片段数达标、段间一致）
    low_confidence   出分了但证据弱：片段数不足，或段间分歧过大
    stale            超过 valid_days，结论已过期
    no_clip          有录像，但一段都没通过筛选（人脸/正脸/语音占比不达标）
    no_source        当天压根没有录像
    failed           截取或推理报错

★ 为什么要抽出这个模块（而不是在各处写字面量元组）

  本仓库在 `src/utils/status.py` 上吃过一次亏：「哪些 status 算可评估」曾以字面量
  形式散在 judge.py（两处）、rules.py、daily_job.py（两处）五个地方，结果
  `cold_start_fallback` 只被加进了其中两处，判定层收不到兜底轨的偏离，
  **建档期 35 天完全没有预警能力**（实测连续 6 天 score 166→58、偏离 6/6，
  risk_level 全 0）。检测层是对的，缺口在白名单不一致。详见 VALIDATION §8.2。

  抑郁通道从第一天就只留一份白名单，别处不许再写。

放在本包内而不是复用 utils/status.py：两套状态语义完全不同（那边是"该轨今天能否
参与判级判型"，这边是"这次评估结论能否对外显示"），共用一个枚举会诱导出
"抑郁状态也能参与判级"的误用，而那正是本包 __init__ 明令禁止的。
"""

STATUS_ASSESSED = "assessed"
STATUS_LOW_CONFIDENCE = "low_confidence"
STATUS_STALE = "stale"
STATUS_NO_CLIP = "no_clip"
STATUS_NO_SOURCE = "no_source"
STATUS_FAILED = "failed"

ALL_STATUSES = frozenset({
    STATUS_ASSESSED,
    STATUS_LOW_CONFIDENCE,
    STATUS_STALE,
    STATUS_NO_CLIP,
    STATUS_NO_SOURCE,
    STATUS_FAILED,
})

# 允许对外显示分数的状态。
#
# low_confidence 在内：它是"出了分但证据弱"，与 stale/no_clip 的"没有结论"性质
# 不同——藏起来会让家属以为系统没测，而实际上是测了、只是把握不大。展示层要靠
# 这个区分给出不同措辞（照抄 build_mpdd_evidence._quality 对 cold_start 的处理：
# 证据比 valid 弱但绝不是 missing）。
DISPLAYABLE_STATUSES = frozenset({
    STATUS_ASSESSED,
    STATUS_LOW_CONFIDENCE,
})

# 有结论但可信度打折的状态（展示时要额外加提示语）
WEAK_EVIDENCE_STATUSES = frozenset({
    STATUS_LOW_CONFIDENCE,
})

STATUS_LABELS = {
    STATUS_ASSESSED: "已评估",
    STATUS_LOW_CONFIDENCE: "已评估（证据不足）",
    STATUS_STALE: "评估已过期",
    STATUS_NO_CLIP: "无合格片段",
    STATUS_NO_SOURCE: "无录像",
    STATUS_FAILED: "评估失败",
}


def is_displayable(status: str | None) -> bool:
    """该状态能否对外显示分数。其余一律显示"暂无最新评估"。"""
    return status in DISPLAYABLE_STATUSES


def is_weak_evidence(status: str | None) -> bool:
    """有结论但需要额外加"证据不足"提示。"""
    return status in WEAK_EVIDENCE_STATUSES


def validate_status(status: str) -> str:
    """校验状态名，非法值立即报错而不是静默通过。

    与 scaler_utils.validate_track 同一思路：状态名打错（比如 "assess"）若静默
    通过，展示层会一路当成"不可显示"处理，家属看到"暂无评估"，而实际上评估是
    成功的——这种错误在日志里没有任何痕迹，极难定位。
    """
    if status not in ALL_STATUSES:
        raise ValueError(
            f"Unknown depression status: {status!r}. Expected one of {sorted(ALL_STATUSES)}"
        )
    return status
