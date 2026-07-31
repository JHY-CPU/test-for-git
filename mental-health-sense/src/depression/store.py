"""抑郁评估结果的落盘与读取

★ 本模块是**展示层唯一允许 import 的入口**（连同 status / contract / aggregate）。

  它只用标准库 + src.utils.io 的路径与原子写 helper，绝不 import runner /
  clip_source——那两个会拉起 subprocess 与外部仓。周报必须在 MPDD 环境完全不可用
  （仓库被移走、conda env 坏了、GPU 掉了）时照常渲染，只是那一节显示"暂无最新评估"。

  这与"单轨失败是正常降级，两轨同时不可用才算整体失败"是同一条原则：
  抑郁这条线不该有能力拖垮周报。
"""

import json
from datetime import datetime
from pathlib import Path

from src.depression.contract import DATE_FMT, validate_contract
from src.depression.status import STATUS_STALE, is_displayable
from src.utils.io import atomic_write_json, get_log_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

LOG_TYPE = "depression"


def get_depression_dir() -> Path:
    """复用 io.get_log_dir，不另造路径函数——路径推导必须只有一处。"""
    return get_log_dir(LOG_TYPE)


def save_assessment(payload: dict) -> Path:
    """校验后原子落盘。

    先校验再写：宁可不产出，也不要在磁盘上留一份结构不对的契约。半成品契约比
    没有契约更糟——下游会读到它、按残缺结构渲染，而"文件存在"这件事本身
    会让人以为评估成功了。
    """
    validate_contract(payload)
    filepath = get_depression_dir() / f"{payload['elder_id']}_{payload['day_key']}.json"
    atomic_write_json(filepath, payload)
    logger.info(
        f"  └─ 抑郁评估已落盘: {filepath.name} "
        f"(status={payload['status']}, level={(payload.get('result') or {}).get('level')})"
    )
    return filepath


def load_assessment(elder_id: str, day_key: str) -> dict | None:
    """读某一天的评估；不存在或损坏返回 None。

    损坏时降级为 None + 一条 ERROR，而不是抛异常：这是旁路产物，一个坏文件
    不该让周报整体崩掉。主链路 load_daily_results 也是同样的处理原则。
    """
    filepath = get_depression_dir() / f"{elder_id}_{day_key}.json"
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return validate_contract(json.load(f))
    except (json.JSONDecodeError, ValueError, OSError) as e:
        logger.error(f"  └─ 抑郁评估文件损坏，已跳过: {filepath.name} ({e})")
        return None


def load_history(elder_id: str, limit: int = 12) -> list[dict]:
    """按 day_key 升序返回最近 N 次评估（跳过损坏文件）。

    给周报的"历史时间线"用。为什么要时间线：MPDD 是**群体绝对基线**，对个体差异
    没有免疫力——一个天生表情少、语速慢、口音重的老人可能常年被判"轻度"，
    但他一直就这样，什么都没变。只看单次的绝对等级会把这种人固定误报成风险。

    把历次评估排成序列看"变没变"而不是"高不高"，等于用绝对基线的输出做了一层
    最轻量的个人趋势观察。注意这**只是展示层排序**，不涉及任何拟合、不产生
    个人基线——真做个人基线就违反了本包的硬约束，而且抑郁评估天然稀疏，
    样本量也根本估不出来。
    """
    log_dir = get_depression_dir()
    if not log_dir.exists():
        return []

    files = sorted(log_dir.glob(f"{elder_id}_*.json"), reverse=True)[:limit]
    out = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                out.append(validate_contract(json.load(f)))
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.error(f"  └─ 抑郁评估文件损坏，已跳过: {fp.name} ({e})")
    return list(reversed(out))


def load_latest(elder_id: str, as_of: str | None = None) -> dict | None:
    """取截至 as_of（含）的最近一次评估，并按有效期改写 status。

    Args:
        as_of: 基准日 "YYYY-MM-DD"。默认今天。
            **必须由调用方给出**而不是内部取 now()：周报是按周生成的，
            补生成历史周报时若用 now()，会把几个月后才做的评估算进那一周，
            与 judge 的 `today_key` 必须由调用方传入是同一个理由
            （该轨今天恰好不可用时，末条是更早的一天，窗口会跟着往前漂）。

    Returns:
        契约字典；超过 valid_until 的把 status 改写成 stale（原始 status 保留在
        `original_status` 里供排查）。找不到返回 None。
    """
    as_of = as_of or datetime.now().strftime(DATE_FMT)

    candidates = [
        a for a in load_history(elder_id, limit=64)
        if a.get("day_key") and a["day_key"] <= as_of
    ]
    if not candidates:
        return None

    latest = dict(candidates[-1])

    valid_until = latest.get("valid_until")
    if valid_until and as_of > valid_until:
        latest["original_status"] = latest["status"]
        latest["status"] = STATUS_STALE

    return latest


def summarize_for_report(elder_id: str, as_of: str | None = None) -> dict:
    """给周报的展示视图（周报只调这一个函数）。

    Returns:
        {
            "displayable": bool,       # False 时周报显示"暂无最新评估"
            "latest": dict | None,
            "timeline": [{"day_key": str, "level": str}, ...],   # 仅可展示的历次
            "reason": str,             # displayable=False 时的原因文案
        }
    """
    latest = load_latest(elder_id, as_of=as_of)

    if latest is None:
        return {
            "displayable": False, "latest": None, "timeline": [],
            "reason": "尚未进行过情绪状态评估",
        }

    if not is_displayable(latest.get("status")):
        from src.depression.status import STATUS_LABELS
        return {
            "displayable": False, "latest": latest, "timeline": [],
            "reason": STATUS_LABELS.get(latest["status"], "暂无最新评估"),
        }

    timeline = [
        {"day_key": a["day_key"], "level": (a.get("result") or {}).get("level", "—")}
        for a in load_history(elder_id, limit=12)
        if is_displayable(a.get("status")) and a.get("result")
        and a.get("day_key") and (as_of is None or a["day_key"] <= as_of)
    ]

    return {
        "displayable": True, "latest": latest, "timeline": timeline, "reason": "",
    }
