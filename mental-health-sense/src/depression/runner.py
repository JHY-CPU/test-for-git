"""抑郁评估编排：取片段 → 逐段推理 → 聚合 → 落盘

调用链：
    run_depression_assessment(elder_id, day_key, video=...)
      → ClipSource.collect()                     截片段（或读实时流产出）
      → _infer_one_clip() × N                    逐段调 infer_elder_depression.py
      → aggregate_clip_results()                 中位数 + 概率平均 + 离散度
      → build_contract() → store.save_assessment()

★ 全程不碰个人基线的任何东西：不读 features_*.csv、不写 residual_stats/ewma、
  不调 judge/rules。见 src/depression/__init__.py 的四条硬约束。
"""

import hashlib
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from src.depression.aggregate import aggregate_clip_results
from src.depression.clip_source import ClipSource, FileClipSource
from src.depression.contract import DATE_FMT, build_contract
from src.depression.mpdd_process import (
    EXIT_OK,
    MpddInvocationError,
    describe_failure,
    mpdd_git_rev,
    run_script,
)
from src.depression.status import (
    STATUS_FAILED,
    STATUS_NO_CLIP,
    STATUS_NO_SOURCE,
)
from src.depression.store import save_assessment
from src.utils.io import get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

INFER_SCRIPT = "infer_elder_depression.py"


def get_description_path(elder_id: str) -> Path:
    """个人介绍文本：data/depression/{elder_id}/description.txt

    MPDD 的 A-V+P 模型把这段文本编码成 1024 维 RoBERTa 嵌入当作第三路模态输入。
    **这份文本必须冻结**——改一个字分数就会变，而与老人的实际状态无关。
    契约里记 sha256 就是为了让"分数为什么跳了"这个问题可查。
    """
    return get_project_root() / "data" / "depression" / elder_id / "description.txt"


def _description_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_clip_result(json_path: Path) -> dict | None:
    """解析 infer_elder_depression.py 的输出 JSON。

    ★ 这个函数刻意与 subprocess 分离，为的是**能用 MPDD 真实产出的 JSON 做单测**
      （MPDD-AVG-2026/demo/result*.json 现成三份）。

      本仓 VALIDATION §8 记过一个教训：单测手搓的字典带着生产链路根本没写过的
      字段，于是断言全绿而缺陷活得好好的。解析层的测试如果也手搓输入，
      就会重演同一个错误——契约字段名写错、类型变了，测试照样通过。

    Returns:
        校验通过的结果字典；结构不对返回 None（该段计入 rejected，不中断整次评估）。
    """
    if not json_path.is_file():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"  └─ 片段结果无法解析: {json_path.name} ({e})")
        return None

    probs = data.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        logger.warning(f"  └─ 片段结果缺少 probabilities: {json_path.name}")
        return None
    if "depression_level" not in data:
        logger.warning(f"  └─ 片段结果缺少 depression_level: {json_path.name}")
        return None

    return data


def _infer_one_clip(
    dep_cfg: dict, clip_path: Path, description_file: Path, tmpdir: Path
) -> tuple[dict | None, bool]:
    """对单个片段跑 MPDD 推理。

    Returns:
        (结果字典或 None, 是否发生了 CUDA→CPU 静默降级)

    失败返回 (None, ...) 而不抛异常，让其余片段继续——3 段里崩 1 段仍能出结论。
    """
    out_json = clip_path.with_suffix(".json")
    args = [
        "--video", str(clip_path),
        "--description_file", str(description_file),
        "--output", str(out_json),
        "--device", str(dep_cfg.get("device", "cuda")),
        "--batch_size", str(dep_cfg.get("batch_size", 64)),
        "--frame_sample_rate", str(dep_cfg.get("frame_sample_rate", 1)),
    ]
    checkpoint = dep_cfg.get("checkpoint")
    if checkpoint:
        root = Path(dep_cfg["mpdd_root"]).expanduser()
        args += ["--checkpoint", str(root / checkpoint)]

    proc = run_script(
        dep_cfg, INFER_SCRIPT, args, tmpdir,
        timeout_sec=int(dep_cfg.get("timeout_sec", 1800)),
    )

    # infer_elder_depression.py:79-81 在 CUDA 不可用时**静默降级到 CPU**，
    # 只往 stderr 打一行 WARN 就继续跑。CPU 与 GPU 的浮点结果不同，若契约里记的
    # device 与实际不符，日后对比历史分数会得出错误结论——所以要从 stderr 里
    # 把这件事捞出来，记进 provenance。
    cpu_fallback = "CUDA 不可用" in (proc.stderr or "")

    if proc.returncode != EXIT_OK:
        logger.warning(f"  └─ [{clip_path.name}] {describe_failure(proc, INFER_SCRIPT)}")
        return None, cpu_fallback

    return parse_clip_result(out_json), cpu_fallback


def _maybe_alert(
    elder_id: str, day_key: str, payload: dict, dep_cfg: dict, config: dict
) -> None:
    """按配置把抑郁评估结果送进预警出口（当前默认关闭）。

    ★ 走**独立的事件流**（channel="depression"）。

      抑郁事件与基线事件各自计数、各自冷却、互不压制——理由同"绝不跨轨比较
      绝对分"：它们不是同一件事，不该共用一个计数器。一条线在冷却中，
      不该让另一条线的新事件被顺带静默。

      "独立"也包括**事件断段窗口**：本通道是稀疏事件驱动（几周才有一次合格
      片段），套用 baseline 的 max_skip_days=3 会让每次评估都判成新事件、
      去重完全失效。窗口改按 channel 取，见 alert._event_gap_days。

    ⚠️ `depression.alert` 默认 false，理由**不是**冷却缺失（那条已经解决），
      而是当前 checkpoint 在其验证集上对全部 9 个样本预测同一类别
      （Macro-F1 0.286），且未在本机位 / 本人身上校准。线接好、开关不开：
      换到可用的 checkpoint 后改一个配置项即可启用。

    本函数不改任何等级映射——抑郁等级到 risk_level 的换算刻意保守，
    见下方 _LEVEL_TO_RISK。
    """
    if not dep_cfg.get("alert", False):
        return

    from src.depression.status import is_displayable

    if not is_displayable(payload.get("status")):
        # 评估失败 / 无片段 / 已过期：我们不知道此刻的状况。
        # 既不该开事件，也**不该关**已有的事件——"测不到"不等于"好转"，
        # 同 judge._has_adverse_movement 拿不到方向元数据时不压等级。
        return

    level_name = (payload.get("result") or {}).get("level")

    # ★ 认不出的等级名必须原地返回，不能落到 `.get(..., 0)` 的 0 上。
    #
    #   删掉"risk_level<=0 就 return"之后，0 的语义变成了"缓解"——于是一个
    #   **不认识的**等级名会去关闭活跃事件并推一条「已回到个人常态范围」，
    #   与上面 is_displayable 那条"测不到不等于好转"自相矛盾。
    #   level 直接取自 checkpoint 的 probabilities 键名（aggregate._average_probabilities），
    #   换一个类别命名不同的 checkpoint 就会踩到。
    if level_name not in _LEVEL_TO_RISK:
        logger.error(
            f"  └─ 未知的抑郁等级 {level_name!r}（已知: {sorted(_LEVEL_TO_RISK)}），"
            f"本次不触发预警。checkpoint 的类别命名可能变了，需同步 _LEVEL_TO_RISK。"
        )
        return

    risk_level = _LEVEL_TO_RISK[level_name]

    # ★ risk_level == 0（评估判"正常"）也必须往下走，不能提前 return。
    #
    #   它是**缓解**：要靠 trigger_alert 把活跃事件关掉并发一条"过去了"。
    #   提前返回的后果与 daily_job 修过的那条完全同型（见 daily_job.py 第 7 步
    #   的注释："旧写法在 L0 那天直接跳过，事件永远停在最后一次 L2/L3 上"）——
    #   实测抑郁事件在评估回到"正常"后仍 active=True，周报的「预警回执」
    #   无限期显示"情绪状态评估 持续中，第 N 天"，家属也收不到缓解通知。
    try:
        from src.risk.alert import trigger_alert
        from src.risk.alert_state import CHANNEL_DEPRESSION

        trigger_alert(
            elder_id=elder_id,
            risk_level=risk_level,
            # 判"正常"时不带类型：此刻没有活跃的风险类型，与 daily_job 在 L0 那天
            # 传 risk_result["risk_types"]（空表）保持一致。
            risk_types=(
                [{"risk_key": "depression", "risk_type": "情绪状态评估"}]
                if risk_level > 0 else []
            ),
            config=config,
            day_key=day_key,
            channel=CHANNEL_DEPRESSION,
        )
    except Exception as e:
        # 预警是旁路的旁路：发不出去只该记日志，不能连累已经算好的评估结果
        logger.error(f"  └─ 抑郁预警触发失败（不影响评估产出）: {e}")


# 抑郁等级 → 风险等级的映射。刻意保守：
#   即使模型判"重度"也只到 L2（子女推送），不到 L3（强制响铃 + 网格员）。
#   群体绝对基线对个体差异没有免疫力，用它去惊动社区资源门槛应当更高。
_LEVEL_TO_RISK = {"正常": 0, "轻度": 1, "重度": 2, "抑郁": 2}


def run_depression_assessment(
    elder_id: str,
    day_key: str | None = None,
    video: str | Path | None = None,
    config: dict | None = None,
    clip_source: ClipSource | None = None,
) -> dict:
    """执行一次抑郁评估，落盘并返回契约字典。

    Args:
        day_key: 评估归属的自然日，默认今天
        video: 录像路径（FileClipSource 模式）。给了就用 FileClipSource
        clip_source: 直接注入片段来源（测试/将来的 StreamClipSource 用）

    Returns:
        契约字典。任何失败路径都会产出一份带相应 status 的契约——
        **不产出比产出一个"失败"更糟**：磁盘上没有文件时，展示层分不清
        "今天没评估"和"评估崩了"，运维也就无从发现故障。
    """
    if config is None:
        from src.utils.io import load_config
        config = load_config()

    dep_cfg = config.get("depression", {}) or {}
    if not dep_cfg.get("enabled", False):
        logger.info("抑郁评估未启用（depression.enabled=false），跳过")
        return {}

    day_key = day_key or datetime.now().strftime(DATE_FMT)
    assess_cfg = dep_cfg.get("assessment", {}) or {}
    valid_days = int(assess_cfg.get("valid_days", 30))

    logger.info(f"=== 抑郁评估启动: {elder_id} @ {day_key} ===")

    provenance = {
        "checkpoint": dep_cfg.get("checkpoint"),
        "description_sha256": None,
        "device": dep_cfg.get("device", "cuda"),
        "batch_size": dep_cfg.get("batch_size", 64),
        "frame_sample_rate": dep_cfg.get("frame_sample_rate", 1),
        "mpdd_git_rev": mpdd_git_rev(dep_cfg) if dep_cfg.get("mpdd_root") else "unknown",
    }
    empty_evidence = {
        "n_clips": 0, "n_rejected": 0, "source_duration_sec": None, "clips": [],
    }

    def _fail(status: str, reason: str) -> dict:
        logger.warning(f"  └─ {reason}")
        payload = build_contract(
            elder_id=elder_id, day_key=day_key, status=status,
            aggregated=None, evidence=empty_evidence, provenance=provenance,
            valid_days=valid_days, low_confidence_reasons=[reason],
        )
        save_assessment(payload)
        return payload

    # 1. 个人介绍（冻结文本）
    description_file = get_description_path(elder_id)
    if not description_file.is_file():
        return _fail(
            STATUS_FAILED,
            f"缺少个人介绍文本 {description_file}。MPDD 的 A-V+P 模型需要它作为"
            f"第三路模态输入；该文本一旦确定必须冻结（改动会改变分数）。",
        )
    description_text = description_file.read_text(encoding="utf-8").strip()
    if not description_text:
        return _fail(STATUS_FAILED, f"个人介绍文本为空: {description_file}")
    provenance["description_sha256"] = _description_sha256(description_text)

    # 2. 取候选片段
    if clip_source is None:
        if video is None:
            return _fail(STATUS_NO_SOURCE, "未提供录像，且实时流采集尚未接入")
        clip_source = FileClipSource(video, dep_cfg)

    tmp_root = Path(tempfile.mkdtemp(prefix="mhs_depression_"))
    try:
        try:
            clip_dir = clip_source.collect(elder_id, day_key, tmp_root)
        except MpddInvocationError as e:
            return _fail(STATUS_FAILED, f"片段截取失败: {e}")

        if clip_dir is None:
            return _fail(STATUS_NO_CLIP, "没有产出可用片段")

        index_path = clip_dir / "index.json"
        if not index_path.is_file():
            return _fail(STATUS_FAILED, f"片段目录缺少 index.json: {clip_dir}")
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        clips_meta = index.get("clips", [])
        if not clips_meta:
            return _fail(STATUS_NO_CLIP, "没有片段通过人脸/正脸/语音筛选")

        # 3. 逐段推理。失败的段计入 rejected 而不是中断——3 段里崩 1 段仍能出结论，
        #    而且 aggregate 会因片段数不足自动降级成 low_confidence。
        clip_results, kept_meta, rejected = [], [], 0
        saw_cpu_fallback = False
        for meta in clips_meta:
            clip_path = Path(meta["path"])
            try:
                result, cpu_fallback = _infer_one_clip(
                    dep_cfg, clip_path, description_file, tmp_root
                )
                saw_cpu_fallback = saw_cpu_fallback or cpu_fallback
            except MpddInvocationError as e:
                logger.warning(f"  └─ [{clip_path.name}] 推理失败: {e}")
                result = None
            if result is None:
                rejected += 1
                continue
            clip_results.append(result)
            kept_meta.append({
                "start": meta.get("start"),
                "end": meta.get("end"),
                "phq9": result.get("estimated_phq9"),
                "level": result.get("depression_level"),
                "face_ratio": meta.get("face_ratio"),
                "frontal_ratio": meta.get("frontal_ratio"),
            })

        if not clip_results:
            return _fail(STATUS_FAILED, f"全部 {rejected} 个片段推理均失败")

        # 4. 聚合
        aggregated = aggregate_clip_results(
            clip_results,
            min_clips=int(assess_cfg.get("min_clips", 2)),
            max_spread_phq9=float(assess_cfg.get("max_spread_phq9", 6.0)),
        )

        if saw_cpu_fallback:
            provenance["device"] = "cpu"
            logger.warning("  └─ CUDA 不可用，本次评估实际在 CPU 上完成（数值与 GPU 不同）")

        payload = build_contract(
            elder_id=elder_id,
            day_key=day_key,
            status=aggregated["status"],
            aggregated=aggregated["result"],
            evidence={
                "n_clips": len(clip_results),
                "n_rejected": rejected,
                "source_duration_sec": index.get("source_duration_sec"),
                "clips": kept_meta,
            },
            provenance=provenance,
            valid_days=valid_days,
            low_confidence_reasons=aggregated["low_confidence_reasons"],
        )
        save_assessment(payload)
        _maybe_alert(elder_id, day_key, payload, dep_cfg, config)
        logger.info(
            f"=== 抑郁评估完成: status={payload['status']}, "
            f"level={(payload.get('result') or {}).get('level')} ==="
        )
        return payload

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
