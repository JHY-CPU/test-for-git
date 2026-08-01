"""
周报生成模块

每周日趋势轨执行后调用：
    1. 取最近7天推理结果
    2. 计算各维度趋势
    3. 填充Prompt → LLM调用（或fallback规则生成）
    4. 保存Markdown周报
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.report.templates import (
    WEEKLY_REPORT_SYSTEM_PROMPT,
    WEEKLY_REPORT_USER_TEMPLATE,
    fill_prompt,
    generate_rule_based_report,
)
from src.risk.judge import judge_risk_level
from src.utils.io import get_log_dir, load_daily_results
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 找目标周时往回捞多少天的日志。给足冗余量，让"补算某个历史周"也能命中。
_LOOKBACK_DAYS = 60


def generate_weekly_report(
    elder_id: str,
    week_start: str | None = None,
    week_end: str | None = None,
    use_llm: bool = True,
    config: dict | None = None,
) -> str:
    """
    生成每周心理健康周报。

    Args:
        elder_id: 老人ID
        week_start: 周起始日期 "YYYY-MM-DD"，默认计算最近7天
        week_end: 周结束日期 "YYYY-MM-DD"
        use_llm: 是否使用LLM生成（False时使用规则模板）
        config: 全局配置。**不传就在这里自己加载**，见下方注释。

    Returns:
        Markdown格式的周报文本
    """
    # ★ config 兜底必须在这里做，而不是指望每个调用方都记得传。
    #
    #   `scripts/run_weekly_pipeline.py` 就没传，于是一路 None 传到
    #   `_format_depression_section`，它第一行 `(config or {}).get("depression")`
    #   取空后**整节 return ""** —— 抑郁小节连同 contract.py 明令"字段在就必须
    #   渲染"的那句校准警示语一起静默消失。实测：
    #       config=load_config() → '## 情绪状态评估（研究性，未校准）…'
    #       config=None          → ''
    #   而 settings.yaml 里 depression.enabled / show_in_report 都是 true。
    #
    #   修在这里而不是脚本里：保护**所有**调用方（脚本、weekly_job、将来的补算
    #   工具），与 daily_job.run_daily_pipeline 自己 load_config 是同一条惯例。
    #   下方 `_generate_with_llm` 原有的同款自救已删除——兜底只留这一处，
    #   两份会漂的兜底比没有更危险（同 imputer 删掉 check_offline_status 的理由）。
    #
    #   读不出配置时回落空字典而不是抛：周报是给家属看的产物，
    #   配置坏了应当出一份缺章节的周报，而不是整周没有周报。
    if config is None:
        from src.utils.io import load_config
        try:
            config = load_config()
        except Exception as e:
            logger.error(f"周报读取配置失败，部分章节将缺失: {e}")
            config = {}

    # 计算日期范围
    if week_end is None:
        week_end_dt = datetime.now()
    else:
        week_end_dt = datetime.strptime(week_end, "%Y-%m-%d")

    if week_start is None:
        week_start_dt = week_end_dt - timedelta(days=6)
    else:
        week_start_dt = datetime.strptime(week_start, "%Y-%m-%d")

    week_start_str = week_start_dt.strftime("%Y-%m-%d")
    week_end_str = week_end_dt.strftime("%Y-%m-%d")

    logger.info(f"周报生成: elder_id={elder_id}, {week_start_str} -> {week_end_str}")

    # 1. 加载推理结果。
    # 取的天数要明显多于 7：load_daily_results 拿的是最近 N 个**文件**，
    # 只取 7 个的话，一旦最新日志比目标周更新（补算历史周、或最近几天没跑），
    # 目标周的记录根本不在候选里。
    try:
        daily_results = load_daily_results(elder_id, n_days=_LOOKBACK_DAYS)
    except Exception:
        daily_results = []

    # 过滤到本周范围内（day_key 为新字段名，兼容旧日志的 date）
    week_results = [
        r for r in daily_results
        if week_start_str <= (r.get("day_key") or r.get("date") or "") <= week_end_str
    ]

    # ★ 本周确实没有数据时出"数据不足"周报，绝不拿别的周顶替。
    # 旧实现在这里回落到 daily_results[-7:]，于是周报**标题写着这一周、
    # 正文却是另一周的数据**——实测标题 2026-07-25~07-31，内容是 8 月下旬
    # 社交异常期的分。家属看到的是一份日期与内容对不上的报告，
    # 而这正是"测不到"被静默呈现成"有结论"的又一种形态。
    if not week_results:
        logger.warning(
            f"  └─ {week_start_str}~{week_end_str} 无推理记录，出数据不足周报"
        )
        report = _empty_report(elder_id, week_start_str, week_end_str, config)
        _save_report(elder_id, week_start_str, report)
        return report

    # 2. 计算各维度趋势
    trends = _compute_weekly_trends(week_results)

    # 3. 风险判定
    #
    # ★ 用**完整判定窗**而不是本周这 7 条。
    #
    #   传 week_results 会让 _walk_back 一旦回溯到 week_start 之前就按缺日跳过、
    #   连跳 max_skip 天后 return，计数上限被硬压在本周长度。后果是周报与预警
    #   自相矛盾：circadian 降级模式需要连续 7 天达标，日轨用 25 天窗算出
    #   consecutive=8 发了「作息节律紊乱」提醒，同一周的周报只有 7 条日志、
    #   其中任一天 degraded 就只数到 6 → 周报写"本周风险类型：无"。
    #   家属唯一会读的产物与他们收到的提醒对不上。这与 VALIDATION §8.6 记录的
    #   是同一个缺陷，只是换了个入口。
    #
    #   判定基准日固定为 week_end：补生成历史周报时不能拿"最新那天"当今天。
    risk_result = judge_risk_level(
        elder_id, daily_results=None, config=config, today_key=week_end_str
    )

    # 4. 风险类型名称
    risk_type_names = [
        rt.get("risk_type", "")
        for rt in risk_result.get("risk_types", [])
    ]
    risk_types_str = "、".join(risk_type_names) if risk_type_names else "无"

    # 5. 生成报告文本
    if use_llm:
        report_body = _generate_with_llm(
            elder_id=elder_id,
            trends=trends,
            risk_result=risk_result,
            risk_types_str=risk_types_str,
            config=config,
        )
    else:
        report_body = generate_rule_based_report(
            elder_id=elder_id,
            week_start=week_start_str,
            week_end=week_end_str,
            risk_label=risk_result.get("risk_label", "正常"),
            risk_types=risk_type_names,
            deviation_days=sum(1 for r in week_results if r.get("is_deviation", False)),
            # trends 的键就是 generate_rule_based_report 的形参名
            # （social_trend / sleep_trend / social_vs_baseline ...），
            # 旧写法再补一个 _trend 后缀会拼出 social_trend_trend，直接 TypeError。
            **trends,
        )

    # 6. 组装完整Markdown周报
    report = f"""# {elder_id} 心理健康周报

**周期**：{week_start_str} ~ {week_end_str}
**风险等级**：{risk_result.get('risk_label', '正常')}
**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 本周概况

{report_body}

---

## 监测详情

| 维度 | 周内走向 | 相对个人常态 |
| :--- | :--- | :--- |
| 社交互动 | {trends.get('social_trend', '平稳')} | 社交方面本周整体{trends.get('social_vs_baseline', '无明显差异')} |
| 睡眠质量 | {trends.get('sleep_trend', '平稳')} | 睡眠方面本周整体{trends.get('sleep_vs_baseline', '无明显差异')} |
| 日常活动 | {trends.get('activity_trend', '平稳')} | 活动方面本周整体{trends.get('activity_vs_baseline', '无明显差异')} |

## 统计指标

{_format_track_stats(week_results)}

{_format_alert_receipts(elder_id, week_end_str, config)}
{_format_depression_section(elder_id, week_end_str, config)}
## 处置建议

{risk_result.get('recommendation', '无特殊建议')}
"""

    # 7. 保存周报
    _save_report(elder_id, week_start_str, report)
    logger.info(f"  └─ 周报已保存: {elder_id}_{week_start_str}")

    return report


_CHANNEL_LABELS = {
    "baseline": "睡眠 / 社会连接",
    "depression": "情绪状态",
}

_RISK_KEY_LABELS = {
    "sleep_stability": "睡眠稳定性偏离",
    "social_decline": "社会连接减弱",
    "circadian_disruption": "作息节律紊乱",
    "depression": "情绪状态评估",
}


def _format_alert_receipts(elder_id: str, as_of: str, config: dict | None = None) -> str:
    """持续中事件的周期性回执。

    ★ 这一节是「按事件去重」这个设计的**配套义务**，不是可选装饰。

      通知层现在只在事件开始 / 升级 / 新类型 / 缓解时推送，同级持续静默。
      如果周报里也一字不提，家属就无法区分「系统安静」与「系统挂了」——
      这与本仓「长期降级和长期正常必须可区分」是同一条原则
      （daily_job 的 check_prolonged_degradation 就是为它而写）。

      所以静默期必须有一个低强度、不打扰的告知出口：进周报、不推送、不响铃。

    照抄 _format_depression_section 已确立的范式：独立小节 + 自己拿数据 +
    try/except 返回 ""，异常不拖垮整份周报。
    """
    events_cfg = ((config or {}).get("alert") or {}).get("events") or {}
    if not events_cfg.get("weekly_receipt", True):
        return ""

    try:
        from src.risk.alert_state import active_events, event_age_days
        events = active_events(elder_id)
    except Exception as e:
        logger.warning(f"  └─ 预警回执章节跳过: {e}")
        return ""

    if not events:
        return ""

    lines = []
    for channel, ev in events.items():
        label = _CHANNEL_LABELS.get(channel, channel)
        age = event_age_days(ev, as_of)
        keys = ev.get("risk_keys") or []
        types = "、".join(_RISK_KEY_LABELS.get(k, k) for k in keys) or "多项指标"
        lines.append(
            f"- **{label}**：{types} 持续中"
            + (f"，第 {age} 天" if age else "")
            + f"（{ev.get('started_day')} 起）"
        )
        lines.append(
            f"  - 已通知 {ev.get('notify_count', 0)} 次，最近一次 "
            f"{ev.get('last_notified_day') or '—'}"
            + ("；已由家属确认，转静默追踪" if ev.get("acknowledged") else "")
        )

    body = "\n".join(lines)
    return (
        "## 预警回执\n\n"
        f"{body}\n\n"
        "> 以下事件仍在持续。系统按**事件**去重，同一等级不重复推送，"
        "**静默不代表已缓解**。等级上升或出现新的风险类型时会立即通知。\n\n"
    )


def _format_depression_section(
    elder_id: str, as_of: str, config: dict | None = None
) -> str:
    """抑郁评估章节（MPDD 群体基线，旁路只读）。

    ★ 刻意**另起一节**，不并进上面的"监测详情"表格。

      那张表的三行都是"跟自己比"的趋势（signed_z 相对个人基线），抑郁是"跟大众比"
      的绝对判断。放进同一张表，等于在版面上暗示两者可比、可相互印证——这与
      `_format_track_stats` 里"两轨的分绝不能平均或并列比大小"是同一条不变量，
      只是发生在展示层而不是计算层。

    读不到就静默显示"暂无最新评估"：这是旁路输出，MPDD 环境挂掉、契约文件损坏，
    都不该让整份周报生成失败。本函数只 import store（零重依赖），
    不碰 runner / clip_source，所以 MPDD 那套依赖压根不会被加载。
    """
    dep_cfg = (config or {}).get("depression", {}) or {}
    if not dep_cfg.get("enabled", False) or not dep_cfg.get("show_in_report", False):
        return ""

    try:
        from src.depression.store import summarize_for_report
        view = summarize_for_report(elder_id, as_of=as_of)
    except Exception as e:
        logger.warning(f"  └─ 抑郁评估章节跳过: {e}")
        return ""

    header = "## 情绪状态评估（研究性，未校准）\n\n"

    if not view["displayable"]:
        # ★ 绝不拿过期的旧分顶替今天。抑郁评估天然稀疏（要"有正脸 + 有连续语音"
        # 的片段，独居老人可能几周才有一次），比周报本身更容易踩这个坑——
        # 而"本周无数据时静默拿别的周顶替"正是本文件修过的缺陷。
        return f"{header}- 暂无最新评估（{view['reason']}）\n\n"

    latest = view["latest"]
    result = latest.get("result") or {}
    ev = latest.get("evidence", {})

    lines = [
        f"- 最近评估：{latest['day_key']}（{ev.get('n_clips', 0)} 段样本）",
        f"- 结果：{result.get('level', '—')}"
        + (
            f" · 估计 PHQ-9 ≈ {result['phq9_median']}"
            f"（段间波动 ±{result.get('phq9_spread', 0)}）"
            if result.get("phq9_median") is not None else ""
        ),
    ]

    # 历史时间线：大众基线对个体差异没有免疫力——天生表情少、语速慢、口音重的
    # 老人可能常年被判同一等级。排成序列看"变没变"而不是"高不高"，
    # 才能把这类固定偏置降成背景噪音。只是展示层排序，不涉及任何拟合。
    timeline = view["timeline"]
    if len(timeline) >= 2:
        trail = " → ".join(f"{t['day_key'][5:]} {t['level']}" for t in timeline[-4:])
        changed = timeline[-1]["level"] != timeline[-2]["level"]
        lines.append(f"- 历次：{trail}　← {'**较上次有变化**' if changed else '基本持平'}")

    for reason in latest.get("low_confidence_reasons", []):
        lines.append(f"- ⚠️ 证据说明：{reason}")

    warning = (latest.get("calibration") or {}).get("warning", "")
    body = "\n".join(lines)
    return (
        f"{header}{body}\n\n"
        f"> ⚠️ {warning}。\n"
        f"> 本项是基于群体模型的绝对评估，与上方\"个人基线偏离\"分属不同尺度，"
        f"不可相互印证。\n\n"
    )


_TRACK_LABELS = {"sleep": "睡眠轨", "social": "社会连接轨"}


def _format_track_stats(week_results: list[dict]) -> str:
    """
    按轨汇总本周的偏离天数与异常分。

    ★ 必须按轨，且两轨的分**绝不能平均或并列比大小**——睡眠轨 8 维、社交轨 5 维，
    权重和不同，残差尺度不可比。这与 judge.judge_risk_level"每轨各自算等级、
    取较高者"是同一条不变量。

    历史缺陷：旧实现取 `r.get('anomaly_score', 0)` —— 双轨改造后顶层根本没有
    这个键（实测 60/60 个日志都没有），于是"平均异常分/最高异常分"两行
    **恒为 0.00**，而真实分数是 0.63~1.13。周报把"一切正常"当成事实报了出去。
    """
    lines = [
        f"- 本周异常天数（任一轨）：{sum(1 for r in week_results if r.get('is_deviation', False))}"
        f"/{len(week_results)}"
    ]

    for track, label in _TRACK_LABELS.items():
        scores = [
            float(r[track]["anomaly_score"])
            for r in week_results
            if isinstance(r.get(track), dict) and r[track].get("anomaly_score") is not None
        ]
        dev_days = sum(
            1 for r in week_results
            if isinstance(r.get(track), dict) and r[track].get("is_deviation")
        )
        if not scores:
            lines.append(f"- {label}：本周无有效数据")
            continue
        lines.append(
            f"- {label}：偏离 {dev_days}/{len(scores)} 天，"
            f"平均异常分 {np.mean(scores):.2f}，最高 {np.max(scores):.2f}"
        )

    return "\n".join(lines)


# 三条趋势线各自的代表特征：(轨, 特征名)
# 用 signed_z 而非 abs 残差——abs 只能说"偏离多大"，说不出"升还是降"。
# 旧实现先取 abs 再判"上升/下降"，得到的结论是"偏离幅度在变大"，
# 却被写成"社交在上升"，方向信息在进函数之前就丢了。
_TREND_SOURCES = {
    "social": ("social", "copresence_min"),
    "sleep": ("sleep", "sleep_efficiency"),
    "activity": ("social", "activity_counts"),
}

_FLAT_TRENDS = {
    f"{name}_trend": "平稳" for name in _TREND_SOURCES
} | {
    f"{name}_vs_baseline": "无明显差异" for name in _TREND_SOURCES
}


def _compute_weekly_trends(week_results: list[dict]) -> dict:
    """
    从一周的双轨推理结果计算各维度趋势。

    取每条趋势线代表特征的 signed_z 序列（observed − predicted，负值=低于个人基线），
    产出两组互不相同的结论：

        {name}_trend        **周内**走向：前半周均值 vs 后半周均值
        {name}_vs_baseline  本周整体相对**个人基线**的高低（signed_z 均值 vs ±1.0）

    ★ 两者都**不是周环比**——本函数只拿得到 week_results 这一周的数据，
      从没加载过上一周。键名此前叫 `{name}_week_change`，而它同时是
      LLM 提示词的占位符名与 templates.generate_rule_based_report 的形参名，
      于是那句谎从键名一路传染到系统提示词（"指出本周相比上周的变化趋势"）
      和周报表格列名。改名是为了让下一个人读到键名就知道它到底是什么。
    """
    if len(week_results) < 2:
        return dict(_FLAT_TRENDS)

    series: dict[str, list[float]] = {name: [] for name in _TREND_SOURCES}

    for r in week_results:
        for name, (track, feature) in _TREND_SOURCES.items():
            track_result = r.get(track)
            if not isinstance(track_result, dict):
                continue
            if not track_result.get("signed_available", False):
                continue
            signed_z = track_result.get("signed_z") or {}
            if feature in signed_z:
                series[name].append(float(signed_z[feature]))

    def _judge_trend(values: list[float], threshold: float = 0.3) -> str:
        """**周内**前后半周均值之差：正=上升，负=下降。与上一周无关。"""
        if len(values) < 2:
            return "平稳"
        mid = len(values) // 2
        diff = float(np.mean(values[mid:]) - np.mean(values[:mid]))
        if diff > threshold:
            return "上升"
        if diff < -threshold:
            return "下降"
        return "平稳"

    def _judge_vs_baseline(values: list[float]) -> str:
        """整周相对**个人基线**的位置（不是相对上一周）"""
        if not values:
            return "无明显差异"
        avg = float(np.mean(values))
        if avg > 1.0:
            return "明显高于常态"
        if avg < -1.0:
            return "明显低于常态"
        return "无明显差异"

    result = {}
    for name, values in series.items():
        result[f"{name}_trend"] = _judge_trend(values)
        result[f"{name}_vs_baseline"] = _judge_vs_baseline(values)
    return result


def _generate_with_llm(
    elder_id: str,
    trends: dict,
    risk_result: dict,
    risk_types_str: str,
    config: dict | None = None,
) -> str:
    """
    调用LLM生成周报正文。

    Args:
        elder_id: 老人ID
        trends: 趋势数据字典
        risk_result: 风险判定结果
        risk_types_str: 风险类型字符串
        config: 全局配置；report.model / report.max_tokens 从这里取
            （此前两项写死在函数体里，settings.yaml 的 report 段无人读取）。
            调用方 generate_weekly_report 已保证它非 None——本函数此前自己也
            兜了一次底，而**只兜这一处**正是缺陷所在：它让"config 会是 None"
            成为已知事实，却把同一份 config 喂给抑郁小节时漏掉了。
            兜底已上移，这里不再重复。

    Returns:
        周报正文
    """
    deviation_days = risk_result.get("consecutive_deviation", 0)

    report_cfg = (config or {}).get("report", {}) or {}
    model = report_cfg.get("model", "claude-sonnet-5")
    max_tokens = report_cfg.get("max_tokens", 400)

    prompt = fill_prompt(
        WEEKLY_REPORT_USER_TEMPLATE,
        social_trend=trends.get("social_trend", "平稳"),
        sleep_trend=trends.get("sleep_trend", "平稳"),
        activity_trend=trends.get("activity_trend", "平稳"),
        deviation_days=deviation_days,
        risk_label=risk_result.get("risk_label", "正常"),
        risk_types=risk_types_str,
        social_vs_baseline=trends.get("social_vs_baseline", "无明显差异"),
        sleep_vs_baseline=trends.get("sleep_vs_baseline", "无明显差异"),
        activity_vs_baseline=trends.get("activity_vs_baseline", "无明显差异"),
    )

    try:
        import anthropic
        client = anthropic.Anthropic()

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=WEEKLY_REPORT_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        return response.content[0].text

    except ImportError:
        logger.warning("anthropic SDK未安装，使用规则模板生成周报")
        return generate_rule_based_report(
            elder_id=elder_id,
            week_start="",
            week_end="",
            risk_label=risk_result.get("risk_label", "正常"),
            risk_types=[risk_types_str] if risk_types_str != "无" else [],
            deviation_days=deviation_days,
            # trends 的键就是 generate_rule_based_report 的形参名
            # （social_trend / sleep_trend / social_vs_baseline ...），
            # 旧写法再补一个 _trend 后缀会拼出 social_trend_trend，直接 TypeError。
            **trends,
        )
    except Exception as e:
        logger.error(f"LLM调用失败: {e}，回退到规则模板")
        return generate_rule_based_report(
            elder_id=elder_id,
            week_start="",
            week_end="",
            risk_label=risk_result.get("risk_label", "正常"),
            risk_types=[risk_types_str] if risk_types_str != "无" else [],
            deviation_days=deviation_days,
            # trends 的键就是 generate_rule_based_report 的形参名
            # （social_trend / sleep_trend / social_vs_baseline ...），
            # 旧写法再补一个 _trend 后缀会拼出 social_trend_trend，直接 TypeError。
            **trends,
        )


def _save_report(elder_id: str, week_start: str, report: str) -> None:
    """保存周报到文件"""
    from src.utils.io import atomic_write_text
    filepath = get_log_dir("weekly_reports") / f"{elder_id}_{week_start}.md"
    atomic_write_text(filepath, report)


def _empty_report(
    elder_id: str, week_start: str, week_end: str, config: dict | None = None
) -> str:
    """生成空数据周报。

    ★ 仍要带上预警回执。

      无数据周恰恰是最需要它的时候：设备掉线导致本周没有判定，但**上一个事件
      可能还开着**。如果这份周报一字不提，家属看到的就是一份纯粹的"没数据"，
      而系统里其实还挂着一个持续中的 L3 事件——静默 + 空周报 = 两层遮蔽。
    """
    receipts = _format_alert_receipts(elder_id, week_end, config)
    return f"""# {elder_id} 心理健康周报

**周期**：{week_start} ~ {week_end}
**风险等级**：数据不足
**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 本周概况

本周暂无足够的监测数据，无法生成有效周报。请检查设备运行状态。

---

{receipts}"""
