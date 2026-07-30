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


def generate_weekly_report(
    elder_id: str,
    week_start: str | None = None,
    week_end: str | None = None,
    use_llm: bool = True,
) -> str:
    """
    生成每周心理健康周报。

    Args:
        elder_id: 老人ID
        week_start: 周起始日期 "YYYY-MM-DD"，默认计算最近7天
        week_end: 周结束日期 "YYYY-MM-DD"
        use_llm: 是否使用LLM生成（False时使用规则模板）

    Returns:
        Markdown格式的周报文本
    """
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

    # 1. 加载本周推理结果
    try:
        daily_results = load_daily_results(elder_id, n_days=7)
    except Exception:
        daily_results = []

    if not daily_results:
        report = _empty_report(elder_id, week_start_str, week_end_str)
        _save_report(elder_id, week_start_str, report)
        return report

    # 过滤到本周范围内（day_key 为新字段名，兼容旧日志的 date）
    week_results = [
        r for r in daily_results
        if week_start_str <= (r.get("day_key") or r.get("date") or "") <= week_end_str
    ]
    if not week_results:
        week_results = daily_results[-7:]

    # 2. 计算各维度趋势
    trends = _compute_weekly_trends(week_results)

    # 3. 风险判定
    risk_result = judge_risk_level(elder_id, week_results)

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
        )
    else:
        report_body = generate_rule_based_report(
            elder_id=elder_id,
            week_start=week_start_str,
            week_end=week_end_str,
            risk_label=risk_result.get("risk_label", "正常"),
            risk_types=risk_type_names,
            deviation_days=sum(1 for r in week_results if r.get("is_deviation", False)),
            **{f"{k}_trend": v for k, v in trends.items()},
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

| 维度 | 本周趋势 | 上周对比 |
| :--- | :--- | :--- |
| 社交互动 | {trends.get('social_trend', '平稳')} | 社交方面与上周相比{trends.get('social_week_change', '无明显变化')} |
| 睡眠质量 | {trends.get('sleep_trend', '平稳')} | 睡眠方面与上周相比{trends.get('sleep_week_change', '无明显变化')} |
| 日常活动 | {trends.get('activity_trend', '平稳')} | 活动方面与上周相比{trends.get('activity_week_change', '无明显变化')} |

## 统计指标

- 本周异常天数：{sum(1 for r in week_results if r.get('is_deviation', False))}/7
- 平均异常分：{np.mean([r.get('anomaly_score', 0) for r in week_results]):.2f}
- 最高异常分：{np.max([r.get('anomaly_score', 0) for r in week_results]):.2f}

## 处置建议

{risk_result.get('recommendation', '无特殊建议')}
"""

    # 7. 保存周报
    _save_report(elder_id, week_start_str, report)
    logger.info(f"  └─ 周报已保存: {elder_id}_{week_start_str}")

    return report


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
    f"{name}_week_change": "无明显变化" for name in _TREND_SOURCES
}


def _compute_weekly_trends(week_results: list[dict]) -> dict:
    """
    从一周的双轨推理结果计算各维度趋势。

    取每条趋势线代表特征的 signed_z 序列（observed − predicted，负值=低于个人基线），
    比较前后半周的均值判方向。
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
        """前后半周均值之差：正=上升，负=下降"""
        if len(values) < 2:
            return "平稳"
        mid = len(values) // 2
        diff = float(np.mean(values[mid:]) - np.mean(values[:mid]))
        if diff > threshold:
            return "上升"
        if diff < -threshold:
            return "下降"
        return "平稳"

    def _judge_week_change(values: list[float]) -> str:
        """整周相对个人基线的位置"""
        if not values:
            return "无明显变化"
        avg = float(np.mean(values))
        if avg > 1.0:
            return "有明显增加"
        if avg < -1.0:
            return "有明显减少"
        return "无明显变化"

    result = {}
    for name, values in series.items():
        result[f"{name}_trend"] = _judge_trend(values)
        result[f"{name}_week_change"] = _judge_week_change(values)
    return result


def _generate_with_llm(
    elder_id: str,
    trends: dict,
    risk_result: dict,
    risk_types_str: str,
) -> str:
    """
    调用LLM生成周报正文。

    Args:
        elder_id: 老人ID
        trends: 趋势数据字典
        risk_result: 风险判定结果
        risk_types_str: 风险类型字符串

    Returns:
        周报正文
    """
    deviation_days = risk_result.get("consecutive_deviation", 0)

    prompt = fill_prompt(
        WEEKLY_REPORT_USER_TEMPLATE,
        social_trend=trends.get("social_trend", "平稳"),
        sleep_trend=trends.get("sleep_trend", "平稳"),
        activity_trend=trends.get("activity_trend", "平稳"),
        deviation_days=deviation_days,
        risk_label=risk_result.get("risk_label", "正常"),
        risk_types=risk_types_str,
        social_week_change=trends.get("social_week_change", "无明显变化"),
        sleep_week_change=trends.get("sleep_week_change", "无明显变化"),
        activity_week_change=trends.get("activity_week_change", "无明显变化"),
    )

    try:
        import anthropic
        client = anthropic.Anthropic()

        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=400,
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
            **{f"{k}_trend": v for k, v in trends.items()},
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
            **{f"{k}_trend": v for k, v in trends.items()},
        )


def _save_report(elder_id: str, week_start: str, report: str) -> None:
    """保存周报到文件"""
    log_dir = get_log_dir("weekly_reports")
    log_dir.mkdir(parents=True, exist_ok=True)
    filepath = log_dir / f"{elder_id}_{week_start}.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)


def _empty_report(elder_id: str, week_start: str, week_end: str) -> str:
    """生成空数据周报"""
    return f"""# {elder_id} 心理健康周报

**周期**：{week_start} ~ {week_end}
**风险等级**：数据不足
**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 本周概况

本周暂无足够的监测数据，无法生成有效周报。请检查设备运行状态。

---
"""
