"""
每周定时任务：趋势轨（单人系统）

每周日凌晨 03:00 对被监测的老人执行：
    1. 每周微调（GRU模型更新）
    2. 生成周报（LLM或规则模板）
    3. 保存周报到文件
"""

from datetime import datetime, timedelta

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_weekly_pipeline(
    elder_id: str,
    config: dict | None = None,
) -> dict:
    """
    执行每周流程：微调 + 周报。

    Args:
        elder_id: 老人ID
        config: 全局配置

    Returns:
        {
            "elder_id": str,
            "week_start": str,
            "week_end": str,
            "retrain_status": str,
            "report_path": str | None,
            "risk_label": str,
        }
    """
    today = datetime.now()
    week_end = today.strftime("%Y-%m-%d")
    week_start = (today - timedelta(days=6)).strftime("%Y-%m-%d")

    logger.info(f"=== 每周管道启动: {elder_id} ({week_start} ~ {week_end}) ===")

    # 1. 微调模型
    retrain_status = "skipped"
    try:
        from src.baseline.trainer import weekly_retrain
        weekly_retrain(elder_id, config)
        retrain_status = "success"
        logger.info(f"  └─ 模型微调完成")
    except Exception as e:
        logger.warning(f"  └─ 模型微调跳过: {e}")
        retrain_status = f"failed: {e}"

    # 2. 生成周报
    report_path = None
    risk_label = "未知"
    try:
        from src.report.weekly_report import generate_weekly_report
        report_text = generate_weekly_report(
            elder_id=elder_id,
            week_start=week_start,
            week_end=week_end,
            use_llm=True,
        )
        risk_label = _extract_risk_label(report_text)
        report_path = str(
            __import__("src.utils.io", fromlist=["get_log_dir"])
            .get_log_dir("weekly_reports") / f"{elder_id}_{week_start}.md"
        )
        logger.info(f"  └─ 周报已生成: {report_path}")
    except Exception as e:
        logger.error(f"  └─ 周报生成失败: {e}")

    logger.info(f"=== 每周管道完成: {elder_id} ===")

    return {
        "elder_id": elder_id,
        "week_start": week_start,
        "week_end": week_end,
        "retrain_status": retrain_status,
        "report_path": report_path,
        "risk_label": risk_label,
    }


def _extract_risk_label(report_text: str) -> str:
    """从周报Markdown中提取风险等级标签"""
    import re
    match = re.search(r'\*\*风险等级\*\*[：:]\s*(.+?)\n', report_text)
    if match:
        return match.group(1).strip()
    return "未知"
