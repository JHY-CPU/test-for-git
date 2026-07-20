"""
每周定时任务：趋势轨

每周日凌晨 03:00 执行：
    1. 遍历所有老人
    2. 每周微调（GRU模型更新）
    3. 生成周报（LLM或规则模板）
    4. 保存周报到文件
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


def run_weekly_batch(
    elder_ids: list[str],
    config: dict | None = None,
) -> list[dict]:
    """
    批量执行每周管线。

    Args:
        elder_ids: 老人ID列表
        config: 配置

    Returns:
        每位老人的执行结果
    """
    logger.info(f"批量每周管道: {len(elder_ids)}位老人")

    results = []
    for elder_id in elder_ids:
        try:
            result = run_weekly_pipeline(elder_id, config)
            results.append(result)
        except Exception as e:
            logger.error(f"每周管道失败: elder_id={elder_id}, error={e}")
            results.append({
                "elder_id": elder_id,
                "status": "error",
                "error": str(e),
            })

    success_count = sum(1 for r in results if r.get("retrain_status") == "success")
    logger.info(f"批量完成: {success_count}/{len(elder_ids)}")

    return results


def _extract_risk_label(report_text: str) -> str:
    """从周报Markdown中提取风险等级标签"""
    import re
    match = re.search(r'\*\*风险等级\*\*[：:]\s*(.+?)\n', report_text)
    if match:
        return match.group(1).strip()
    return "未知"
