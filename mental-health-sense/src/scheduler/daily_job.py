"""
每日定时任务：常态轨

每日凌晨 02:00 执行：
    1. 遍历所有老人
    2. 聚合昨日传感器数据 → 特征向量
    3. 缺失值处理 + 数据校验
    4. 保存特征到CSV
    5. 每日推理（GRU预测 → 残差 → EWM A更新）
    6. 风险判定
    7. 如果偏离，记录预警
"""

from datetime import datetime, timedelta

from src.data_pipeline.aggregator import (
    DataInsufficientError,
    aggregate_daily_features,
)
from src.data_pipeline.imputer import impute_missing
from src.data_pipeline.validator import validate_daily_data
from src.utils.io import save_daily_features, load_features_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_daily_pipeline(
    elder_id: str,
    date_str: str | None = None,
    raw_data: dict | None = None,
    config: dict | None = None,
) -> dict:
    """
    执行单日全流程：聚合 → 填充 → 校验 → 保存 → 推理 → 判定。

    Args:
        elder_id: 老人ID
        date_str: 日期（默认昨天）
        raw_data: 原始传感器数据字典，包含:
            - sleep: 睡眠雷达数据
            - activity: PIR+IPC数据
            - social: 拾音+音箱数据
            - acoustic: SenseVoice数据
            None表示自动从data/raw/读取
        config: 全局配置

    Returns:
        {
            "elder_id": str,
            "date": str,
            "data_quality": str,
            "inference_result": dict | None,
            "risk_result": dict | None,
            "status": str,
        }
    """
    if date_str is None:
        date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"=== 每日管道启动: {elder_id} @ {date_str} ===")

    # 1. 聚合特征
    try:
        if raw_data is not None:
            feature_vec = aggregate_daily_features(
                date_str=date_str,
                sleep_data=raw_data.get("sleep"),
                activity_data=raw_data.get("activity"),
                social_data=raw_data.get("social"),
                acoustic_data=raw_data.get("acoustic"),
            )
        else:
            # 从data/raw/目录自动读取
            feature_vec = _load_raw_and_aggregate(elder_id, date_str)
    except DataInsufficientError as e:
        logger.warning(f"  └─ 数据不足: {e}")
        # 仍然尝试保存（标记为insufficient）
        return {
            "elder_id": elder_id,
            "date": date_str,
            "data_quality": "insufficient",
            "inference_result": None,
            "risk_result": None,
            "status": "data_insufficient",
        }

    # 2. 缺失值填充
    prev_vec = None
    try:
        df = load_features_csv(elder_id)
        if len(df) > 0:
            from src.baseline.scaler_utils import FULL_FEATURE_NAMES
            prev_row = df[df["data_quality"] == "valid"].tail(1)
            if len(prev_row) > 0:
                prev_vec = prev_row[FULL_FEATURE_NAMES].to_numpy(dtype=np.float64).flatten()
    except Exception:
        pass

    filled_vec, missing_count, missing_names = impute_missing(feature_vec, prev_vec)

    # 3. 数据校验
    recent_quality = _get_recent_quality(elder_id)
    quality = validate_daily_data(filled_vec, missing_count, recent_quality)

    logger.info(
        f"  └─ 数据质量: {quality}, 缺失: {missing_count}/{len(missing_names)}"
    )

    # 4. 保存特征
    save_daily_features(elder_id, date_str, filled_vec, missing_count, quality)

    # 5. 每日推理（仅valid数据执行）
    inference_result = None
    risk_result = None

    if quality == "valid":
        try:
            from src.baseline.inference import daily_inference
            inference_result = daily_inference(elder_id, date_str, config)

            if inference_result.get("status") == "success":
                # 6. 风险判定
                from src.risk.judge import quick_judge
                risk_result = quick_judge(elder_id, date_str)

                # 7. 需要时触发预警
                if risk_result.get("risk_level", 0) >= 1:
                    from src.risk.alert import trigger_alert
                    trigger_alert(
                        elder_id=elder_id,
                        risk_level=risk_result["risk_level"],
                        risk_types=risk_result.get("risk_types", []),
                    )

        except Exception as e:
            logger.error(f"  └─ 推理/判定失败: {e}")

    status = "success" if quality == "valid" else "skipped_inference"

    logger.info(f"=== 每日管道完成: {elder_id}, status={status} ===")

    return {
        "elder_id": elder_id,
        "date": date_str,
        "data_quality": quality,
        "inference_result": inference_result,
        "risk_result": risk_result,
        "status": status,
    }


def run_daily_batch(
    elder_ids: list[str],
    date_str: str | None = None,
    config: dict | None = None,
) -> list[dict]:
    """
    批量执行每日管线（按顺序遍历所有老人）。

    Args:
        elder_ids: 老人ID列表
        date_str: 日期
        config: 配置

    Returns:
        每位老人的执行结果
    """
    if date_str is None:
        date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"批量每日管道: {len(elder_ids)}位老人 @ {date_str}")

    results = []
    for elder_id in elder_ids:
        result = run_daily_pipeline(elder_id, date_str, config=config)
        results.append(result)

    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info(f"批量完成: {success_count}/{len(elder_ids)}")

    return results


def _get_recent_quality(elder_id: str, n_days: int = 5) -> list[str]:
    """获取最近N天的数据质量列表"""
    try:
        df = load_features_csv(elder_id)
        recent = df.sort_values("date", ascending=False).head(n_days)
        recent = recent.sort_values("date")
        return recent["data_quality"].tolist()
    except Exception:
        return []


def _load_raw_and_aggregate(elder_id: str, date_str: str):
    """
    从data/raw/目录自动读取原始传感器数据并聚合。
    这是一个适配层，根据实际数据格式实现。

    当前为占位实现，实际对接时替换。
    """
    import json
    from pathlib import Path

    raw_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

    def _load_json(subdir: str, filename: str) -> dict | None:
        filepath = raw_dir / subdir / elder_id / f"{date_str}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    sleep_data = _load_json("sleep", f"{date_str}.json")
    activity_data = _load_json("activity", f"{date_str}.json")
    social_data = _load_json("social", f"{date_str}.json")
    acoustic_data = _load_json("acoustic", f"{date_str}.json")

    return aggregate_daily_features(
        date_str=date_str,
        sleep_data=sleep_data,
        activity_data=activity_data,
        social_data=social_data,
        acoustic_data=acoustic_data,
    )
