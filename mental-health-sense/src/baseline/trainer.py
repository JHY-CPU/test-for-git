"""
冷启动训练与每周微调

核心函数：
    - train_initial_baseline(): Day 14冷启动，首次训练GRU + 初始化EWMA
    - weekly_retrain(): 每周微调，用最近30天数据更新模型
"""

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.baseline.ewma import CumulativeEWMABaseline
from src.baseline.gru_model import PersonalBaselineGRU
from src.baseline.scaler_utils import FULL_FEATURE_DIM, fit_scaler, save_scaler
from src.utils.io import (
    get_baseline_dir,
    get_recent_vectors,
    load_features_csv,
    save_gru_model,
    save_residual_stats,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_initial_baseline(
    elder_id: str,
    config: dict | None = None,
) -> tuple[PersonalBaselineGRU, StandardScaler, dict, CumulativeEWMABaseline]:
    """
    Day 14结束时调用，建立该老人的个人基线。

    步骤：
        1. 读取前14天特征 → (14, 10)
        2. StandardScaler.fit → scaler.pkl
        3. 构建7→1滑动窗口 → 7个训练样本
        4. 训练GRU (150 epoch)
        5. 计算训练集残差统计 → residual_stats.pkl
        6. 初始化EWMA（14天异常分=0滚动更新）
        7. 保存 gru.pth

    Args:
        elder_id: 老人ID
        config: 全局配置字典（可选，使用默认值）

    Returns:
        (model, scaler, residual_stats, ewma)

    Raises:
        ValueError: 数据不足14天时
    """
    logger.info(f"冷启动训练开始: elder_id={elder_id}")

    # 加载配置
    if config is None:
        from src.utils.io import load_config
        config = load_config()

    gru_cfg = config.get("gru", {})
    train_cfg = config.get("training", {}).get("initial", {})
    ewma_cfg = config.get("ewma", {})

    window = gru_cfg.get("window", 7)
    epochs = train_cfg.get("epochs", 150)
    lr = train_cfg.get("lr", 0.001)
    patience = train_cfg.get("patience", 20)
    hidden_dim = gru_cfg.get("hidden_dim", 16)
    num_layers = gru_cfg.get("num_layers", 1)
    dropout = gru_cfg.get("dropout", 0.2)
    ewma_alpha = ewma_cfg.get("alpha", 0.05)

    # 1. 读取特征数据
    df = load_features_csv(elder_id)
    valid_df = df[df["data_quality"] == "valid"]

    if len(valid_df) < 14:
        raise ValueError(
            f"需要至少14天有效数据，当前只有{len(valid_df)}天"
        )

    from src.baseline.scaler_utils import FULL_FEATURE_NAMES
    data = valid_df[FULL_FEATURE_NAMES].to_numpy(dtype=np.float64)[:14]  # (14, 10)

    logger.info(f"  └─ 加载 {len(data)} 天特征数据")

    # 1b. 训练数据健康门禁（防止 GRU "学坏"）
    # 建档期若混入异常态，GRU 会把异常学成正常基线。先做 MAD 离群筛查，
    # 剔除可疑异常天后再训练；若整体离群比例过高则拒绝建档，建议顺延。
    from src.baseline.data_health import detect_outlier_days, describe_outlier_days

    health_cfg = config.get("data_health", {})
    z_threshold = health_cfg.get("z_threshold", 3.5)
    min_bad_features = health_cfg.get("min_bad_features", 2)
    max_outlier_ratio = health_cfg.get("max_outlier_ratio", 0.5)

    health_report = detect_outlier_days(data, z_threshold, min_bad_features)
    if health_report["outlier_day_indices"]:
        for line in describe_outlier_days(data, health_report):
            logger.warning(f"  └─ 建档期离群天: {line}")

        if health_report["outlier_ratio"] > max_outlier_ratio:
            raise ValueError(
                f"建档期数据离群比例过高（{health_report['outlier_ratio']:.0%} > "
                f"{max_outlier_ratio:.0%}），数据整体异常，建议顺延建档而非用脏数据建立基线"
            )

        keep_mask = np.ones(len(data), dtype=bool)
        keep_mask[health_report["outlier_day_indices"]] = False
        removed = len(data) - int(keep_mask.sum())
        data = data[keep_mask]
        logger.info(
            f"  └─ 健康门禁：剔除 {removed} 个离群天，剩余 {len(data)} 天用于训练"
        )

    if len(data) < window + 1:
        raise ValueError(
            f"剔除离群天后有效数据不足（{len(data)}天 < {window + 1}天），建议顺延建档"
        )

    # 2. 拟合Scaler（在清洗后的数据上拟合，避免归一化基准被异常天带偏）
    scaler = StandardScaler()
    scaler = fit_scaler(scaler, data)
    data_norm = scaler.transform(data)  # (14, 10)

    logger.info(f"  └─ Scaler拟合完成: mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")

    # 3. 构建滑动窗口样本
    X_list, y_list = [], []
    for i in range(window, len(data_norm)):
        X_list.append(data_norm[i - window:i])  # (7, 10)
        y_list.append(data_norm[i])              # (10,)

    if len(X_list) == 0:
        raise ValueError(
            f"数据不足以构建训练样本，需要至少{window + 1}天，实际{len(data_norm)}天"
        )

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)

    logger.info(f"  └─ 训练样本: {len(X)} 个 (window={window})")

    # 4. 训练GRU模型
    model = PersonalBaselineGRU(
        feature_dim=FULL_FEATURE_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # early-stopping：冷启动样本极少（14天仅约7个窗口样本），高 epoch 易过拟合。
    # 连续 patience 轮 loss 无改善则提前停止，并回滚到最优权重。
    import copy

    model.train()
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    stopped_epoch = epochs
    for epoch in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        if cur_loss < best_loss - 1e-6:
            best_loss = cur_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 30 == 0:
            logger.debug(f"  └─ Epoch {epoch + 1}/{epochs}, Loss: {cur_loss:.6f}")

        if patience and epochs_no_improve >= patience:
            stopped_epoch = epoch + 1
            logger.info(f"  └─ early-stopping：连续{patience}轮无改善，第{stopped_epoch}轮停止")
            break

    # 回滚到最优权重（避免停止时正好在一个抖动高点）
    model.load_state_dict(best_state)

    logger.info(
        f"  └─ GRU训练完成: best_loss={best_loss:.6f}, "
        f"epochs={stopped_epoch}/{epochs}, params={model.count_parameters()}"
    )

    # 5. 计算残差统计
    model.eval()
    with torch.no_grad():
        train_pred = model(X)
        residuals = torch.abs(train_pred - y).numpy()  # (n_samples, 10)
        residual_stats = {
            "mean": residuals.mean(axis=0),  # (10,)
            "std": residuals.std(axis=0),    # (10,)
        }

    logger.info(f"  └─ 残差统计: mean={residual_stats['mean'].mean():.4f}, "
                f"std={residual_stats['std'].mean():.4f}")

    # 6. 初始化EWMA累积基线
    from src.utils.io import get_feature_weight_array
    weights = get_feature_weight_array()

    ewma = CumulativeEWMABaseline(alpha=ewma_alpha)
    for day in range(window, len(data_norm)):
        # 用模型预测该天，计算异常分
        input_seq = torch.tensor(
            data_norm[day - window:day].reshape(1, window, -1),
            dtype=torch.float32,
        )
        target = torch.tensor(data_norm[day].reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            pred_val = model(input_seq)
            residual = torch.abs(pred_val - target).numpy().flatten()
        anomaly_score = float(np.dot(residual, weights) / np.sum(weights))
        ewma.update(anomaly_score)

    logger.info(f"  └─ EWMA初始化: n={ewma.n}, mean={ewma.mean:.4f}, std={ewma.std:.4f}")

    # 7. 保存全部基线文件
    save_gru_model(model, elder_id, "gru.pth")
    save_scaler(scaler, get_baseline_dir(elder_id) / "scaler.pkl")
    save_residual_stats(residual_stats, elder_id)
    ewma.save(get_baseline_dir(elder_id) / "ewma.pkl")

    # 记录基线元信息：训练完成时的 EWMA 样本数 + 训练日期。
    # 观察期须以"训练后经过的推理天数"（ewma.n - ewma_n_at_train）判断，
    # 不能直接用 ewma.n —— 训练已用建档期样本预热 EWMA（n≈7），
    # 否则观察期会瞬间耗尽，形同虚设。
    from datetime import datetime as _dt
    from src.utils.io import save_baseline_meta
    save_baseline_meta(elder_id, {
        "ewma_n_at_train": ewma.n,
        "train_date": _dt.now().strftime("%Y-%m-%d"),
        "feature_dim": FULL_FEATURE_DIM,
    })

    logger.info(f"  └─ 基线文件保存完成: {get_baseline_dir(elder_id)}")

    return model, scaler, residual_stats, ewma


def weekly_retrain(
    elder_id: str,
    config: dict | None = None,
) -> None:
    """
    每周日凌晨执行：用最近30天数据微调GRU模型。

    步骤：
        1. 取最近30天有效特征
        2. 加载现有scaler和模型
        3. 用scaler归一化（不重新拟合！）
        4. 构建训练集
        5. 低学习率微调（防止灾难性遗忘）
        6. 指数加权合并新旧残差统计
        7. 保存模型+统计

    Args:
        elder_id: 老人ID
        config: 全局配置字典
    """
    logger.info(f"每周微调开始: elder_id={elder_id}")

    if config is None:
        from src.utils.io import load_config
        config = load_config()

    gru_cfg = config.get("gru", {})
    train_cfg = config.get("training", {}).get("finetune", {})

    window = gru_cfg.get("window", 7)
    epochs = train_cfg.get("epochs", 50)
    lr = train_cfg.get("lr", 0.0003)
    recent_days = train_cfg.get("recent_days", 30)
    merge_alpha = train_cfg.get("residual_merge_alpha", 0.3)
    exclude_deviation = train_cfg.get("exclude_deviation_days", True)

    # 1. 取最近N天有效特征（带日期，用于剔除已判定为偏离的异常天）
    from src.baseline.scaler_utils import FULL_FEATURE_NAMES
    try:
        df = load_features_csv(elder_id)
    except Exception:
        logger.warning(f"  └─ 无法获取最近数据，跳过微调")
        return

    df = df[df["data_quality"] == "valid"].sort_values("date")
    df_recent = df.tail(recent_days).copy()

    # 剔除异常天：若某天已被 daily_inference 判为 is_deviation=True，
    # 说明处于异常态，纳入微调会把异常学成正常基线，故排除。
    if exclude_deviation:
        from src.utils.io import load_daily_results
        results = load_daily_results(elder_id, n_days=max(recent_days * 2, 60))
        deviation_dates = {
            r["date"] for r in results if r.get("is_deviation", False)
        }
        if deviation_dates:
            before = len(df_recent)
            df_recent = df_recent[~df_recent["date"].isin(deviation_dates)]
            excluded = before - len(df_recent)
            if excluded:
                logger.info(f"  └─ 微调剔除 {excluded} 个偏离天（防基线被异常期污染）")

    recent = df_recent[FULL_FEATURE_NAMES].to_numpy(dtype=np.float64)

    if len(recent) < 14:
        logger.warning(f"  └─ 有效数据不足14天（{len(recent)}天），跳过微调")
        return

    logger.info(f"  └─ 加载 {len(recent)} 天特征数据")

    # 2. 加载现有scaler（不重新拟合）
    from src.baseline.scaler_utils import load_scaler, transform_data
    from src.utils.io import load_gru_model, load_residual_stats

    scaler = load_scaler(get_baseline_dir(elder_id) / "scaler.pkl")
    data_norm = scaler.transform(recent)  # (n, 10)

    # 3. 加载现有模型
    model = load_gru_model(PersonalBaselineGRU, elder_id, "gru.pth")

    # 4. 构建训练集
    X_list, y_list = [], []
    for i in range(window, len(data_norm)):
        X_list.append(data_norm[i - window:i])
        y_list.append(data_norm[i])

    if len(X_list) == 0:
        logger.warning(f"  └─ 无法构建训练样本")
        return

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)

    # 5. 低学习率微调
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    best_loss = float("inf")
    for epoch in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

    logger.info(f"  └─ 微调完成: best_loss={best_loss:.6f}")

    # 6. 更新残差统计（指数加权合并）
    model.eval()
    with torch.no_grad():
        preds = model(X)
        new_residuals = torch.abs(preds - y).numpy()
        new_mean = new_residuals.mean(axis=0)
        new_std = new_residuals.std(axis=0)

    try:
        old_stats = load_residual_stats(elder_id)
        merged_mean = (1 - merge_alpha) * old_stats["mean"] + merge_alpha * new_mean
        merged_std = (1 - merge_alpha) * old_stats["std"] + merge_alpha * new_std
        logger.info(f"  └─ 残差统计合并: alpha={merge_alpha}")
    except Exception:
        merged_mean = new_mean
        merged_std = new_std
        logger.info(f"  └─ 新建残差统计")

    merged_stats = {"mean": merged_mean, "std": merged_std}

    # 7. 保存（覆盖前先备份上一版模型，便于微调把模型搞坏时回滚）
    import shutil
    baseline_dir = get_baseline_dir(elder_id)
    cur_model = baseline_dir / "gru.pth"
    if cur_model.exists():
        shutil.copy2(cur_model, baseline_dir / "gru.prev.pth")
        logger.info(f"  └─ 已备份上一版模型: gru.prev.pth")

    save_gru_model(model, elder_id, "gru.pth")
    save_residual_stats(merged_stats, elder_id)

    logger.info(f"  └─ 微调文件保存完成")
