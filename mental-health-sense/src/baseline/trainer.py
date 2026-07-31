"""
冷启动训练与每周微调（按轨独立）

核心函数：
    - train_initial_baseline(): 某轨建档期冷启动，首次训练GRU + 初始化EWMA
    - train_all_tracks():       两轨各训一次，一轨失败不影响另一轨
    - weekly_retrain():         某轨每周微调，用最近30天数据更新模型

★ 本次重构最重要的一条改动：残差统计改在**留出段**上估计。

旧实现用训练集残差估 residual_stats（`torch.abs(train_pred - y)`），这是一条
闭合的误报链：

    模型对训练集的残差被压得很小
      → residual_stats["std"] 趋零
      → rules.py 里 signed_z = residual / std 的分母趋零
      → 任何微小偏差被放大成巨大 z 分
      → 建档期一过就疯狂误报

留出段是模型训练时没见过的日子，它的残差尺度才反映真实的预测误差。
建档 35 天切成：7 天喂窗口 + 21 天训练 + 7 天留出。
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.baseline.ewma import TrackEWMAPools
from src.baseline.gru_model import PersonalBaselineGRU
from src.baseline.scaler_utils import (
    TRACKS,
    get_feature_dim,
    get_feature_names,
    fit_scaler,
    save_scaler,
    validate_track,
)
from src.utils.io import (
    DAY_KEY_COL,
    get_baseline_dir,
    get_feature_weight_array,
    get_model_filename,
    get_scaler_path,
    load_features_csv,
    load_residual_stats,
    save_gru_model,
    save_residual_stats,
    update_baseline_meta,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _is_weekend(day_key: str) -> bool:
    """day_key 是否为周末（周六=5 / 周日=6）"""
    from datetime import datetime
    return datetime.strptime(day_key, "%Y-%m-%d").weekday() >= 5


def _track_gru_config(config: dict, track: str) -> dict:
    """
    取某轨的 GRU 超参。window / num_layers / dropout 两轨共用，
    feature_dim / hidden_dim 按轨取。
    """
    gru_cfg = config.get("gru", {})
    track_cfg = gru_cfg.get(track, {})

    feature_dim = track_cfg.get("feature_dim", get_feature_dim(track))
    # 配置里的 feature_dim 与特征表不一致时立即报错：静默用错维度会让
    # 整条残差链失去意义，而且极难排查。
    if feature_dim != get_feature_dim(track):
        raise ValueError(
            f"config gru.{track}.feature_dim={feature_dim} 与特征表 "
            f"{get_feature_dim(track)} 不一致；改特征表后必须同步配置并重建档"
        )

    return {
        "feature_dim": feature_dim,
        "hidden_dim": track_cfg.get("hidden_dim", 8),
        "num_layers": gru_cfg.get("num_layers", 1),
        "dropout": gru_cfg.get("dropout", 0.2),
        "window": gru_cfg.get("window", 7),
    }


def _train_loop(
    model: PersonalBaselineGRU,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    patience: int | None = None,
) -> float:
    """全 batch 训练循环，带 early-stopping + 回滚最优权重。

    冷启动与每周微调共用同一套过拟合抑制策略：连续 patience 轮 loss 无改善则
    提前停止，并把模型权重回滚到最优点（避免停在抖动高点）。patience=None/0 时
    不启用 early-stopping，跑满 epochs。

    Returns:
        best_loss（最优训练损失）
    """
    import copy

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    for epoch in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, y)
        cur_loss = loss.item()

        # 先按"产生 cur_loss 的这组权重"记录最优点，再做梯度更新——保证 best_state
        # 与 best_loss 严格对应（若在 step 之后保存，存下的是"更新一步之后"的权重，
        # 与刚记录的 best_loss 差一个梯度步，回滚点会偏移）。
        if cur_loss < best_loss - 1e-6:
            best_loss = cur_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if patience and epochs_no_improve >= patience:
            logger.info(f"  └─ early-stopping：连续{patience}轮无改善，第{epoch + 1}轮停止")
            break

    # 回滚到最优权重
    model.load_state_dict(best_state)
    return best_loss


def _build_windows(
    data_norm: np.ndarray,
    window: int,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    构建 [start, end) 区间内的滑动窗口样本。

    目标日 i 的输入取 data_norm[i-window:i]，即使输入落在训练区间内也没问题——
    一步提前预测本来就是"用之前的日子预测今天"，输入重叠不构成信息泄漏，
    真正要防的是"目标日参与过训练"。

    Returns:
        (X, y, target_indices)
    """
    X_list, y_list, idx_list = [], [], []
    for i in range(max(start, window), end):
        X_list.append(data_norm[i - window:i])
        y_list.append(data_norm[i])
        idx_list.append(i)

    if not X_list:
        return (
            torch.empty(0, window, data_norm.shape[1]),
            torch.empty(0, data_norm.shape[1]),
            [],
        )

    return (
        torch.tensor(np.array(X_list), dtype=torch.float32),
        torch.tensor(np.array(y_list), dtype=torch.float32),
        idx_list,
    )


def _last_contiguous_span(df, day_key_col: str = DAY_KEY_COL):
    """取 df 里**最后一段日期连续**的行（按自然日判断，非行号）。

    GRU 的输入窗口 data_norm[i-window:i] 隐含"这 window 行是相邻的 window 天"。
    一旦中间剔掉了 degraded / 偏离天，这个假设就不成立，模型学到的是错误的
    时间关系。取最后一段连续日虽然会损失样本，但保证喂进去的确实是连续的天。

    取"最后一段"而不是"最长一段"：微调的目的是跟上老人**最近**的状态，
    用三个月前的一段长数据去微调没有意义。
    """
    from datetime import datetime, timedelta

    if len(df) == 0:
        return df

    keys = df[day_key_col].tolist()
    start_idx = 0
    for i in range(len(keys) - 1, 0, -1):
        try:
            prev = datetime.strptime(str(keys[i - 1]), "%Y-%m-%d")
            cur = datetime.strptime(str(keys[i]), "%Y-%m-%d")
        except ValueError:
            start_idx = i          # 日期解析不了就从这里断开，保守处理
            break
        if cur - prev != timedelta(days=1):
            start_idx = i
            break

    if start_idx:
        logger.info(
            f"  └─ 日期不连续，取最后一段连续 {len(keys) - start_idx} 天"
            f"（丢弃前 {start_idx} 天，避免把非相邻日当成相邻日喂进窗口）"
        )
    return df.iloc[start_idx:]


def _residual_stats_from(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    双残差契约：signed 与 abs 两套统计各自独立维护。

        signed = observed − predicted   ← 判方向（值下降则为负，与直觉一致）
        abs    = |signed|               ← 打幅度分

    为什么两套都要存：anomaly_score 只关心"偏离多大"，用 abs；
    风险规则要判"睡眠效率是不是**下降**了"，必须用 signed。
    只留 abs 的话方向在进 rules.py 之前就丢了，所有 down 方向规则都失效；
    而用 abs 的 std 去标准化 signed 残差则是量纲错配（|r| 的分布与 r 不同）。
    """
    signed = (target - pred).numpy()
    abs_res = np.abs(signed)
    return {
        "signed": {
            "mean": signed.mean(axis=0),
            "std": signed.std(axis=0),
        },
        "abs": {
            "mean": abs_res.mean(axis=0),
            "std": abs_res.std(axis=0),
        },
    }


def train_initial_baseline(
    elder_id: str,
    track: str,
    config: dict | None = None,
) -> tuple[PersonalBaselineGRU, StandardScaler, dict, TrackEWMAPools]:
    """
    某轨建档期结束时调用，建立该轨的个人基线。

    步骤（以默认 35 天、window=7、holdout=7 为例）：
        1. 读取前 build_days 天该轨特征 → (35, dim)
        2. 健康门禁：MAD 离群筛查，剔除可疑异常天
        3. StandardScaler.fit（在清洗后数据上）→ scaler_{track}.pkl
        4. 切分：训练目标日 [7, 28)，留出目标日 [28, 35)
        5. 训练GRU（只用训练段）
        6. ★ 在**留出段**上估残差统计（signed + abs）→ residual_stats_{track}.pkl
        7. 初始化EWMA（社交轨按 is_weekend 分池）
        8. 保存 gru_{track}.pth

    Returns:
        (model, scaler, residual_stats, ewma_pools)

    Raises:
        ValueError: 数据不足或离群比例过高
    """
    track = validate_track(track)
    logger.info(f"[{track}] 冷启动训练开始: elder_id={elder_id}")

    if config is None:
        from src.utils.io import load_config
        config = load_config()

    gru_cfg = _track_gru_config(config, track)
    train_cfg = config.get("training", {}).get("initial", {})
    ewma_cfg = config.get("ewma", {})

    window = gru_cfg["window"]
    build_days = train_cfg.get("build_days", 35)
    holdout_days = train_cfg.get("holdout_days", 7)
    epochs = train_cfg.get("epochs", 150)
    lr = train_cfg.get("lr", 0.001)
    patience = train_cfg.get("patience", 20)
    ewma_alpha = ewma_cfg.get("alpha", 0.05)

    names = get_feature_names(track)

    # 1. 读取该轨特征数据
    # ★ 必须同时剔除含缺测维（NaN）的行，不能只看 data_quality。
    #   缺 1 维时质量仍判 valid（DEGRADED_THRESHOLD=2），而缺测维现在保持 NaN
    #   （imputer 不再填 0）。把 NaN 喂进 StandardScaler.fit 会让 mean_/scale_
    #   变成 NaN，整条基线报废；而旧的"填 0"写法虽然不炸，却把一个 −15σ 的点
    #   算进了归一化基准——两种都不能要。scaler 一旦建档就永久冻结，
    #   这里的每一行都会影响之后所有 signed_z 的分母。
    df = load_features_csv(elder_id, track)
    valid_df = df[df["data_quality"] == "valid"].sort_values(DAY_KEY_COL)
    complete_mask = valid_df[names].notna().all(axis=1)
    dropped = int((~complete_mask).sum())
    if dropped:
        logger.info(f"  └─ [{track}] 建档剔除 {dropped} 个含缺测维的日子")
    valid_df = valid_df[complete_mask]

    if len(valid_df) < build_days:
        raise ValueError(
            f"[{track}] 需要至少{build_days}天有效数据，当前只有{len(valid_df)}天"
        )

    valid_df = valid_df.iloc[:build_days]
    data = valid_df[names].to_numpy(dtype=np.float64)
    day_keys = valid_df[DAY_KEY_COL].tolist()

    logger.info(f"  └─ 加载 {len(data)} 天特征数据（{len(names)} 维）")

    # 2. 训练数据健康门禁（防止 GRU "学坏"）
    # 建档期若混入异常态，GRU 会把异常学成正常基线。先做 MAD 离群筛查，
    # 剔除可疑异常天后再训练；若整体离群比例过高则拒绝建档，建议顺延。
    from src.baseline.data_health import detect_outlier_days, describe_outlier_days

    health_cfg = config.get("data_health", {})
    health_report = detect_outlier_days(
        data,
        health_cfg.get("z_threshold", 3.5),
        health_cfg.get("min_bad_features", 2),
    )
    max_outlier_ratio = health_cfg.get("max_outlier_ratio", 0.5)

    if health_report["outlier_day_indices"]:
        for line in describe_outlier_days(data, health_report, track):
            logger.warning(f"  └─ 建档期离群天: {line}")

        if health_report["outlier_ratio"] > max_outlier_ratio:
            raise ValueError(
                f"[{track}] 建档期数据离群比例过高（{health_report['outlier_ratio']:.0%} > "
                f"{max_outlier_ratio:.0%}），数据整体异常，建议顺延建档而非用脏数据建立基线"
            )

        keep_mask = np.ones(len(data), dtype=bool)
        keep_mask[health_report["outlier_day_indices"]] = False
        removed = len(data) - int(keep_mask.sum())
        data = data[keep_mask]
        day_keys = [k for k, keep in zip(day_keys, keep_mask) if keep]
        logger.info(f"  └─ 健康门禁：剔除 {removed} 个离群天，剩余 {len(data)} 天用于训练")

    # 训练段至少要有 1 个样本，留出段至少 2 个（否则 std 无从估计）
    min_required = window + 1 + 2
    if len(data) < min_required:
        raise ValueError(
            f"[{track}] 剔除离群天后有效数据不足（{len(data)}天 < {min_required}天），建议顺延建档"
        )

    # 3. 拟合Scaler（在清洗后的数据上拟合，避免归一化基准被异常天带偏）
    scaler = fit_scaler(StandardScaler(), data, track)
    data_norm = scaler.transform(data)

    logger.info(
        f"  └─ Scaler拟合完成: mean[0]={scaler.mean_[0]:.4f}, std[0]={scaler.scale_[0]:.4f}"
    )

    # 4. 切分训练段与留出段（按时间顺序，不做随机划分——时序数据随机划分
    #    会让"未来"泄漏进训练集）
    n = len(data_norm)
    actual_holdout = min(holdout_days, n - window - 1)
    if actual_holdout < 2:
        raise ValueError(
            f"[{track}] 留出段不足 2 天（可用 {actual_holdout}），无法估计残差统计。"
            f"请延长建档期或减少 holdout_days"
        )
    split = n - actual_holdout

    X_train, y_train, _ = _build_windows(data_norm, window, window, split)
    X_hold, y_hold, hold_idx = _build_windows(data_norm, window, split, n)

    if len(X_train) == 0:
        raise ValueError(
            f"[{track}] 训练段无法构建样本，需要至少{window + 1}天，实际{split}天"
        )

    logger.info(
        f"  └─ 切分: 训练样本 {len(X_train)} 个 / 留出样本 {len(X_hold)} 个 (window={window})"
    )

    # 5. 训练GRU模型（只用训练段）
    model = PersonalBaselineGRU(
        feature_dim=gru_cfg["feature_dim"],
        hidden_dim=gru_cfg["hidden_dim"],
        num_layers=gru_cfg["num_layers"],
        dropout=gru_cfg["dropout"],
    )
    best_loss = _train_loop(model, X_train, y_train, epochs=epochs, lr=lr, patience=patience)

    logger.info(
        f"  └─ GRU训练完成: best_loss={best_loss:.6f}, params={model.count_parameters()}"
    )

    # 6. ★ 在留出段上估残差统计（不用训练集！见模块 docstring 的误报链说明）
    model.eval()
    with torch.no_grad():
        hold_pred = model(X_hold)
        residual_stats = _residual_stats_from(hold_pred, y_hold)
        train_pred = model(X_train)
        train_abs = np.abs((y_train - train_pred).numpy())

    logger.info(
        f"  └─ 残差统计（留出段）: signed_std={residual_stats['signed']['std'].mean():.4f}, "
        f"abs_mean={residual_stats['abs']['mean'].mean():.4f}"
    )
    logger.info(
        f"  └─ 对照：训练段 abs_mean={train_abs.mean():.4f}"
        f"（若远小于留出段，说明存在过拟合，但阈值已用留出段估计，不受影响）"
    )

    # 7. 初始化EWMA累积基线（社交轨按 is_weekend 分池）
    weights = get_feature_weight_array(track)
    ewma = TrackEWMAPools(
        track=track, alpha=ewma_alpha,
        max_freeze_days=ewma_cfg.get("max_freeze_days", 14),
    )

    all_X, all_y, all_idx = _build_windows(data_norm, window, window, n)
    with torch.no_grad():
        all_pred = model(all_X)
    all_abs = np.abs((all_y - all_pred).numpy())

    for row, target_idx in enumerate(all_idx):
        anomaly_score = float(np.dot(all_abs[row], weights) / np.sum(weights))
        # 传 day_key：建档期样本按日期严格递增，去重不会误跳；同时把 last_day_key
        # 推进到建档末日，避免建档后紧接着重跑这几天又喂一遍。
        ewma.update(
            anomaly_score,
            is_weekend=_is_weekend(day_keys[target_idx]),
            day_key=day_keys[target_idx],
        )

    logger.info(f"  └─ EWMA初始化: {ewma}")

    # 8. 保存全部基线文件
    save_gru_model(model, elder_id, get_model_filename(track))
    save_scaler(scaler, get_scaler_path(elder_id, track))
    save_residual_stats(residual_stats, elder_id, track)
    ewma.save(get_baseline_dir(elder_id))

    # 记录基线元信息：训练完成时的 EWMA 样本数 + 训练日期。
    # 观察期须以"训练后经过的推理天数"（total - ewma_n_at_train）判断，
    # 不能直接用样本数 —— 训练已用建档期样本预热 EWMA，否则观察期形同虚设。
    from datetime import datetime as _dt
    update_baseline_meta(elder_id, track, {
        "ewma_n_at_train": ewma.total_samples(),
        "train_date": _dt.now().strftime("%Y-%m-%d"),
        "feature_dim": gru_cfg["feature_dim"],
        "hidden_dim": gru_cfg["hidden_dim"],
        "feature_names": names,
        "build_days": build_days,
        "holdout_days": actual_holdout,
        "threshold_source": "holdout",
        "last_day_key": day_keys[-1],
    })

    logger.info(f"  └─ [{track}] 基线文件保存完成: {get_baseline_dir(elder_id)}")

    return model, scaler, residual_stats, ewma


def train_all_tracks(elder_id: str, config: dict | None = None) -> dict:
    """
    两轨各训一次。一轨失败不影响另一轨——这正是双轨的故障隔离价值：
    小贝壳数据不够时社交轨照常建档。

    Returns:
        {"sleep": "success" | "failed: ...", "social": ...}
    """
    results = {}
    for track in TRACKS:
        try:
            train_initial_baseline(elder_id, track, config)
            results[track] = "success"
        except Exception as e:
            logger.warning(f"[{track}] 建档失败: {e}")
            results[track] = f"failed: {e}"
    return results


def weekly_retrain(
    elder_id: str,
    track: str,
    config: dict | None = None,
) -> None:
    """
    每周对某轨微调：用最近30天数据低学习率更新模型。

    步骤：
        1. 取最近30天该轨有效特征（剔除已判偏离的天）
        2. 加载现有scaler和模型（scaler 不重新拟合！）
        3. 切训练段 / 留出段
        4. 低学习率微调（防止灾难性遗忘）
        5. ★ 在留出段上估新残差统计，再与旧统计指数加权合并
        6. 保存模型+统计（覆盖前备份上一版）
    """
    track = validate_track(track)
    logger.info(f"[{track}] 每周微调开始: elder_id={elder_id}")

    if config is None:
        from src.utils.io import load_config
        config = load_config()

    gru_cfg = _track_gru_config(config, track)
    train_cfg = config.get("training", {}).get("finetune", {})

    window = gru_cfg["window"]
    epochs = train_cfg.get("epochs", 50)
    lr = train_cfg.get("lr", 0.0003)
    recent_days = train_cfg.get("recent_days", 30)
    merge_alpha = train_cfg.get("residual_merge_alpha", 0.3)
    exclude_deviation = train_cfg.get("exclude_deviation_days", True)
    patience = train_cfg.get("patience", 10)
    holdout_days = config.get("training", {}).get("initial", {}).get("holdout_days", 7)

    names = get_feature_names(track)

    # 1. 取最近N天有效特征
    try:
        df = load_features_csv(elder_id, track)
    except Exception:
        logger.warning(f"  └─ [{track}] 无法获取最近数据，跳过微调")
        return

    # 同建档：含缺测维（NaN）的行不参与微调，理由见 train_initial_baseline
    df = df[df["data_quality"] == "valid"].sort_values(DAY_KEY_COL)
    df = df[df[names].notna().all(axis=1)]
    df_recent = df.tail(recent_days).copy()

    # 剔除异常天：若某天已被 daily_inference 判为该轨 is_deviation=True，
    # 说明处于异常态，纳入微调会把异常学成正常基线，故排除。
    if exclude_deviation:
        from src.utils.io import load_daily_results
        results = load_daily_results(elder_id, n_days=max(recent_days * 2, 60))
        deviation_dates = set()
        for r in results:
            track_result = r.get(track)
            if isinstance(track_result, dict) and track_result.get("is_deviation"):
                day = r.get("day_key") or r.get("date")
                if day:
                    deviation_dates.add(day)
        if deviation_dates:
            before = len(df_recent)
            df_recent = df_recent[~df_recent[DAY_KEY_COL].isin(deviation_dates)]
            excluded = before - len(df_recent)
            if excluded:
                logger.info(f"  └─ 微调剔除 {excluded} 个偏离天（防基线被异常期污染）")

    # ★ 只保留**最后一段连续自然日**。
    #
    # 上面剔完 degraded 与偏离天之后，df_recent 的行与行之间不再是相邻的日子，
    # 而 _build_windows 仍然把 data_norm[i-window:i] 当成"前 window 天"。
    # 于是模型会用 8/10/11/15 号去预测 16 号，学到的"昨天→今天"关系是假的；
    # 剔除得越多（也就是异常越严重时）窗口越失真——正好在最需要基线准确的时候。
    #
    # 这与本轮统一的日历语义一致：持续性统计已经按自然日回溯（continuity.py），
    # 模型输入没有理由继续按行号。
    df_recent = _last_contiguous_span(df_recent)
    recent = df_recent[names].to_numpy(dtype=np.float64)

    if len(recent) < window + 3:
        logger.warning(
            f"  └─ [{track}] 有效数据不足{window + 3}天（{len(recent)}天），跳过微调"
        )
        return

    logger.info(f"  └─ 加载 {len(recent)} 天特征数据")

    # 2. 加载现有scaler（不重新拟合）
    from src.baseline.scaler_utils import load_scaler
    from src.utils.io import load_gru_model

    scaler = load_scaler(get_scaler_path(elder_id, track))
    data_norm = scaler.transform(recent)

    # 3. 加载现有模型（维度必须与配置一致，否则 load_state_dict 报错）
    model = load_gru_model(
        PersonalBaselineGRU,
        elder_id,
        get_model_filename(track),
        feature_dim=gru_cfg["feature_dim"],
        hidden_dim=gru_cfg["hidden_dim"],
        num_layers=gru_cfg["num_layers"],
        dropout=gru_cfg["dropout"],
    )

    # 4. 切分并微调
    n = len(data_norm)
    actual_holdout = min(holdout_days, max(n - window - 1, 0))
    split = n - actual_holdout

    X_train, y_train, _ = _build_windows(data_norm, window, window, split)
    if len(X_train) == 0:
        logger.warning(f"  └─ [{track}] 无法构建训练样本")
        return

    best_loss = _train_loop(model, X_train, y_train, epochs=epochs, lr=lr, patience=patience)
    logger.info(f"  └─ 微调完成: best_loss={best_loss:.6f}")

    # 5. 在留出段估新统计；留出段太短则退回全段（并记录，因为此时阈值偏乐观）
    X_eval, y_eval, _ = _build_windows(data_norm, window, split, n)
    if len(X_eval) < 2:
        # ★ 不再退回全段。用训练集残差估阈值会让 std 趋零 → 阈值分母趋零 →
        #   建档期一过疯狂误报，这是本仓明令禁止的失效链（"残差统计只能在
        #   留出段估计"）。原来的退路在 n ≥ window+3 时不可达，但它一旦可达
        #   就会**静默违反**这条不变量——只留一行 warning，没人会注意到。
        #   宁可显式失败：微调跳过一次没有代价，阈值被悄悄弄坏有。
        raise ValueError(
            f"[{track}] 留出段只有 {len(X_eval)} 个样本（需 ≥2），无法估残差统计。"
            f"残差统计只能在留出段估计，不得退回训练集。"
        )

    model.eval()
    with torch.no_grad():
        new_stats = _residual_stats_from(model(X_eval), y_eval)

    try:
        old_stats = load_residual_stats(elder_id, track)
        merged = {}
        for kind in ("signed", "abs"):
            merged[kind] = {
                stat: (1 - merge_alpha) * old_stats[kind][stat] + merge_alpha * new_stats[kind][stat]
                for stat in ("mean", "std")
            }
        logger.info(f"  └─ 残差统计合并: alpha={merge_alpha}")
    except (FileNotFoundError, KeyError, TypeError):
        # 旧格式（只有 mean/std 顶层键）或文件缺失：直接用新统计，不做合并。
        # 不静默 catch 全部异常——真正的编程错误应该冒出来。
        merged = new_stats
        logger.info(f"  └─ 新建残差统计（旧统计缺失或格式不兼容）")

    # 6. 保存（覆盖前先备份上一版模型，便于微调把模型搞坏时回滚）
    import shutil
    baseline_dir = get_baseline_dir(elder_id)
    cur_model = baseline_dir / get_model_filename(track)
    if cur_model.exists():
        shutil.copy2(cur_model, baseline_dir / f"gru_{track}.prev.pth")
        logger.info(f"  └─ 已备份上一版模型: gru_{track}.prev.pth")

    save_gru_model(model, elder_id, get_model_filename(track))
    save_residual_stats(merged, elder_id, track)

    logger.info(f"  └─ [{track}] 微调文件保存完成")


def retrain_all_tracks(elder_id: str, config: dict | None = None) -> dict:
    """两轨各微调一次，一轨失败不影响另一轨"""
    results = {}
    for track in TRACKS:
        try:
            weekly_retrain(elder_id, track, config)
            results[track] = "success"
        except Exception as e:
            logger.warning(f"[{track}] 微调失败: {e}")
            results[track] = f"failed: {e}"
    return results
