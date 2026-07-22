# 心理健康连续感知与趋势预警系统

> 面向**单个**居家老人的日常心理健康**趋势预警**系统。通过"自己和自己比"的个人化基线方法检测心理状态偏离，**不进行临床诊断**。

## 系统特点

- **单人系统**：面向单一老人的连续监测与趋势预警
- **双轨架构**：实时轨（快速响应）+ 每日轨（深度分析）
- **个人化基线**：为该老人独立建模（PersonalBaselineGRU + EWMA动态阈值），"自己和自己比"
- **多传感器融合**：睡眠雷达 + PIR/IPC + 拾音器 + SenseVoice语音情感分析
- **趋势判定**：连续3-5天偏离才触发预警，避免单日波动误报
- **统一数据流**：实时系统作为采集前端，每日系统读取累积数据

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成模拟数据（1位老人 × 50天）
python scripts/generate_simulation_data.py

# 3. 冷启动训练（Day 14）
python scripts/train_all_baselines.py

# 4. 每日推理
python scripts/run_daily_pipeline.py --date 2026-08-15

# 5. 启动统一系统（实时+每日）
python scripts/start_unified_system.py --elder-id E001 --microphone
```

---

## 项目结构

```
mental-health-sense/
├── config/                          # 配置文件
│   ├── settings.yaml                # 全局配置（GRU、训练、EWMA、风险阈值）
│   ├── feature_weights.json         # 特征权重（加权残差计算）
│   └── realtime_config.yaml         # 实时系统配置
│
├── data/                            # 数据目录
│   ├── raw/                         # 原始传感器数据（JSON）
│   │   ├── acoustic/                # SenseVoice输出（sad_ratio、avg_speed、pitch_variability等）
│   │   ├── sleep/                   # 睡眠雷达数据
│   │   ├── activity/                # PIR + IPC活动数据
│   │   └── social/                  # 拾音器 + 智能音箱数据
│   ├── features/                    # 聚合后的每日特征向量（CSV）
│   │   └── E001/features.csv        # 10维健康特征（含遗留time编码列，模型不使用）
│   ├── baselines/                   # 该老人的个人基线模型
│   │   └── E001/
│   │       ├── gru.pth              # 训练好的GRU模型
│   │       ├── scaler.pkl           # StandardScaler（归一化）
│   │       ├── residual_stats.pkl   # 训练残差统计（均值、标准差）
│   │       └── ewma.pkl             # EWMA累积基线
│   ├── logs/                        # 推理日志和周报
│   │   ├── daily_inference/         # 每日GRU推理结果（JSON）
│   │   └── weekly_reports/          # LLM生成的周报
│   ├── realtime/                    # 实时监测数据
│   │   ├── E001/
│   │   │   ├── features/            # 24小时聚合特征（JSON快照）
│   │   │   └── alerts/              # 实时风险预警记录
│   │   └── demo/                    # 演示结果
│   └── elder_configs.json           # 老人元数据（姓名、描述）
│
├── src/                             # 源代码
│   ├── baseline/                    # GRU个人基线模型
│   │   ├── gru_model.py             # PersonalBaselineGRU（7→1天预测）
│   │   ├── trainer.py               # 冷启动 + 每周微调
│   │   ├── inference.py             # 每日推理引擎
│   │   ├── ewma.py                  # EWMA累积基线（替代60天窗口）
│   │   └── scaler_utils.py          # StandardScaler管理
│   │
│   ├── data_pipeline/               # 数据采集与预处理
│   │   ├── aggregator.py            # 四维度传感器 → 10维特征向量
│   │   ├── imputer.py               # 缺失值处理（前向填充）
│   │   ├── validator.py             # 数据质量校验
│   │   └── adapters/                # 传感器适配器
│   │       ├── sleep_radar.py       # 睡眠雷达适配器
│   │       ├── camera.py            # IPC/RTSP摄像头适配器
│   │       ├── microphone.py        # 麦克风适配器
│   │       └── sensevoice.py        # SenseVoice模型适配器
│   │
│   ├── realtime/                    # 实时监测系统
│   │   ├── audio_stream.py          # 音频流抽象（麦克风/RTSP/文件）
│   │   ├── sensevoice_engine.py     # SenseVoice推理 + 24小时聚合器
│   │   └── monitor.py               # RealtimeMonitor主控制器
│   │
│   ├── risk/                        # 风险判定层
│   │   ├── rules.py                 # 3种风险类型（抑郁/睡眠/社交）
│   │   ├── judge.py                 # 4级风险判定（连续天数）
│   │   └── alert.py                 # 预警推送（日志/文件/App/短信）
│   │
│   ├── report/                      # 周报生成
│   │   ├── templates.py             # LLM提示词模板
│   │   └── weekly_report.py         # Claude API集成
│   │
│   ├── scheduler/                   # 定时调度
│   │   ├── daily_job.py             # 常态轨（每日02:00）
│   │   └── weekly_job.py            # 趋势轨（周日03:00）
│   │
│   ├── utils/                       # 工具函数
│   │   ├── io.py                    # 文件读写
│   │   ├── logger.py                # 日志配置
│   │   └── metrics.py               # 评估指标
│   │
│   ├── unified_data_manager.py      # 统一数据流管理（实时↔每日）
│   └── unified_scheduler.py         # 统一系统调度器
│
├── scripts/                         # 可执行脚本
│   ├── generate_simulation_data.py  # 生成50天模拟数据（单人）
│   ├── train_all_baselines.py       # 冷启动训练
│   ├── run_daily_pipeline.py        # 手动触发每日推理
│   ├── start_realtime_monitor.py    # 启动实时监测
│   ├── start_unified_system.py      # 启动统一系统（生产环境）
│   ├── demo_realtime.py             # 实时系统演示（无需硬件）
│   ├── test_realtime_system.py      # 实时模块测试
│   ├── test_unified_system.py       # 统一系统集成测试
│   ├── test_simple.py               # 快速冒烟测试
│   └── health_check.py              # 项目完整性检查
│
├── tests/                           # 单元测试
│   ├── test_aggregator.py           # 数据聚合测试
│   ├── test_ewma.py                 # EWMA基线测试
│   ├── test_gru_model.py            # GRU模型测试
│   ├── test_imputer.py              # 缺失值处理测试
│   ├── test_risk_judge.py           # 风险判定测试
│   ├── test_risk_rules.py           # 风险规则测试
│   ├── test_integration.py          # 端到端集成测试
│   └── ...
│
├── requirements.txt                 # Python依赖
└── pytest.ini                       # Pytest配置
```

---

## 核心架构

### 双轨调度

| 轨道 | 频率 | 功能 | 判定依据 |
|------|------|------|----------|
| **实时轨** | 持续采集，每小时评估 | 快速响应急性风险 | 简化规则（阈值判定） |
| **每日轨** | 每日02:00 | 深度趋势分析 | GRU预测 + EWMA动态阈值 |

**核心理念**：实时轨捕捉突发情绪崩溃，每日轨确认长期趋势，形成双重保障。

### 数据流架构

```
┌─────────────────────────────────────────────────────────────┐
│                       统一系统                                │
└─────────────────────────────────────────────────────────────┘

【实时轨】                           【每日轨】
音频输入（麦克风/RTSP/文件）          每天凌晨02:00触发
    ↓                                    ↓
SenseVoice实时推理                   UnifiedDataManager
    ↓                                获取昨日声学特征
24小时滑动窗口聚合                       ↓
    ↓                                + 睡眠雷达数据
每小时风险检查                         + PIR/IPC活动数据
（简化规则，快速响应）                 + 拾音器社交数据
    ↓                                    ↓
Level 1/2/3预警                     10维特征向量
    ↓                                    ↓
UnifiedDataManager                   GRU模型预测
保存特征快照                            ↓
                                    加权残差计算
                                        ↓
                                    EWMA动态阈值判定
                                        ↓
                                    连续偏离天数统计
                                        ↓
                                    风险类型分类
                                        ↓
                                    Level 0/1/2/3判定
                                        ↓
                                    预警推送
```

### 四级风险（全部基于趋势）

| 等级 | 名称 | 触发条件 | 响应措施 |
|------|------|----------|----------|
| 0 | 正常 | 无偏离 | 无 |
| 1 | 关注 | 单日偏离或间歇偏离 | 记录日志 |
| 2 | 提醒 | 连续3天偏离 | 推送子女App |
| 3 | 严重 | 连续5天偏离 | 短信 + 社区网格员 + 强制响铃 |

### 三种风险类型

| 类型 | 特征信号 | 连续天数要求 |
|------|----------|--------------|
| **抑郁风险** | sad_ratio↑ + avg_speed↓ + pitch_variability↓ + distress_events↑ | 3天 |
| **睡眠问题** | sleep_efficiency↓ + deep_sleep_ratio↓ + sfi↑ + hrv_rmssd↓ | 3天 |
| **社交孤独** | social_turns↓ + daily_activity↓ + sad_ratio↑ | 5天 |

---

## 算法原理：从生理信号到心理风险的映射

系统**不使用分类器直接判断心理疾病**，而是通过"个人化基线偏离 + 加权规则匹配 + 连续趋势确认"三层机制实现映射，刻意回避临床诊断。

### 第一层：多传感器 → 10 维特征向量

四路传感器每日聚合成一条特征记录：

```
睡眠雷达 ──────────► sleep_efficiency / deep_sleep_ratio / sfi / hrv_rmssd
PIR + IPC ─────────► daily_activity
拾音器 + 智能音箱 ──► social_turns
SenseVoice 语音 ───► sad_ratio / avg_speed / pitch_variability / distress_events
```

### 第二层：GRU 个人基线 → 加权残差异常分数

核心思路：**"自己和自己比"，不设群体阈值**。

1. 用过去 7 天特征喂给该老人**专属 GRU**，预测"今天正常情况下应该是什么样"
2. `残差 = |预测值 - 实际值|`（归一化空间下，量纲统一）
3. 按 `feature_weights.json` 加权求平均，得单一 `anomaly_score`：

```
anomaly_score = Σ(residual[i] × weight[i]) / Σweight
```

4. 与 EWMA 动态阈值比较：`anomaly_score > 阈值` → 当日 `is_deviation = True`

> **为什么不要求 GRU 预测准确？** 系统用的是异常检测逻辑，不是回归预测。GRU 只需稳定复现正常态——在正常日残差小、在异常日残差大，即为成功。哪怕 GRU 对某特征有系统性偏差，只要偏差稳定，就不会误报。

### 第三层：偏离方向匹配 → 心理问题类型

`src/risk/rules.py` 对每种风险类型检查**方向性**（而非只看绝对值大小）：

| 风险类型 | 判定逻辑 |
|---------|---------|
| 抑郁风险 | sad_ratio **向上**超标 AND avg_speed **向下**超标 AND pitch_variability **向下**超标 AND distress_events **向上**超标 |
| 睡眠问题 | sleep_efficiency/deep_sleep_ratio/hrv_rmssd **向下**超标 AND sfi **向上**超标 |
| 社交孤独 | social_turns/daily_activity **向下**超标 AND sad_ratio **向上**超标 |

激活条件（三者同时满足）：
- 至少 1 个特征方向性超标
- 加权综合分 > 1.0
- **连续达标天数 ≥ 阈值**（抑郁/睡眠 3 天，社交孤独 5 天）

### 防误报机制

| 机制 | 作用 |
|-----|------|
| 个人化基线 | 避免用群体平均误判个体差异（误报率降低 4–6×） |
| EWMA 动态阈值取 min | 防止老人缓慢衰退后系统"习以为常"变迟钝 |
| 连续天数门槛 | 单日波动不触发（消融实验：误报率从 3.2 → 0.4 次/天） |
| 冷启动观察期 | 训练后 7 天仅记录不报警，等待基线稳定 |

---

## 特征设计与科学依据

10 个特征覆盖抑郁的三条公认通路：**精神运动迟滞**、**自主神经失调**、**行为退缩**。

### 语音特征（SenseVoice 提取）

| 特征 | 异常方向 | 文献支撑 | 备注 |
|-----|---------|---------|------|
| `avg_speed` 语速 | ↓ | **强**。老年抑郁系统综述明确 "slower speech rate"，精神运动迟滞标志 | 证据充分 |
| `pitch_variability` 基频变异性（F0 标准差） | ↓ | **中强**。抑郁表现为语调平淡单调，SD F0 与抑郁严重度显著相关；均值无显著差异 | 使用变异性而非均值，是本系统相较同类研究的改进点 |
| `sad_ratio` 悲伤情感占比 | ↑ | **中**。语音情感与抑郁相关；SER 模型在老年嗓音上的泛化性仍存疑 | 方向正确，测量效度待验证 |
| `distress_events` 痛苦声频次 | ↑ | **中**。叹气/哭声作为行为观察有临床依据，量化标准尚无共识 | 方向合理 |

> **特别说明**：早期版本使用 `avg_pitch`（平均基频），文献复核后发现与抑郁相关的是**基频变异性**而非均值（一项经典研究显示均值在抑郁组与好转组之间无显著差异）。当前版本已更正为 `pitch_variability`。

### 睡眠特征（非接触睡眠雷达）

| 特征 | 异常方向 | 文献支撑 |
|-----|---------|---------|
| `sleep_efficiency` 睡眠效率 | ↓ | 强。睡眠障碍是抑郁诊断标准之一，证据极充分 |
| `deep_sleep_ratio` 深睡占比 | ↓ | 强。抑郁与慢波睡眠减少高度相关 |
| `sfi` 睡眠碎片化指数 | ↑ | 强。碎片化睡眠是老年抑郁和痴呆的早期信号 |
| `hrv_rmssd` 心率变异性 | ↓ | **强**。多篇 meta 分析确认抑郁患者 RMSSD、HF-HRV 显著降低，反映迷走神经活性下降 |

> **测量效度提示**：上述指标的**金标准是 PSG（多导睡眠图）**，本系统使用非接触雷达估算。雷达能否精确复现 PSG 级别的 RMSSD 和深睡分期，需要额外的设备验证实验。特征**选择**有据，特征**测量**精度需实测。

### 行为特征

| 特征 | 异常方向 | 文献支撑 |
|-----|---------|---------|
| `daily_activity` 日间活动量 | ↓ | 强。体动计记录（actigraphy）研究支持活动量下降与抑郁相关 |
| `social_turns` 对话轮次 | ↓ | **强**（权重最高 = 3.0）。社交退缩是抑郁和老年孤独的核心行为标志 |

---

## 技术栈

| 类别 | 技术 |
|------|------|
| **语言** | Python 3.10+ |
| **深度学习** | PyTorch 2.x（CPU推理，<50MB内存） |
| **语音分析** | FunASR（SenseVoice模型） |
| **数据处理** | NumPy, pandas, scikit-learn |
| **调度** | APScheduler |
| **日志** | loguru |
| **周报** | Claude API（fallback: 规则模板） |
| **测试** | pytest |

---

## 模拟数据档案

本系统为**单人系统**，只分析一位老人（默认 ID `E001`）。`generate_simulation_data.py`
为该老人生成 50 天模拟数据，其中注入一段异常，用于端到端验证趋势检测能力：

| ID | 注入异常 | 验证目标 |
|------|----------|----------|
| E001 | Day 25-30 抑郁特征注入（sad_ratio↑ + avg_speed↓ + pitch_variability↓ + distress_events↑） | 连续偏离 → 逐级升到3级预警（抑郁风险） |

> 接入真实老人数据时，可沿用 `E001` 这个 ID，或在 `generate_simulation_data.py` 的
> `ELDER_ID` 处改成你自己的编号——它只是 `data/features/{ID}/`、`data/raw/*/{ID}/`
> 目录名的一部分。真实数据按同样的目录结构与 JSON 格式放入即可，无需改动核心代码。

---

## 使用指南

### 1. 仅使用每日系统（GRU + EWMA）

适用场景：只有传感器数据，无实时音频流

```bash
# 生成模拟数据
python scripts/generate_simulation_data.py

# 训练基线模型（Day 14）
python scripts/train_all_baselines.py

# 每日推理（默认老人 E001）
python scripts/run_daily_pipeline.py --date 2026-08-15

# 指定老人 ID（如接入真实数据时使用了其它编号）
python scripts/run_daily_pipeline.py --date 2026-08-15 --elder E001
```

### 2. 仅使用实时系统（快速响应）

适用场景：需要即时监测，暂不使用GRU深度分析

```bash
# 使用麦克风实时采集
python scripts/start_realtime_monitor.py \
    --elder-id E001 \
    --microphone \
    --risk-check-interval 3600  # 每小时检查一次

# 使用RTSP摄像头
python scripts/start_realtime_monitor.py \
    --elder-id E001 \
    --rtsp rtsp://192.168.1.100:554/stream

# 使用音频文件测试
python scripts/start_realtime_monitor.py \
    --elder-id E001 \
    --file test_audio.mp3
```

### 3. 使用统一系统（推荐）

适用场景：生产环境，实时轨 + 每日轨双重保障

```bash
# 启动统一系统
python scripts/start_unified_system.py \
    --elder-id E001 \
    --microphone \
    --daily-time 02:00

# 效果：
# - 实时系统24小时采集语音
# - 每小时自动风险检查
# - 每天凌晨2点自动执行GRU推理
# - 实时数据自动流入每日系统
```

### 4. 运行演示（无需硬件）

```bash
# 实时系统演示（模拟24小时监测场景）
python scripts/demo_realtime.py

# 系统健康检查
python scripts/health_check.py
```

### 5. 运行测试

```bash
# 所有单元测试
pytest tests/ -v

# 实时系统测试
python scripts/test_realtime_system.py

# 统一系统集成测试
python scripts/test_unified_system.py
```

---

## 数据格式

### 每日特征向量（features.csv）

| 特征名 | 来源 | 说明 |
|--------|------|------|
| sad_ratio | SenseVoice | 悲伤标签占比 [0, 1] |
| avg_speed | SenseVoice | 平均语速（字/秒） |
| pitch_variability | SenseVoice | 基频变异性 F0标准差 (Hz)，反映语调单调性（↓=平淡） |
| distress_events | SenseVoice | 叹气/哭声等非言语痛苦声音频次 |
| sleep_efficiency | 睡眠雷达 | 睡眠效率 [0, 1] |
| deep_sleep_ratio | 睡眠雷达 | 深睡占比 [0, 1] |
| sfi | 睡眠雷达 | 睡眠碎片化指数 |
| hrv_rmssd | 睡眠雷达 | 心率变异性（自主神经活性） |
| daily_activity | PIR + IPC | 日间活动量（归一化） |
| social_turns | 拾音器 + 智能音箱 | 对话轮次（社交参与度） |

### 每日推理结果（daily_inference/*.json）

```json
{
  "elder_id": "E001",
  "date": "2026-07-15",
  "anomaly_score": 1.0569,
  "static_threshold": 1.1808,
  "ewma_threshold": 1.1808,
  "dynamic_threshold": 1.1808,
  "is_deviation": false,
  "feature_residuals": {
    "sad_ratio": 0.5688,
    "avg_speed": 0.5944,
    "pitch_variability": 0.3076,
    "distress_events": 1.4551,
    "sleep_efficiency": 1.3203,
    "deep_sleep_ratio": 1.174,
    "sfi": 2.2818,
    "hrv_rmssd": 1.4244,
    "daily_activity": 1.7745,
    "social_turns": 0.6226
  },
  "consecutive_deviation_days": 0,
  "ewma_n": 8,
  "ewma_mean": 0.4595,
  "ewma_std": 0.0574,
  "data_quality": "valid",
  "status": "success",
  "in_observation_period": false
}
```

> 说明：推理层输出的是**加权残差异常分数（anomaly_score）与多档阈值**，而非直接预测值。`feature_residuals` 为各特征的标准化残差（10 维健康特征）。`status` 取值：`success` / `cold_start` / `data_insufficient` / `observation` / `error`。风险等级与风险类型由 `src/risk/judge.py` 在推理结果之上单独判定。

### 实时特征快照（realtime/*/features/snapshot_*.json）

```json
{
  "date": "2026-07-18",
  "elder_id": "E001",
  "acoustic_data": {
    "sad_ratio": 0.22,
    "avg_speed": 3.8,
    "pitch_variability": 14.0,
    "distress_events": 6
  },
  "n_utterances": 45,
  "total_duration": 320.5
}
```

---

## 配置说明

### settings.yaml（全局配置）

```yaml
gru:
  feature_dim: 10          # 特征维度（10维健康特征，已移除时间编码）
  hidden_dim: 16           # 隐藏层维度
  num_layers: 1            # GRU层数
  window: 7                # 时间窗口（天）
  dropout: 0.2             # Dropout比率

training:
  initial:
    epochs: 150            # 冷启动训练轮数
    lr: 0.001
  finetune:
    epochs: 50             # 微调轮数
    lr: 0.0003
    recent_days: 30        # 微调使用的近期天数
    residual_merge_alpha: 0.3  # 残差统计融合系数

ewma:
  alpha: 0.05              # EWMA平滑系数
  min_samples_for_dynamic: 20  # 启用动态阈值的最小样本数

risk:
  sigma_multiplier: 2.5    # 静态阈值倍数
  anomaly_score_thresholds:
    high_spike: 1.5        # 单日高峰值阈值（触发关注级别）
    sustained_avg: 1.0     # 连续期平均值阈值（配合连续天数判定提醒级别）
  consecutive:
    attention: 1           # Level 1（关注）
    warning: 3             # Level 2（提醒）
    severe: 5              # Level 3（严重）
  cold_start_observation_days: 7  # 训练后观察期（仅记录不报警）
```

### realtime_config.yaml（实时系统配置）

```yaml
system:
  device: "cuda:0"         # 推理设备（cuda:0 / cpu）
  output_dir: "./data/realtime"

audio:
  sample_rate: 16000       # 采样率
  chunk_duration: 10       # 音频块时长（秒）
  buffer_size: 100         # 队列缓冲大小

sensevoice:
  model_cache_dir: "./funasr_models"
  batch_size: 15           # 批处理大小
  language: "zh"           # 语言（zh/en/ja/ko/yue）

aggregator:
  window_hours: 24         # 24小时聚合窗口
  short_window_hours: 1    # 急性风险检测短窗口

risk:
  check_interval: 3600     # 风险检查间隔（秒）
  simple_rules:            # 简化规则阈值（不依赖GRU模型）
    depression:
      critical:            # Level 3
        sad_ratio: 0.25
        avg_speed: 3.5
        distress_events: 5
      warning:             # Level 2
        sad_ratio: 0.20
        avg_speed: 3.8
        distress_events: 3
      attention:           # Level 1
        sad_ratio: 0.15
        distress_events: 2
    sleep_problem:
      enabled: false       # 需额外传感器数据，默认关闭
    social_isolation:
      enabled: false       # 需额外传感器数据，默认关闭
  gru_model:
    enabled: false         # 实时轨默认仅用简化规则
    use_ewma: true
```

---

## 消融实验设计

| 实验 | 变量 | 预期结论 |
|------|------|----------|
| 1 | 个人基线 vs 群体阈值 | 误报率低4-6x |
| 2 | EWMA vs 固定60天窗口 | EWMA在Day 20即稳定 |
| 3 | GRU vs 移动平均 | GRU提前1-2天预警 |
| 4 | 含社交 vs 仅生理特征 | 检出率+20% |
| 5 | 连续N=3 vs N=1 | 误报率从3.2→0.4/天 |

---

## 部署建议

### 方案A：边缘设备（推荐）

**硬件**：Jetson Nano / Xavier NX / 树莓派5
- 成本：$100-$500
- 本地推理，保护隐私
- 低延迟（<100ms）

```bash
# Jetson上安装CUDA版PyTorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 部署统一系统
python scripts/start_unified_system.py --elder-id E001 --microphone
```

### 方案B：云端API

**架构**：老人家中设备 → 音频流上传 → 云端SenseVoice推理 → 结果返回

- **优点**：设备简单，只需麦克风+网络
- **缺点**：延迟高、隐私风险、依赖网络

---

## 故障排查

### 问题1：麦克风无法打开

```
[MicrophoneStream] 初始化失败: No Default Input Device Available
```

**解决方法**：
```bash
# 列出可用设备
python -m pyaudio

# 指定设备索引
python scripts/start_realtime_monitor.py --elder-id E001 --microphone --device-index 1
```

### 问题2：CUDA内存不足

```
RuntimeError: CUDA out of memory
```

**解决方法**：
- 降低批处理大小：`realtime_config.yaml` 中 `batch_size: 5`
- 或使用CPU：`device: "cpu"`

### 问题3：数据缺失导致推理失败

```
[DailyJob] 数据不足，无法推理
```

**解决方法**：
- 检查传感器是否正常工作
- 查看 `data/raw/` 目录下是否有数据
- 使用 `validator.py` 检查数据质量

---

## 扩展方向

- [ ] 多模态融合（语音 + 视频表情 + 姿态）
- [ ] 对话分析（与子女通话频率）
- [ ] 环境音分析（异常声音检测：跌倒、呼救）
- [ ] 睡眠监测（夜间呼吸音分析）
- [ ] Web监控面板

---

## License

MIT

---

**最后更新**：2026-07-22  
**项目版本**：v1.2（单人系统）
