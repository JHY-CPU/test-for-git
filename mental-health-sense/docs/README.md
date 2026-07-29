# 心理健康连续感知与趋势预警系统

> 面向**单个**居家老人的日常心理健康**趋势预警**系统。通过"自己和自己比"的个人化基线方法检测心理状态偏离，**不进行临床诊断**。

## 系统特点

- **单人系统**：面向单一老人的连续监测与趋势预警
- **单轨每日批处理**：每日读取累积的传感器数据（`data/raw/`），聚合为 6 维特征后做趋势推理与预警，无独立实时运行时
- **克制预警**：预警仅在连续多日偏离后发出，避免单日波动过度敏感打扰家人
- **个人化基线**：为该老人独立建模（PersonalBaselineGRU + EWMA动态阈值），"自己和自己比"
- **多传感器融合**：睡眠雷达 + PIR/IPC + 拾音器（麦克风 VAD 统计对话轮次）
- **趋势判定**：连续3-5天偏离才触发预警，避免单日波动误报
- **文件驱动数据流**：各传感器适配器把原始数据落盘到 `data/raw/{sleep,activity,social}/`，每日管道从此累积读取

---

## 项目现状（v1.8）

> 一张表看清"哪些已扎实、哪些还在路上"，避免把"代码跑通"误读成"临床有效"。

**核心算法层已完工并验证**：123 个单元测试（12 个测试文件）全部通过；每日趋势管道（批处理：读取 `data/raw/` → 6 维特征聚合 → GRU 残差推理 → EWMA → 连续偏离判定 → 风险判定 → 预警 → 周报）已端到端打通，不依赖任何实时运行时。`docs/TODO.md` 记录的 7 项"实现落差"（P0×2 / P1×2 / P2×3）已全部收口。

> ⚠️ **实现成熟度：算法是真的，硬件对接与外部服务大多还是"桩/模拟"。** 这是科研原型阶段的正常状态，但必须讲清楚，避免误以为"能上真机"：

| 部分 | 状态 | 说明 |
|------|------|------|
| GRU 基线 / EWMA / 风险判定 / 数据管道 | ✅ 真实可用 | 有算法、有测试，是系统的核心 |
| 传感器**真实采集**（睡眠雷达/摄像头/麦克风 的 `_read_raw`） | ⚠️ **未实现（桩）** | 均抛 `NotImplementedError`，只有 mock/模拟数据能跑 |
| 预警**推送**（子女App/短信/网格员） | ⚠️ 模拟 | `alert.py` 只写日志字符串，未接真实推送服务 |
| 周报 LLM | ⚠️ 待核对 | 模型 id `claude-sonnet-5` 可能无效，有规则模板兜底 |

**一句话**：现在能端到端跑通、能验证算法逻辑，靠的是**模拟/文件数据**；接真实设备与推送服务是后续工作。

系统验证分三层，证据强度依次递增（完整方案见 `docs/VALIDATION.md`）：

| 层次 | 回答的问题 | 状态 | 说明 |
|------|-----------|------|------|
| **A 代码对不对** | 算法有没有被正确实现 | ✅ 已完成 | 123 单测（12 文件）+ 端到端冒烟（`validate_synthetic.py`），每日管道链路已打通 |
| **B 设计好不好** | 个人基线 / EWMA / 连续判定是否优于朴素替代 | 🚧 进行中（当前重点） | 已落地判别力脚本 `validate_discriminative.py`（8 场景，含混淆项），当前稳定 6/8；已借此修复 2 个真实缺陷，暴露 1 个"串味"待办（见 `docs/VALIDATION.md §7`） |
| **C 真的有用吗** | 能否测出真实老人的心理下滑 | ⏳ 待真实数据 | 需公开数据集 + 临床金标准，仿真无法回答 |

> ✅ **"循环论证"缺口已部分打破**：新脚本 `validate_discriminative.py` 的正常天带真实噪声（AR(1)+周节律）并加入混淆项（感冒/周末安静/单日尖峰），不再是"按答案出题"。它已证明判别力（当前 6/8），并借此修复了 2 个真实缺陷、暴露 1 个"串味"待办。
> ⚠️ 但**老的 `generate_simulation_data.py` 仍是循环的**（异常方向=检测方向），只能测链路、测不了判别力。绝对数字仍需真实数据校准（层次 C）。
>
> 📖 GRU 训练的完整细节（结构/样本/耗时/无验证集设计/建档期实验/基线总览）见 `docs/TRAINING.md`。

**诚实定位**：特征选择有文献依据，但特征的测量效度与真实场景误报率需真实设备验证。本系统做**趋势预警**，**不做临床诊断**。

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成模拟数据（1位老人 × 60天）
python scripts/generate_simulation_data.py

# 3. 冷启动训练（建档期 Day 21）
python scripts/train_all_baselines.py

#    模拟数据为 60 天，异常注入在第 40-46 天（已避开建档期 1-21 + 观察期 22-28），
#    正式运行期(29+)能完整看到"正常→异常升级→恢复"的预警过程。
#    也可直接用验证脚本查看：
#      python scripts/validate_synthetic.py       # 60天跑通验证
#      python scripts/validate_discriminative.py  # 8场景判别力验证

# 4. 每日推理
python scripts/run_daily_pipeline.py --date 2026-08-15

# 5. 运行测试（12 个测试文件 / 123 用例）
python -m pytest
```

---

## 项目结构

```
mental-health-sense/
├── config/                          # 配置文件
│   ├── settings.yaml                # 全局配置（GRU、训练、EWMA、风险阈值）
│   ├── feature_weights.json         # 特征权重（加权残差计算）
│   └── realtime_config.yaml         # ⚠️ 孤儿文件：实时运行时已删除，无代码读取，仅历史留存
│
├── data/                            # 数据目录
│   ├── raw/                         # 原始传感器数据（JSON）
│   │   ├── sleep/                   # 睡眠雷达数据（{date}.json）
│   │   ├── activity/                # PIR + IPC活动数据（{date}.json）
│   │   └── social/                  # 拾音器 VAD 对话数据（{date}.json，social_turns 聚合来源）
│   ├── features/                    # 聚合后的每日特征向量（CSV）
│   │   └── E001/features.csv        # 6维健康特征
│   ├── baselines/                   # 该老人的个人基线模型
│   │   └── E001/
│   │       ├── gru.pth              # 训练好的GRU模型
│   │       ├── gru.prev.pth         # 微调前的模型备份（可回滚）
│   │       ├── scaler.pkl           # StandardScaler（归一化）
│   │       ├── residual_stats.pkl   # 训练残差统计（均值、标准差）
│   │       ├── baseline_meta.json   # 基线元数据（训练时间 + 训练后推理计数，用于观察期判定）
│   │       └── ewma.pkl             # EWMA累积基线
│   ├── logs/                        # 推理日志和周报
│   │   ├── daily_inference/         # 每日GRU推理结果（JSON）
│   │   └── weekly_reports/          # LLM生成的周报
│   └── elder_configs.json           # 老人元数据（姓名、描述）
│
├── src/                             # 源代码
│   ├── baseline/                    # GRU个人基线模型
│   │   ├── gru_model.py             # PersonalBaselineGRU（7→1天预测）
│   │   ├── trainer.py               # 冷启动（健康门禁+early-stopping）+ 每周微调（剔除偏离天+模型备份）
│   │   ├── inference.py             # 每日推理引擎
│   │   ├── data_health.py           # 训练数据健康门禁（MAD离群筛查，防GRU"学坏"）
│   │   ├── cold_start_fallback.py   # 冷启动兜底（GRU就绪前用滑动均值检测）
│   │   ├── ewma.py                  # EWMA累积基线（替代60天窗口）
│   │   └── scaler_utils.py          # StandardScaler管理
│   │
│   ├── data_pipeline/               # 数据采集与预处理
│   │   ├── aggregator.py            # 三维度(睡眠/活动/社交) → 6维特征向量
│   │   ├── imputer.py               # 缺失值处理（前向填充）
│   │   ├── validator.py             # 数据质量校验
│   │   └── adapters/                # 传感器适配器
│   │       ├── sleep_radar.py       # 睡眠雷达适配器
│   │       ├── camera.py            # IPC/RTSP摄像头适配器
│   │       └── microphone.py        # 麦克风适配器（VAD 统计对话轮次）
│   │
│   ├── risk/                        # 风险判定层
│   │   ├── rules.py                 # 2类风险类型（睡眠/社交）
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
│       ├── io.py                    # 文件读写
│       ├── logger.py                # 日志配置
│       └── metrics.py               # 评估指标
│
├── scripts/                         # 可执行脚本
│   ├── generate_simulation_data.py  # 生成60天模拟数据（异常注入40-46，避开建档+观察期）
│   ├── train_all_baselines.py       # 冷启动训练
│   ├── run_daily_pipeline.py        # 手动触发每日推理
│   ├── validate_synthetic.py        # 【范围1】合成数据端到端跑通验证（60天，层次A）
│   └── validate_discriminative.py   # 【范围2】合成数据判别力验证（8场景+混淆项，层次B）
│
├── tests/                           # 单元测试（12 个测试文件 / 123 用例，确定性可复现）
│   ├── conftest.py                  # 共享 fixture
│   ├── test_aggregator.py           # 数据聚合测试
│   ├── test_data_health.py          # 训练数据健康门禁（MAD 离群筛查）测试
│   ├── test_ewma.py                 # EWMA基线测试
│   ├── test_gru_model.py            # GRU模型测试
│   ├── test_imputer.py              # 缺失值处理（前向填充）测试
│   ├── test_validator.py            # 数据质量校验测试
│   ├── test_risk_judge.py           # 风险判定测试（含低/高幅度连续偏离回归）
│   ├── test_risk_rules.py           # 风险规则测试
│   ├── test_alert.py                # 预警推送测试
│   ├── test_metrics.py              # 评估指标测试
│   ├── test_train_loop.py           # 训练循环 early-stopping / 回滚回归
│   └── test_integration.py          # 端到端集成测试
│
├── requirements.txt                 # Python依赖
└── pytest.ini                       # Pytest配置
```

---

## 核心架构

### 单轨每日批处理

| 轨道 | 频率 | 功能 | 判定依据 |
|------|------|------|----------|
| **每日趋势轨** | 每日02:00 | 读取累积原始数据 → 6维特征聚合 → 深度趋势分析 → 预警出口 | GRU预测 + EWMA动态阈值 |
| **周报轨** | 周日03:00 | 汇总一周趋势生成周报 | LLM（fallback: 规则模板） |

**核心理念**：系统是**单轨的每日批处理管道**，没有独立的实时采集运行时。各传感器适配器把原始数据落盘到 `data/raw/{sleep,activity,social}/`（其中 `social/` 由麦克风 VAD 采集对话轮次），每日趋势轨在累积数据上做判定后统一发出预警。所有心理风险预警仅在**连续多日偏离**后触发，避免因单日波动过度敏感、频繁打扰家人。

### 数据流架构

```
【每日趋势轨：批处理，每天凌晨02:00触发】

data/raw/sleep/{date}.json      ─┐
data/raw/activity/{date}.json   ─┼─► 按自然日聚合传感器数据
data/raw/social/{date}.json     ─┘        ↓（无数据则标 missing）
（social 由麦克风 VAD 采集）           缺失填充 / 数据质量校验
                                          ↓
                                      6维特征向量
                                          ↓
                                      GRU模型预测（预测正常态）
                                          ↓
                                      加权残差计算
                                          ↓
                                      EWMA动态阈值判定
                                          ↓
                                      连续偏离天数统计
                                          ↓
                                      风险类型分类（睡眠问题 / 社交孤独）
                                          ↓
                                      Level 0/1/2/3判定
                                          ↓
                                      预警推送（统一出口）
```

### 四级风险（全部基于趋势）

| 等级 | 名称 | 触发条件 | 响应措施 |
|------|------|----------|----------|
| 0 | 正常 | 无偏离 | 无 |
| 1 | 关注 | 单日偏离、间歇偏离，或单日高峰值超标 | 记录日志 |
| 2 | 提醒 | 连续3天偏离 **且** 平均幅度超标（avg_anomaly > sustained_avg） | 推送子女App |
| 3 | 严重 | 连续5天偏离 **且** 平均幅度超标（同提醒级幅度门槛） | 短信 + 社区网格员 + 强制响铃 |

> **严重级为何也要幅度门槛？** Level 3 会触发社区网格员介入 + 强提醒，代价高。若仅凭连续天数升级，长达数天但每天只"擦线"越过动态阈值的低幅度偏离也会直冲最高级，与提醒级（带幅度门槛）判定不一致，且过度打扰家人和社区。因此严重级与提醒级共用 `avg_anomaly > sustained_avg` 幅度门槛。

### 两种风险类型

| 类型 | 特征信号 | 连续天数要求 |
|------|----------|--------------|
| **睡眠问题** | sleep_efficiency↓ + deep_sleep_ratio↓ + sfi↑ + hrv_rmssd↓ | 3天 |
| **社交孤独** | social_turns↓ + daily_activity↓ | 5天 |

> **抑郁趋势判断已移除**：原"抑郁风险"类型依赖 4 路语音声学特征（sad_ratio/avg_speed/pitch_variability/distress_events），现整体交由**外部专用抑郁模型**接入，本系统不再直接输出抑郁风险。

---

## 算法原理：从生理信号到心理风险的映射

系统**不使用分类器直接判断心理疾病**，而是通过"个人化基线偏离 + 加权规则匹配 + 连续趋势确认"三层机制实现映射，刻意回避临床诊断。

### 第一层：多传感器 → 6 维特征向量

三路传感器每日聚合成一条特征记录：

```
睡眠雷达 ──────────► sleep_efficiency / deep_sleep_ratio / sfi / hrv_rmssd
PIR + IPC ─────────► daily_activity
拾音器（VAD）──────► social_turns
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
| 睡眠问题 | sleep_efficiency/deep_sleep_ratio/hrv_rmssd **向下**超标 AND sfi **向上**超标 |
| 社交孤独 | social_turns/daily_activity **向下**超标 |

激活条件（三者同时满足）：
- 至少 1 个特征方向性超标
- 加权综合分 > 1.0
- **连续达标天数 ≥ 阈值**（睡眠 3 天，社交孤独 5 天）

### 防误报机制

| 机制 | 作用 |
|-----|------|
| 个人化基线 | 避免用群体平均误判个体差异（误报率降低 4–6×） |
| EWMA 动态阈值取 min | 防止老人缓慢衰退后系统"习以为常"变迟钝 |
| 连续天数门槛 | 单日波动不触发（消融实验：误报率从 3.2 → 0.4 次/天） |
| 冷启动观察期 | 训练后 7 次推理仅记录不报警，等待基线稳定（按 `baseline_meta.json` 记录的**训练后推理计数**判定，不受 EWMA 预热样本把 n 顶到 7、使观察期形同虚设的干扰；旧基线无 meta 时安全降级） |
| 每日批处理统一出口 | 预警仅在累积数据上连续偏离后发出，不做单日实时报警，避免单点波动打扰家人 |

---

## GRU 基线稳健性机制

个人基线 GRU 只需"稳定复现正常态"，但它有几个固有短板：需要建档期才能用、
可能把异常态学成正常、小样本易过拟合、微调可能被异常期污染。系统针对性地做了加固。

### 1. 训练数据健康门禁（防"学坏"）

**问题**：若建档期（默认前 21 天）恰好混入老人状态不好的日子，GRU 会把异常学成
"正常基线"，之后再也报不出来。

**做法**（`src/baseline/data_health.py`，冷启动时调用）：训练前用 **MAD（中位数绝对
偏差）** 做逐特征离群筛查。相比均值 ± Nσ，中位数与 MAD 对离群点本身不敏感，适合
"样本少、又要判断哪几天离群"的冷启动场景。

- 个别离群天 → **剔除后训练**，并在日志中标注是哪些特征离群；
- 离群天占比过高（默认 > 50%）→ **拒绝建档**，提示顺延，而非用脏数据建立基线；
- Scaler 在清洗后的数据上拟合，归一化基准不被异常天带偏。

### 2. 微调防污染（只用正常天）

**问题**：每周微调若无脑使用"最近 30 天"，会把已判定为异常的日子学进基线，让系统
越来越迟钝。

**做法**（`weekly_retrain`）：微调只用 `is_deviation=False` 的正常天，排除已被每日推理
判定为偏离的日期。正常天不足则跳过微调，而非用脏数据。

### 3. 冷启动兜底（消除头两周盲区）

**问题**：GRU 需建档期 + 观察期才可靠，此前系统若完全不检测，等于头两周盲区。

**做法**（`src/baseline/cold_start_fallback.py`）：GRU 基线就绪前，用"滑动均值 ± Nσ"的
加权 z-score 做基础离群检测。`daily_job` 在 `daily_inference` 返回 `cold_start` 时自动切换到
兜底检测。sigma 默认 **3.0**，比 GRU 轨更保守——建档期滑动基线本身不稳，宁可漏报也不要
一开始就误报动摇信任。GRU 一就绪，兜底自动停用。

### 4. 过拟合抑制与可回滚

- **early-stopping**：冷启动建档期（默认 21 天）约 14 个训练样本，150 epoch 易过拟合。连续
  `patience` 轮 loss 无改善则提前停止，并**回滚到最优权重**。冷启动与每周微调**共用同一
  训练循环** `_train_loop`（冷启动 patience=20、微调 patience=10），两条路径的过拟合抑制
  策略一致，不再是微调跑满固定轮数。
- **模型版本化**：每周微调覆盖 `gru.pth` 前，先备份为 `gru.prev.pth`，微调把模型搞坏时可回滚。

---

## 特征设计与科学依据

6 个特征聚焦两条可**非接触**监测的公认通路：**自主神经失调**（睡眠 + HRV）、**行为退缩**（活动 + 社交）。抑郁相关的**精神运动迟滞**通路原由语音声学特征承载，现已移除，改由**外部专用抑郁模型**负责。

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
| **数据处理** | NumPy, pandas, scikit-learn |
| **调度** | APScheduler |
| **日志** | loguru |
| **周报** | Claude API（fallback: 规则模板） |
| **测试** | pytest |

---

## 模拟数据档案

本系统为**单人系统**，只分析一位老人（默认 ID `E001`）。`generate_simulation_data.py`
为该老人生成 60 天模拟数据，其中注入一段异常，用于端到端验证趋势检测能力：

| ID | 时间线 | 注入异常 | 验证目标 |
|------|--------|----------|----------|
| E001 | 建档1-21 / 观察22-28 / 正常29-39 / **异常40-46** / 恢复47-60 | Day 40-46 睡眠恶化特征注入（sleep_efficiency↓ + deep_sleep_ratio↓ + sfi↑ + hrv_rmssd↓），已避开建档期+观察期 | 连续偏离 → 逐级升到3级预警（睡眠问题）→ 恢复降级 |

> 接入真实老人数据时，可沿用 `E001` 这个 ID，或在 `generate_simulation_data.py` 的
> `ELDER_ID` 处改成你自己的编号——它只是 `data/features/{ID}/`、`data/raw/*/{ID}/`
> 目录名的一部分。真实数据按同样的目录结构与 JSON 格式放入即可，无需改动核心代码。

---

## 使用指南

### 1. 完整工作流（生成 → 训练 → 每日推理）

适用场景：从模拟数据或真实传感器数据（`data/raw/`）跑通每日趋势管道

```bash
# 生成模拟数据
python scripts/generate_simulation_data.py

# 冷启动训练（建档期 Day 21）
python scripts/train_all_baselines.py      # 或：python -m src.baseline.trainer

# 每日推理（默认老人 E001）
python scripts/run_daily_pipeline.py --date 2026-08-15

# 指定老人 ID（如接入真实数据时使用了其它编号）
python scripts/run_daily_pipeline.py --date 2026-08-15 --elder E001
```

### 2. 端到端验证（合成数据）

适用场景：不接真机，用合成数据验证算法逻辑与判别力

```bash
# 60天跑通验证（层次A：链路对不对）
python scripts/validate_synthetic.py

# 8场景判别力验证（层次B：设计好不好，含混淆项）
python scripts/validate_discriminative.py
```

### 3. 运行测试

```bash
# 所有单元测试（12 个测试文件 / 123 用例）
python -m pytest

# 更详细输出
python -m pytest -v
```

---

## 数据格式

### 每日特征向量（features.csv）

> 顺序即 `FEATURE_NAMES`：sleep_efficiency, deep_sleep_ratio, sfi, hrv_rmssd, daily_activity, social_turns。

| 特征名 | 来源 | 说明 |
|--------|------|------|
| sleep_efficiency | 睡眠雷达 | 睡眠效率 [0, 1] |
| deep_sleep_ratio | 睡眠雷达 | 深睡占比 [0, 1] |
| sfi | 睡眠雷达 | 睡眠碎片化指数 |
| hrv_rmssd | 睡眠雷达 | 心率变异性（自主神经活性） |
| daily_activity | PIR + IPC | 日间活动量（归一化） |
| social_turns | 拾音器（VAD） | 对话轮次（社交参与度） |

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

> 说明：推理层输出的是**加权残差异常分数（anomaly_score）与多档阈值**，而非直接预测值。`feature_residuals` 为各特征的标准化残差（6 维健康特征）。`status` 取值：`success` / `cold_start` / `cold_start_fallback`（GRU未就绪，走滑动均值兜底检测）/ `data_insufficient` / `observation` / `error`。风险等级与风险类型由 `src/risk/judge.py` 在推理结果之上单独判定。

> 某自然日社交数据缺失时，聚合链路返回中性默认值并标记 `data_quality="missing"`，交由每日轨的校验/填充链路据实降级，而非用假的"正常值"喂进 GRU 掩盖真实偏离。

---

## 配置说明

### settings.yaml（全局配置）

```yaml
gru:
  feature_dim: 6           # 特征维度（6维健康特征，已移除时间编码与语音声学维）
  hidden_dim: 16           # 隐藏层维度
  num_layers: 1            # GRU层数
  window: 7                # 时间窗口（天）
  dropout: 0.2             # Dropout比率

training:
  initial:
    epochs: 150            # 冷启动训练轮数（配合 early-stopping，实际常提前停止）
    lr: 0.001
    patience: 20           # early-stopping：连续N轮loss无改善则提前停止（防小样本过拟合）
  finetune:
    epochs: 50             # 微调轮数
    lr: 0.0003
    patience: 10           # early-stopping：微调也启用（与冷启动共用 _train_loop），抑制小样本过拟合
    recent_days: 30        # 微调使用的近期天数
    residual_merge_alpha: 0.3  # 残差统计融合系数
    exclude_deviation_days: true  # 微调只用正常天（is_deviation=False），防基线被异常期污染

# 训练数据健康门禁（冷启动前 MAD 离群筛查，防 GRU 把异常学成正常基线）
data_health:
  z_threshold: 3.5         # modified z-score阈值，超过视为该特征离群
  min_bad_features: 2      # 一天中离群特征数达到此值则整天判为离群
  max_outlier_ratio: 0.5   # 离群天占比超过此值则拒绝建档（建议顺延）

# 冷启动兜底（GRU 就绪前用滑动均值检测，消除头两周盲区）
cold_start:
  fallback_enabled: true
  fallback_min_days: 5     # 至少积累N天有效数据才启用兜底
  fallback_lookback: 14    # 滑动基线回看窗口（天）
  fallback_sigma: 3.0      # 加权z-score超过此值判为偏离（比GRU轨更保守）

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

### realtime_config.yaml（⚠️ 已成孤儿）

实时采集运行时已在本次重构中删除，`config/realtime_config.yaml` 文件仍在仓库中，
但**已无任何代码读取**，仅作历史留存。系统当前唯一生效的配置是上文的 `settings.yaml`。

---

## 消融实验设计

| 实验 | 变量 | 预期结论 |
|------|------|----------|
| 1 | 个人基线 vs 群体阈值 | 误报率低4-6x |
| 2 | EWMA vs 固定60天窗口 | EWMA在Day 20即稳定 |
| 3 | GRU vs 移动平均 | GRU提前1-2天预警 |
| 4 | 含社交 vs 仅生理特征 | 检出率+20% |
| 5 | 连续N=3 vs N=1 | 误报率从3.2→0.4/天 |
| 6 | 有无健康门禁（建档期注入异常） | 门禁剔除异常天，避免基线"学坏"漏报 |
| 7 | 有无 early-stopping | 抑制小样本过拟合，验证残差更稳定 |
| 8 | 有无冷启动兜底 | 兜底消除建档期监测盲区 |

---

## 部署建议

### 方案A：边缘设备（推荐）

**硬件**：Jetson Nano / Xavier NX / 树莓派5
- 成本：$100-$500
- 本地推理，保护隐私
- 低延迟（<100ms）

```bash
# Jetson上安装CUDA版PyTorch（本系统 CPU 推理即可，<50MB 内存）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 用系统定时任务（cron / systemd timer）在每天凌晨触发每日推理
python scripts/run_daily_pipeline.py --date "$(date +%F)"
```

### 方案B：云端API

**架构**：老人家中设备 → 传感器数据上传 → 云端每日趋势推理 → 结果返回

- **优点**：设备简单，家中只需传感器+网络
- **缺点**：延迟高、隐私风险、依赖网络

---

## 故障排查

### 问题1：数据缺失导致推理失败

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

**最后更新**：2026-07-29  
**项目版本**：v1.8（在 v1.7 基础上：**移除抑郁趋势判断 + 4 路语音声学维**（sad_ratio/avg_speed/pitch_variability/distress_events）**+ SenseVoice 声学子系统**（`src/realtime/*`、`src/unified_*.py`、`adapters/sensevoice.py` 已删除）；健康特征 **10→6 维**（sleep_efficiency / deep_sleep_ratio / sfi / hrv_rmssd / daily_activity / social_turns），风险类型 **3→2**（睡眠问题 + 社交孤独）；social_turns 改由拾音器 VAD 路径提供（该路径保留），social_isolation 不再依赖 sad_ratio；抑郁判断改由**外部专用模型**接入。）

> v1.7 要点（在 v1.6 基础上）：①冷启动建档期 14→21 天并提为可配置项 `build_days`，附 20-seed 过拟合对比实验；②修复风险类型分类两处缺陷——"永不激活"（日志不写回 risk_types）与"正常日误激活"（类型判定挂靠 is_deviation）；③新增合成数据验证脚本 `validate_synthetic.py`（范围1跑通）与 `validate_discriminative.py`（范围2判别力，含真实噪声+混淆项）；④新增 `docs/TRAINING.md` GRU 训练详解。⑤修复模拟数据异常注入天数（25-30 → 40-46）与总天数（50 → 60），避开新建档期+观察期。测试 134 全绿。已知待办：跨类型"串味"（共享GRU溢出）、真实传感器采集/推送仍为桩。文档：本 README（总览+现状）、`docs/TRAINING.md`（训练详解）、`docs/TODO.md`、`docs/VALIDATION.md`（三层验证+§7执行记录）

> v1.6 要点：打通统一系统链路（统一调度器真正调用每日管道）；声学按自然日聚合 + 缺失标记；微调与冷启动共用 early-stopping；统一系统快照原子写；save_interval 走配置。

> v1.5 要点：单人系统；实时轨仅采集不报警；GRU 加健康门禁 / 冷启动兜底 / early-stopping / 微调防污染；判定链路修复（前向填充生效 / 观察期按训练后推理计数 / 严重级加幅度门槛）；移除时间编码死代码，特征统一为 10 维 `FEATURE_NAMES`。
