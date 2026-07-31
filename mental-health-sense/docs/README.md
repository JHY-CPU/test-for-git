# 心理健康连续感知与趋势预警系统

> 面向**单个**居家老人的日常心理健康**趋势预警**系统。通过"自己和自己比"的个人化基线方法检测心理状态偏离，**不进行临床诊断**。

## 系统特点

- **单人系统**：面向单一老人的连续监测与趋势预警
- **双轨每日批处理**：每日读取累积的传感器数据（`data/raw/`），聚合为双轨特征（睡眠 8 维 / 社交 5 维）后**各自独立**做趋势推理与预警，无独立实时运行时
- **克制预警**：预警仅在连续多日偏离后发出，避免单日波动过度敏感打扰家人
- **个人化基线**：为该老人独立建模（PersonalBaselineGRU + EWMA动态阈值），"自己和自己比"
- **多传感器融合**：小贝壳无感睡眠监测仪 + 萤石 T1C 人体移动传感器 + C6c 摄像机（边缘侧人形检测，**不出图、不采音频**）
- **趋势判定**：连续3-5天偏离才触发预警，避免单日波动误报
- **文件驱动数据流**：各传感器适配器把原始数据落盘到 `data/raw/{sleep,activity,social}/`，每日管道从此累积读取

---

## 项目现状（v2.1 双轨）

> 一张表看清"哪些已扎实、哪些还在路上"，避免把"代码跑通"误读成"临床有效"。

**核心算法层已完工并验证**：332 个单元测试全部通过；范围 1（链路正确性）20/20、范围 2（判别力）9/9。每日趋势管道（批处理：读取 `data/raw/` → 双轨特征聚合（睡眠 8 维 / 社交 5 维）→ 每轨独立 GRU 残差推理 → EWMA（偏离日冻结）→ 连续偏离判定 → 风险判定（含方向闸门，每轨各自算等级取较高者）→ 预警 → 周报）已端到端打通，不依赖任何实时运行时。

> ⚠️ **实现成熟度：算法是真的，硬件对接与外部服务大多还是"桩/模拟"。** 这是科研原型阶段的正常状态，但必须讲清楚，避免误以为"能上真机"：

| 部分 | 状态 | 说明 |
|------|------|------|
| GRU 基线 / EWMA / 风险判定 / 数据管道 | ✅ 真实可用 | 有算法、有测试，是系统的核心 |
| 传感器**真实采集**（小贝壳 / 萤石 C6c+T1C 的 `_read_raw`） | ⚠️ **未实现（桩）** | 均抛 `NotImplementedError`，只有 mock/模拟数据能跑 |
| 预警**推送**（子女App/短信/网格员） | ⚠️ 模拟 | `alert.py` 只写日志字符串，未接真实推送服务 |
| 周报 LLM | ⚠️ 待核对 | 模型 id `claude-sonnet-5` 可能无效，有规则模板兜底 |

**一句话**：现在能端到端跑通、能验证算法逻辑，靠的是**模拟/文件数据**；接真实设备与推送服务是后续工作。

系统验证分三层，证据强度依次递增（完整方案见 `docs/VALIDATION.md`）：

| 层次 | 回答的问题 | 状态 | 说明 |
|------|-----------|------|------|
| **A 代码对不对** | 算法有没有被正确实现 | ✅ 已完成 | 332 单测（14 文件）+ 端到端冒烟（`validate_synthetic.py` 20/20，含双轨信号隔离），每日管道链路已打通 |
| **B 设计好不好** | 个人基线 / EWMA / 连续判定是否优于朴素替代 | 🚧 进行中（当前重点） | 判别力脚本 `validate_discriminative.py`（10 场景，含 4 项混淆 + 1 项已知漏报），当前 **9/9**（KM 场景不计分）；已借此修复 5 个真实缺陷，仍有 2 项已知局限（见 `docs/VALIDATION.md §7`） |
| **C 真的有用吗** | 能否测出真实老人的心理下滑 | ⏳ 待真实数据 | 需公开数据集 + 临床金标准，仿真无法回答 |

> ✅ **"循环论证"缺口已部分打破**：`validate_discriminative.py` 的正常天带真实噪声（AR(1)+周末效应）并加入混淆项（单日尖峰 / 短期社交低落 / 睡眠全面好转），不再是"按答案出题"。当前 **9/9**，并借此修复了 5 个真实缺陷。
> ⚠️ 但**`generate_simulation_data.py` 仍是循环的**（异常方向=检测方向），只能测链路与信号隔离、测不了判别力。绝对数字仍需真实数据校准（层次 C）。
>
> 🔁 **两个验证脚本都已固定随机种子**（数据用 crc32、GRU 用 `torch.manual_seed`），连跑两次结果一致。这条不是锦上添花：`validate_synthetic.py` 曾因未固定 torch 种子而给出**不可复现的 19/19**，掩盖了一个真实的灵敏度缺口（社交崩塌报不出类型），补上种子后才暴露出来（见 `docs/VALIDATION.md` 缺陷⑥）。
>
> 📖 GRU 训练的完整细节（结构/样本/耗时/无验证集设计/建档期实验/基线总览）见 `docs/TRAINING.md`。

**诚实定位**：特征选择有文献依据，但特征的测量效度与真实场景误报率需真实设备验证。本系统做**趋势预警**，**不做临床诊断**。

---

## 快速开始

```bash
# 1. 安装依赖（本机只有 python3，仓库不带 venv）
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
#    核心链路只需 numpy / pandas / scikit-learn / torch / joblib / PyYAML / loguru / pytest。
#    anthropic 只用于周报正文，未安装时自动回落到规则模板。
#    2026-07-31 已移除 funasr / modelscope / pyaudio / scipy / opencv-python /
#    APScheduler 六项零引用依赖——其中 pyaudio 需要 portaudio 头文件，
#    在干净的 Linux 上会让整条 pip install 中止，torch 一个都装不上。

# 2. 生成模拟数据（1位老人 × 60天）
python scripts/generate_simulation_data.py

# 3. 双轨建档（建档期 35 天，睡眠轨/社交轨各训一个 GRU）
python scripts/train_all_baselines.py

#    模拟数据为 60 天（2026-07-01 ~ 2026-08-29），两段异常**错开**注入：
#      day 40-46 睡眠恶化   day 50-58 社交退缩
#    错开是为了验证双轨的信号隔离——一轨报警时另一轨应保持安静。
#    均已避开建档期(1-35)，正式运行期能完整看到"正常→异常升级→恢复"。
#    也可直接用验证脚本查看：
#      python scripts/validate_synthetic.py       # 范围1：60天跑通 + 信号隔离（20 项断言）
#      python scripts/validate_discriminative.py  # 范围2：10 场景判别力（含混淆项）

# 4. 每日推理
python scripts/run_daily_pipeline.py --date 2026-08-15

# 5. 周轨：双轨微调 + 周报（--no-retrain 只出周报，不动基线）
python scripts/run_weekly_pipeline.py

# 6. 运行测试（14 个测试文件 / 332 用例）
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
│   │   ├── sleep/                   # 小贝壳睡眠数据（{date}.json）
│   │   ├── activity/                # PIR + IPC活动数据（{date}.json）
│   │   └── camera/                  # C6c 边缘人形检测（{date}.json，copresence_min 聚合来源）
│   ├── features/                    # 聚合后的每日特征向量（CSV）
│   │   └── E001/                    # features_sleep.csv（8维）+ features_social.csv（5维）
│   ├── baselines/                   # 该老人的个人基线模型
│   │   └── E001/
│   │       ├── gru_{sleep,social}.pth        # 按轨的GRU模型
│   │       ├── gru_{sleep,social}.prev.pth   # 微调前的备份（可回滚）
│   │       ├── scaler_{sleep,social}.pkl     # 按轨的 StandardScaler
│   │       ├── residual_stats_{sleep,social}.pkl  # 留出段残差统计（signed + abs 两套）
│   │       ├── ewma_sleep.pkl                # 睡眠轨 EWMA（不分池）
│   │       ├── ewma_social_{weekday,weekend}.pkl  # 社交轨 EWMA（周末效应分池）
│   │       ├── ewma_{sleep,social}_state.pkl # 冻结计数 + 去重游标
│   │       └── baseline_meta.json            # 元信息（两轨共用，按 track 分键）
│   ├── logs/                        # 推理日志、证据契约和周报
│   │   ├── daily_inference/         # 每日双轨推理结果（JSON）
│   │   ├── mpdd_evidence/           # 给 MPDD-AVP 的单向证据契约（JSON）
│   │   └── weekly_reports/          # 周报（LLM 或规则模板）
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
│   │   ├── aggregator.py            # 三路传感器 → 双轨特征向量（睡眠8维 / 社交5维）
│   │   ├── imputer.py               # 缺失值处理（前向填充）
│   │   ├── validator.py             # 数据质量校验
│   │   └── adapters/                # 传感器适配器（_read_raw 均为桩，只有 mock/file 模式可跑）
│   │       ├── xiaobeike.py         # 小贝壳无感睡眠监测仪
│   │       ├── camera.py            # 萤石 C6c 边缘人形检测
│   │       ├── ezviz_events.py      # 萤石 T1C PIR 事件
│   │       └── circadian.py         # 小时活动序列 → RA / IV
│   │
│   ├── risk/                        # 风险判定层
│   │   ├── rules.py                 # 3类风险类型（睡眠稳定性/社会连接/作息节律）
│   │   ├── judge.py                 # 4级风险判定（每轨各自算等级取较高者 + 方向闸门）
│   │   └── alert.py                 # 预警推送（当前只写日志字符串；⚠️ 无冷却/去重）
│   │
│   ├── report/                      # 周报生成
│   │   ├── templates.py             # LLM提示词模板
│   │   └── weekly_report.py         # Claude API集成
│   │
│   ├── scheduler/                   # 定时任务（由 cron/systemd 触发脚本调用）
│   │   ├── daily_job.py             # 常态轨（每日03:00）
│   │   └── weekly_job.py            # 趋势轨（周日04:00）
│   │
│   ├── utils/                       # 工具函数
│       ├── io.py                    # 文件读写、路径管理
│       ├── status.py                # 推理状态枚举与"可评估"判据（单一事实来源）
│       ├── seeding.py               # 确定性种子派生（crc32，绝不用内置 hash()）
│       ├── logger.py                # 日志配置
│       └── metrics.py               # 评估指标
│
├── scripts/                         # 可执行脚本
│   ├── generate_simulation_data.py  # 生成60天双轨模拟数据（睡眠异常40-46 / 社交异常50-58，错开）
│   ├── train_all_baselines.py       # 双轨建档（两轨各一个 GRU/scaler/残差统计/EWMA）
│   ├── run_daily_pipeline.py        # 手动触发每日推理
│   ├── run_weekly_pipeline.py       # 手动触发周轨（微调 + 周报）
│   ├── validate_synthetic.py        # 【范围1】合成数据端到端跑通验证（60天，层次A）
│   └── validate_discriminative.py   # 【范围2】合成数据判别力验证（10场景+混淆项，层次B）
│
├── tests/                           # 单元测试（14 个测试文件 / 332 用例，确定性可复现）
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
│   ├── test_adapters.py             # 传感器适配器测试
│   ├── test_weekly_report.py        # 周报统计指标回归（曾恒为 0.00）
│   ├── test_train_loop.py           # 训练循环 early-stopping / 回滚回归
│   └── test_integration.py          # 端到端集成测试
│
├── requirements.txt                 # Python依赖
└── pytest.ini                       # Pytest配置
```

---

## 核心架构

### 双轨每日批处理

| 轨道 | 频率 | 功能 | 判定依据 |
|------|------|------|----------|
| **每日趋势轨** | 每日02:00 | 读取累积原始数据 → 双轨特征聚合 → 每轨独立趋势分析 → 预警出口 | 每轨 GRU 预测 + 该轨 EWMA 动态阈值 |
| **周报轨** | 周日03:00 | 汇总一周趋势生成周报 | LLM（fallback: 规则模板） |

> 注意"双轨"在本文档有两个不同含义，别混：这张表说的是**调度轨**（每日 / 每周）；
> 下文的**睡眠轨 / 社交轨**说的是特征与模型的拆分。二者互相独立。

**核心理念**：系统是**每日批处理管道**，没有独立的实时采集运行时。各传感器适配器把原始数据落盘到 `data/raw/{sleep,activity,camera}/`，每日趋势轨在累积数据上做判定后统一发出预警。特征与模型按信号来源拆成**睡眠轨（8 维）/ 社交轨（5 维）**，两轨各有独立的 GRU、scaler、残差统计与 EWMA，互不共享——这是切断跨类型"串味"的关键。所有心理风险预警仅在**连续多日偏离**后触发，避免因单日波动过度敏感、频繁打扰家人。

### 数据流架构

```
【每日趋势轨：批处理，每天凌晨02:00触发】

data/raw/sleep/{date}.json      ─┐   小贝壳无感睡眠监测仪
data/raw/activity/{date}.json   ─┼─► 萤石 T1C 人体移动传感器
data/raw/camera/{date}.json     ─┘   萤石 C6c（边缘人形检测，不出图/不采音）
                                          ↓ 按自然日聚合（无数据则标 missing）
                                      缺失填充 / 数据质量校验
                                          ↓
                          ┌───────────────┴───────────────┐
                   睡眠轨 8 维                      社交轨 5 维
                          ↓                               ↓
                 scaler_sleep 归一化              scaler_social 归一化
                          ↓                               ↓
                 gru_sleep 预测正常态             gru_social 预测正常态
                          ↓                               ↓
                 abs 残差 → 加权总分              abs 残差 → 加权总分
                 signed_z 保号（判方向）           signed_z 保号（判方向）
                          ↓                               ↓
              ewma_sleep 动态阈值           ewma_social 动态阈值（工作日/周末分池）
              （偏离日冻结，上限 14 天）      （偏离日冻结，上限 14 天）
                          ↓                               ↓
                 该轨连续偏离天数统计              该轨连续偏离天数统计
                          └───────────────┬───────────────┘
                                          ↓
                    风险类型分类（读 signed_z，方向性判定）
                    睡眠稳定性 / 社会连接减弱 / 作息节律紊乱
                                          ↓
                    Level 0/1/2/3 判定（每轨各自算，取较高者）
                    + 方向闸门：升级到 L2/L3 需有"朝坏方向"的证据
                                          ↓
                                 预警推送（统一出口）
```

> **两轨绝不跨轨比较绝对分**：8 维与 5 维的权重和不同、残差尺度不同，
> 取最大或求平均都没有统计意义，只能各自与自己的阈值比、再比较等级。

### 四级风险（全部基于趋势）

| 等级 | 名称 | 触发条件 | 响应措施 |
|------|------|----------|----------|
| 0 | 正常 | 无偏离 | 无 |
| 1 | 关注 | 单日偏离、间歇偏离，或单日高峰值超标 | 记录日志 |
| 2 | 提醒 | 连续3天偏离 **且** 平均幅度超标（avg_anomaly > sustained_avg）**且**过方向闸门 | 推送子女App |
| 3 | 严重 | 连续5天偏离 **且** 平均幅度超标（同提醒级幅度门槛）**且**过方向闸门 | 短信 + 社区网格员 + 强制响铃 |

> 等级按**每轨各自计算、取较高者**（`judge.py`），不跨轨比较绝对分。

> **严重级为何也要幅度门槛？** Level 3 会触发社区网格员介入 + 强提醒，代价高。若仅凭连续天数升级，长达数天但每天只"擦线"越过动态阈值的低幅度偏离也会直冲最高级，与提醒级（带幅度门槛）判定不一致，且过度打扰家人和社区。因此严重级与提醒级共用 `avg_anomaly > sustained_avg` 幅度门槛。

> **方向闸门是什么？** `anomaly_score` 用的是 **abs** 残差，方向无关——"睡眠效率从
> 0.88 涨到 0.97"与"掉到 0.68"产生同样大的分数。实测中一次**睡眠全面好转**因此被判到
> L3，家属会收到"您父亲睡眠严重异常"，而实情是老人睡得更好了。故 L2/L3 额外要求有
> "朝坏方向"的正面证据（读 `signed_z`）：若驱动升级的那段连续偏离全是好转，封顶在
> L1"关注"——变化本身值得留意（可能是躁狂期、也可能是数据问题），但不该发风险提醒。
> 数据缺失**不算**好转的证据，拿不到方向信息时保留原等级。详见 `docs/VALIDATION.md` 缺陷⑤。

### 三种风险类型

规则采用"**必选维 + 可选池**"结构（`src/risk/rules.py`）：必选维必须全部方向性超标，
再从可选池里凑够 `min_optional` 个，才算当天达标。

| 类型 | 必选维 | 可选池 | 持续性要求 |
|------|--------|--------|------------|
| **睡眠稳定性** `sleep_stability` | `sleep_efficiency`↓ + `waso_min`↑ | `bed_exit_count`↑ / `sol_min`↑ / `deep_sleep_ratio`↓（取 1） | 连续 3 天 |
| **社会连接减弱** `social_decline` | `copresence_min`↓ + `out_of_home_min`↓ + `activity_counts`↓ | 无 | 7 天窗内 5 天 |
| **作息节律紊乱** `circadian_disruption` | `rar_amplitude`↓ + `rar_iv`↑ | `sleep_onset_clock` 双向漂移（取 1） | 连续 5 天 |

**每条规则的 `threshold_ratio` 各自可配**（`config/settings.yaml`）。原则是
**AND 的条件越多，每个条件的门槛可以越低**，联合误报率才可比：
睡眠/节律为 1.5，`social_decline` 为 **1.0**（三项全中、无可选池）。

`social_decline` 单独降到 1.0 是端到端实测逼出来的：本轨三个必选维的**变异系数
差一个量级**（`activity_counts` CV≈13%，`copresence_min` CV≈36% 且周末 ×2.4），
同一个 ratio 对它们不是同一件事。实测社交退缩期两者原始跌幅都在 10 倍上下
（copresence 57→6 分钟），但 copresence 的 normalized 跌幅被自身大方差摊薄、
残差 std 又被工作日/周末双峰抬高（残差统计是合池估计的，7 天留出段切不出两个池），
再叠加 GRU 只看过去 7 天——异常持续到第 3 天后输入窗填满异常日、预测跟着跌，
残差进一步收缩——z 只剩 −1.09~−1.26，恰好在 1.2 门槛两侧抖动，
"7 天窗内 5 天"永远凑不满。

降门槛的安全性由**结构**而非单维门槛提供：三项 AND 再叠加"7 天滚动窗需 5 天"。
`TN_short_social`（社交低落仅 4 天）专门守这条线，确认没有拿误报换灵敏度。
详见 `config/settings.yaml` 内注释。

> **抑郁趋势判断已移除**：原"抑郁风险"类型依赖 4 路语音声学特征
> （sad_ratio/avg_speed/pitch_variability/distress_events），现整体交由
> **外部专用抑郁模型**接入，本系统不再直接输出抑郁风险。

---

## 算法原理：从生理信号到心理风险的映射

系统**不使用分类器直接判断心理疾病**，而是通过"个人化基线偏离 + 加权规则匹配 + 连续趋势确认"三层机制实现映射，刻意回避临床诊断。

### 第一层：多传感器 → 双轨特征向量（睡眠 8 维 / 社交 5 维）

传感器每日聚合成一条特征记录，再按**轨**拆成两个独立向量：

```
睡眠轨（8 维，源：小贝壳无感睡眠监测仪）
  小贝壳 ──────────► sleep_efficiency / waso_min / sol_min / bed_exit_count
                     deep_sleep_ratio / sleep_onset_clock / night_hr_mean / daytime_nap_min

社交轨（5 维，源：萤石 C6c 摄像机 + T1C 人体移动传感器，不含音频）
  C6c（边缘人形检测）─► copresence_min
  T1C + C6c 事件 ────► out_of_home_min / activity_counts / rar_amplitude / rar_iv
```

**为什么拆两轨**：v2.0 单轨用一个 GRU 联合预测全部维度，某维剧烈异常时共享隐藏状态
被带偏，其他维残差虚高，导致"睡眠异常顺带报社交孤独"（串味）。双轨各有独立的
GRU 与 scaler，从结构上切断这条通道。详见 `docs/VALIDATION.md` 缺陷③。

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
| 睡眠稳定性 | `sleep_efficiency` **向下** AND `waso_min` **向上**（必选）+ 可选池取 1 |
| 社会连接减弱 | `copresence_min`/`out_of_home_min`/`activity_counts` 三项全部**向下** |
| 作息节律紊乱 | `rar_amplitude` **向下** AND `rar_iv` **向上**（必选）+ `sleep_onset_clock` 双向漂移 |

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
| ~~冷启动观察期~~（**已取消**） | 曾为"训练后 7 次推理仅记录不报警"。建档期延长到 35 天后，EWMA 初始已有 28 个样本、动态阈值直接稳定，观察期不再必要——留着只是白白多 7 天不报警的盲区。配置项 `cold_start_observation_days` 保留但置 0，代码路径仍在（真机数据若发现阈值预热更慢，可随时调回） |
| 每日批处理统一出口 | 预警仅在累积数据上连续偏离后发出，不做单日实时报警，避免单点波动打扰家人 |

---

## GRU 基线稳健性机制

个人基线 GRU 只需"稳定复现正常态"，但它有几个固有短板：需要建档期才能用、
可能把异常态学成正常、小样本易过拟合、微调可能被异常期污染。系统针对性地做了加固。

### 1. 训练数据健康门禁（防"学坏"）

**问题**：若建档期（默认前 35 天）恰好混入老人状态不好的日子，GRU 会把异常学成
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

**问题**：GRU 需要建档期才可靠，此前系统若完全不检测，等于头几周盲区。

**做法**（`src/baseline/cold_start_fallback.py`）：GRU 基线就绪前，用"滑动均值 ± Nσ"的
加权 z-score 做基础离群检测。`daily_job` 在 `daily_inference` 返回 `cold_start` 时自动切换到
兜底检测。sigma 默认 **3.0**，比 GRU 轨更保守——建档期滑动基线本身不稳，宁可漏报也不要
一开始就误报动摇信任。GRU 一就绪，兜底自动停用。

### 4. 过拟合抑制与可回滚

- **early-stopping**：建档期（默认 35 天）滑窗后不到 30 个训练样本，150 epoch 易过拟合。连续
  `patience` 轮 loss 无改善则提前停止，并**回滚到最优权重**。冷启动与每周微调**共用同一
  训练循环** `_train_loop`（冷启动 patience=20、微调 patience=10），两条路径的过拟合抑制
  策略一致，不再是微调跑满固定轮数。
- **模型版本化**：每周微调覆盖 `gru.pth` 前，先备份为 `gru.prev.pth`，微调把模型搞坏时可回滚。

---

## 特征设计与科学依据

6 个特征聚焦两条可**非接触**监测的公认通路：**自主神经失调**（睡眠 + HRV）、**行为退缩**（活动 + 社交）。抑郁相关的**精神运动迟滞**通路原由语音声学特征承载，现已移除，改由**外部专用抑郁模型**负责。

### 睡眠轨（8 维，源：小贝壳无感睡眠监测仪）

| 特征 | 权重 | 异常方向 | 文献支撑 |
|-----|------|---------|---------|
| `sleep_efficiency` 睡眠效率 | 3.0 | ↓ | 强。睡眠障碍是抑郁诊断标准之一，证据极充分 |
| `waso_min` 入睡后觉醒时长 | 3.0 | ↑ | 强。夜间维持睡眠困难是老年抑郁的核心睡眠表型 |
| `bed_exit_count` 离床次数 | 2.5 | ↑ | 强。夜间起床频次上升与睡眠碎片化、夜尿、焦虑相关 |
| `sol_min` 入睡潜伏期 | 2.0 | ↑ | 中强。入睡困难与焦虑/反刍思维相关 |
| `sleep_onset_clock` 就寝相位 | 2.0 | **any** | 中强。相位漂移（提前或延后）都提示节律失稳，无好坏方向 |
| `deep_sleep_ratio` 深睡占比 | 1.5 | ↓ | 中。抑郁与慢波睡眠减少相关，但**受雷达分期精度限制已降权** |
| `daytime_nap_min` 日间小睡 | 1.5 | ↑ | 中。日间补眠增多反映夜间睡眠质量下降 |
| `night_hr_mean` 夜间平均心率 | 1.0 | ↑ | 中。**是 `hrv_rmssd` 的弱代理，不等价于 HRV**——小贝壳只给 BPM 级聚合值，拿不到逐拍间期，故只能低权重使用 |

> **测量效度提示**：上述指标的**金标准是 PSG（多导睡眠图）**，本系统使用非接触监测估算。
> 设备能否复现 PSG 级别的深睡分期，需额外的设备验证实验。特征**选择**有据，特征**测量**精度需实测。
> `night_hr_mean` 替代 `hrv_rmssd` 是**设备能力所限的降级**，已如实降权到 1.0 并在
> `adapters/xiaobeike.py` 内注明，不宣称等价于 HRV。

### 社交轨（5 维，源：萤石 C6c 摄像机 + T1C 人体移动传感器，不含音频）

| 特征 | 权重 | 异常方向 | 文献支撑 |
|-----|------|---------|---------|
| `copresence_min` 共处时长 | 3.0 | ↓ | 中强。本轨社会接触的**唯一**指标（边缘侧人形检测，不出图不传音） |
| `rar_amplitude` 节律振幅 | 3.0 | ↓ | 强。RAR（rest-activity rhythm）振幅下降是抑郁的稳健体动学标志 |
| `rar_iv` 节律日内变异 | 2.5 | ↑ | 强。IV 上升反映作息碎片化 |
| `activity_counts` 活动计数 | 2.5 | ↓ | 强。体动计（actigraphy）研究支持活动量下降与抑郁相关 |
| `out_of_home_min` 疑似外出时长 | 2.5 | ↓ | 中强。**测量为推断**（由室内无人推断外出），命名保留"疑似"以示不确定 |

> **对话轮次已移除**：v2.0 的 `social_turns` 依赖拾音器 VAD，音频链路整体删除后，
> 社会接触改由 C6c 边缘侧人形检测的 `copresence_min` 承担。这是**隐私换指标粒度**的取舍——
> 不再采集音频，代价是拿不到"是否在交谈"，只能知道"是否有人同处一室"。

---

## 技术栈

| 类别 | 技术 |
|------|------|
| **语言** | Python 3.10+ |
| **深度学习** | PyTorch 2.x（CPU推理，<50MB内存） |
| **数据处理** | NumPy, pandas, scikit-learn |
| **调度** | cron / systemd 触发脚本 |
| **日志** | loguru |
| **周报** | Claude API（fallback: 规则模板） |
| **测试** | pytest |

---

## 模拟数据档案

本系统为**单人系统**，只分析一位老人（默认 ID `E001`）。`generate_simulation_data.py`
为该老人生成 60 天模拟数据，其中注入一段异常，用于端到端验证趋势检测能力：

| ID | 时间线 | 注入异常 | 验证目标 |
|------|--------|----------|----------|
| E001 | 建档 1-35 / 正常 36-39 / **睡眠异常 40-46** / 恢复 47-49 / **社交异常 50-58** / 恢复 59-60 | Day 40-46 睡眠恶化（`sleep_efficiency`↓ + `waso_min`↑ + `bed_exit_count`↑）；Day 50-58 社交退缩（`copresence_min`↓ + `out_of_home_min`↓ + `activity_counts`↓），均避开建档期 | 连续偏离 → 升到 L3 → 恢复降级；**两段异常错开以验证双轨信号隔离** |

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

# 双轨建档（建档期 35 天；两轨各训一个 GRU，互不共享权重与 scaler）
python scripts/train_all_baselines.py      # 或：python -m src.baseline.trainer

# 每日推理（默认老人 E001）
python scripts/run_daily_pipeline.py --date 2026-08-15

# 指定老人 ID（如接入真实数据时使用了其它编号）
python scripts/run_daily_pipeline.py --date 2026-08-15 --elder E001
```

### 2. 端到端验证（合成数据）

适用场景：不接真机，用合成数据验证算法逻辑与判别力

```bash
# 范围1：60天跑通 + 双轨信号隔离（层次A：链路对不对，20 项断言）
python scripts/validate_synthetic.py
python scripts/validate_synthetic.py --keep   # 保留 V001 数据供人工检查

# 范围2：10 场景判别力（层次B：设计好不好，含混淆项）
python scripts/validate_discriminative.py
python scripts/validate_discriminative.py --only TN_sleep_improved   # 只跑单个场景
```

> 两个脚本都固定了随机种子（数据用 crc32、GRU 用 `torch.manual_seed`），
> 连跑两次结果应完全一致。**若结果会飘，先修脚本再看结论**——
> 不可复现的绿色比红色更危险，它会让人停止追查（见 `docs/VALIDATION.md` 缺陷⑥）。

### 3. 运行测试

```bash
# 所有单元测试（14 个测试文件 / 332 用例）
python -m pytest

# 更详细输出
python -m pytest -v
```

---

## 数据格式

### 每日特征向量（features.csv）

> 顺序即 `SLEEP_FEATURES` / `SOCIAL_FEATURES`（`src/baseline/scaler_utils.py`）。

| 轨 | 特征名 | 来源 | 说明 |
|----|--------|------|------|
| 睡眠 | sleep_efficiency | 小贝壳 | 睡眠效率 [0, 1] |
| 睡眠 | waso_min | 小贝壳 | 入睡后觉醒总时长（分钟） |
| 睡眠 | sol_min | 小贝壳 | 入睡潜伏期（分钟） |
| 睡眠 | bed_exit_count | 小贝壳 | 夜间离床次数 |
| 睡眠 | deep_sleep_ratio | 小贝壳 | 深睡占比 [0, 1] |
| 睡眠 | sleep_onset_clock | 小贝壳 | 就寝时刻（自 20:00 起的分钟数，跨零点连续） |
| 睡眠 | night_hr_mean | 小贝壳 | 夜间平均心率（bpm，HRV 的弱代理） |
| 睡眠 | daytime_nap_min | 小贝壳 | 日间小睡时长（分钟） |
| 社交 | copresence_min | C6c（边缘人形检测） | 共处时长（分钟） |
| 社交 | out_of_home_min | T1C + C6c | 疑似外出时长（分钟，由室内无人推断） |
| 社交 | rar_amplitude | T1C + C6c | 昼夜节律振幅 [0, 1] |
| 社交 | rar_iv | T1C + C6c | 节律日内变异 |
| 社交 | activity_counts | T1C + C6c | 日间活动计数 |

### 每日推理结果（daily_inference/*.json）

**顶层按轨分块**，两轨各自一套分数与阈值（节选自真实输出）：

```json
{
  "elder_id": "E001",
  "day_key": "2026-08-29",
  "sleep": {
    "track": "sleep",
    "anomaly_score": 1.0276,
    "static_threshold": 2.2486,
    "ewma_threshold": 5.6056,
    "dynamic_threshold": 2.2486,
    "is_deviation": false,
    "signed_residuals": { "sleep_efficiency": 0.3879, "waso_min": -0.8794, "...": 0 },
    "abs_residuals":    { "sleep_efficiency": 0.3879, "waso_min":  0.8794, "...": 0 },
    "signed_z":         { "sleep_efficiency": 0.5258, "waso_min": -0.8236, "...": 0 },
    "signed_available": true,
    "ewma_pool": "default",
    "ewma_n": 53,
    "ewma_min_samples": 20,
    "ewma_frozen": false,
    "in_observation_period": false,
    "status": "success"
  },
  "social": {
    "track": "social",
    "anomaly_score": 0.9538,
    "dynamic_threshold": 2.3347,
    "is_deviation": false,
    "ewma_pool": "weekday",
    "status": "success"
  }
}
```

> 说明：推理层输出的是**加权残差异常分数（anomaly_score）与多档阈值**，而非直接预测值。
>
> - **三套残差**：`abs_residuals` 用于合成 `anomaly_score`（方向无关）；`signed_z` 保号，
>   供 `rules.py` 做方向性判定与 `judge.py` 的方向闸门。二者分工不能混——
>   用 abs 判方向会让"好转"和"恶化"无法区分（见 `docs/VALIDATION.md` 缺陷⑤）。
> - **`ewma_pool`**：社交轨分 `weekday`/`weekend` 两池（周末有子女探访效应）；睡眠轨单池 `default`。
> - **`ewma_frozen`**：当天是否因判偏离而冻结了基线更新（见 `docs/VALIDATION.md` 缺陷④）。
> - `status` 取值：`success` / `cold_start` / `cold_start_fallback`（GRU 未就绪，走滑动均值兜底）/
>   `data_insufficient` / `observation` / `error`。
> - 风险**等级**与风险**类型**由 `src/risk/judge.py` 在两轨推理结果之上单独判定，
>   **每轨各自算等级、取较高者**，绝不跨轨比较绝对分。

> 某自然日社交数据缺失时，聚合链路返回中性默认值并标记 `data_quality="missing"`，交由每日轨的校验/填充链路据实降级，而非用假的"正常值"喂进 GRU 掩盖真实偏离。

---

## 配置说明

### settings.yaml（全局配置）

```yaml
gru:
  # 按轨分块：两轨各自的维度与隐藏层
  sleep:
    feature_dim: 8         # 睡眠轨 8 维
    hidden_dim: 8          # 隐藏层维度（刻意取小，全模型仅 504 参数）
  social:
    feature_dim: 5         # 社交轨 5 维
    hidden_dim: 8          # 同上（405 参数）
  # 以下两轨共用
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
  cold_start_observation_days: 0  # 训练后观察期（已取消：建档期 35 天已让阈值稳定）
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
- 查看 `data/raw/{sleep,activity,camera}/{elder_id}/` 下是否有当天的 JSON
- 使用 `validator.py` 检查数据质量

### 问题2：某一轨报 `data_insufficient`，另一轨正常

双轨是**独立**推理的，一轨的数据缺失不影响另一轨出结果。各轨的数据来源：

| 轨 | 原始目录 | 提供的维 |
|----|---------|---------|
| 睡眠 | `data/raw/sleep/` | 全部 8 维 |
| 社交 | `data/raw/activity/` | `activity_counts` / `out_of_home_min` / `rar_amplitude` / `rar_iv`（后两个由 `hourly_activity` 算出） |
| 社交 | `data/raw/camera/` | 仅 `copresence_min` |

所以社交轨报错优先查 `activity/`（它撑起 4 个维中的全部节律特征）；
`camera/` 缺失只会让 `copresence_min` 变 missing。

### 问题3：验证脚本两次运行结果不一样

不应该发生——两个脚本都固定了随机种子（数据用 crc32、GRU 用 `torch.manual_seed`）。
若真的会飘，**先修脚本再看结论**：不可复现的绿色比红色更危险，
它会让人停止追查。本项目已被这件事咬过一次（见 `docs/VALIDATION.md` 缺陷⑥）。

### 问题4：等级升到 L2/L3 但风险类型为空

正常情况，不是 bug。等级看的是**总分持续偏离**，类型看的是**特定维的方向性组合**：
总分可能被某个高权重维单独拉高，而没有任何一条规则的必选维凑齐。
若确认应该报类型却报不出，查该规则必选维的 `signed_z` 是否都过了
`threshold_ratio`——变异系数大的维（如 `copresence_min`）z 分容易被压小。

### 问题5：连续偏离够天数了，等级却封在 L1

大概率是方向闸门生效：驱动升级的那段连续偏离**全是朝好方向**（睡得更好、出门更多）。
这是刻意设计，不是漏报——好转不该发风险提醒。

判断方法：`judge.py` 的返回值里每轨带 `adverse_direction` 字段（**注意它不落盘**，
`daily_inference/*.json` 里没有，需要在代码里取或加日志）。
若你认为该报，检查日志里该轨 `signed_z` 的符号是否与
`config/feature_weights.json` 中该维的 `direction` 一致。

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
**项目版本**：v2.1（在 v1.8 基础上：**单轨 6 维 → 双轨（睡眠 8 维 / 社交 5 维）**，两轨各有独立 GRU 与 scaler，从结构上切断跨类型"串味"；风险类型 2→3（睡眠稳定性 / 社会连接减弱 / 作息节律紊乱），规则改为"必选维 + 可选池"结构，`threshold_ratio` 按规则可配；`social_turns` 随音频链路删除，社会接触改由 C6c 边缘侧人形检测的 `copresence_min` 承担；`hrv_rmssd` 降级为 `night_hr_mean` 弱代理并降权；建档期 21→35 天、取消冷启动观察期；EWMA 增加**偏离日冻结**（上限 14 天），风险判定增加**方向闸门**，`validate_synthetic.py` 补固定 torch 种子——修复了三个一层压一层的缺陷：持续性异常被自适应基线掩盖、好转被判成"严重风险"、以及单维门槛对变异系数不同的维不等价（`copresence_min` 的 z 在门槛上抖动、社交崩塌报不出类型），详见 `docs/VALIDATION.md` 缺陷④⑤⑥。）

> v1.7 要点（在 v1.6 基础上）：①冷启动建档期 14→21 天并提为可配置项 `build_days`，附 20-seed 过拟合对比实验；②修复风险类型分类两处缺陷——"永不激活"（日志不写回 risk_types）与"正常日误激活"（类型判定挂靠 is_deviation）；③新增合成数据验证脚本 `validate_synthetic.py`（范围1跑通）与 `validate_discriminative.py`（范围2判别力，含真实噪声+混淆项）；④新增 `docs/TRAINING.md` GRU 训练详解。⑤修复模拟数据异常注入天数（25-30 → 40-46）与总天数（50 → 60），避开新建档期+观察期。测试 134 全绿。已知待办：跨类型"串味"（共享GRU溢出）、真实传感器采集/推送仍为桩。文档：本 README（总览+现状）、`docs/TRAINING.md`（训练详解）、`docs/TODO.md`、`docs/VALIDATION.md`（三层验证+§7执行记录）

> v1.6 要点：打通统一系统链路（统一调度器真正调用每日管道）；声学按自然日聚合 + 缺失标记；微调与冷启动共用 early-stopping；统一系统快照原子写；save_interval 走配置。

> v1.5 要点：单人系统；实时轨仅采集不报警；GRU 加健康门禁 / 冷启动兜底 / early-stopping / 微调防污染；判定链路修复（前向填充生效 / 观察期按训练后推理计数 / 严重级加幅度门槛）；移除时间编码死代码，特征统一为 10 维 `FEATURE_NAMES`。
