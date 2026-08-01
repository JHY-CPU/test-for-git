# 心理健康连续感知与趋势预警系统

> 面向**单个**居家老人的日常心理健康**趋势预警**系统。通过"自己和自己比"的个人化基线方法检测心理状态偏离，**不进行临床诊断**。

## 系统特点

- **单人系统**：面向单一老人的连续监测与趋势预警
- **双轨每日批处理**：每日读取累积的传感器数据（`data/raw/`），聚合为双轨特征（睡眠 8 维 / 社交 5 维）后**各自独立**做趋势推理与预警，无独立实时运行时
- **克制预警**：预警仅在连续多日偏离后发出，避免单日波动过度敏感打扰家人
- **个人化基线**：为该老人独立建模（PersonalBaselineGRU + EWMA动态阈值），"自己和自己比"
- **多传感器融合**：小贝壳无感睡眠监测仪 + 萤石 T1C 人体移动传感器 + C6c 摄像机（边缘侧人形检测）
- **三线并行监测**：睡眠障碍 / 孤独（GRU 个人基线，两条独立轨） + 抑郁（MPDD 群体基线，旁路只读通道）
- **趋势判定**：连续3-5天偏离才触发预警，避免单日波动误报
- **文件驱动数据流**：各传感器适配器把原始数据落盘到 `data/raw/{sleep,activity,social}/`，每日管道从此累积读取

---

## 项目现状（v2.1 双轨）

> 一张表看清"哪些已扎实、哪些还在路上"，避免把"代码跑通"误读成"临床有效"。

**核心算法层已完工并验证**：500 个单元测试全部通过；范围 1（链路正确性）20/20、范围 2（判别力）9/9。每日趋势管道（批处理：读取 `data/raw/` → 双轨特征聚合（睡眠 8 维 / 社交 5 维）→ 每轨独立 GRU 残差推理 → EWMA（偏离日冻结）→ 连续偏离判定 → 风险判定（含方向闸门，每轨各自算等级取较高者）→ 预警 → 周报）已端到端打通，不依赖任何实时运行时。

> ⚠️ **实现成熟度：算法是真的，硬件对接与外部服务大多还是"桩/模拟"。** 这是科研原型阶段的正常状态，但必须讲清楚，避免误以为"能上真机"：

| 部分 | 状态 | 说明 |
|------|------|------|
| GRU 基线 / EWMA / 风险判定 / 数据管道 | ✅ 真实可用 | 有算法、有测试，是系统的核心 |
| 传感器**真实采集**（小贝壳 / 萤石 C6c+T1C 的 `_read_raw`） | ⚠️ **未实现（桩）** | 均抛 `NotImplementedError`，只有 mock/模拟数据能跑 |
| 预警**推送**（子女App/短信/网格员） | ⚠️ 模拟 | `alert.py` 按 `settings.yaml` 的 `alert` 段决定动作，**按事件去重**（见下方「预警按事件去重」），但只写日志字符串，未接真实推送服务 |
| 周报 LLM | ✅ 可用（可选依赖） | 模型与 token 上限从 `report.model` / `report.max_tokens` 读；`anthropic` 未安装时自动回落规则模板 |

**一句话**：现在能端到端跑通、能验证算法逻辑，靠的是**模拟/文件数据**；接真实设备与推送服务是后续工作。

### 已知局限（如实记录，非阻断）

| # | 局限 | 影响 | 现状 |
|---|------|------|------|
| ① | ~~**`alert.py` 无冷却/去重/抑制**~~ | 曾经：永久性变化**无限期每日重复报同一等级**，`PERM_step` 实测 91 天 91 次；E001 60 天 12 次推送对应 4 个真实事件 | ✅ **已修（2026-07-31）**：换成事件模型，91→4、12→4。见「预警按事件去重」与 VALIDATION §10 |
| ② | 残差统计未按周内分池 | `copresence_min`（周末 ×2.4）等有周末效应的维，残差 std 被双峰分布抬高、z 分被系统性压小 | 以 `social_decline` 的 `threshold_ratio=1.0` 局部补偿；根治需远长于 7 天的留出段 |
| ③ | GRU 固定 7 天窗对持续性变化钝感 | 异常持续到第 3 天后输入窗被异常日填满、预测跟着漂移，残差收缩 | EWMA 层已用偏离日冻结缓解，模型层无对应机制；属固定窗口预测器的固有取舍 |
| ④ | `KM_social_partial` 必然漏报 | 只共处↓、外出照常（子女不来但自己照常出门）不触发 | "三项全中"规则换取低误报的代价，已计入验证记录 |

### 本轮改动（2026-07-31 代码走查）

这一轮**不改任何算法思路**，只把已有设计真正接通、并把断言挪到正确的层。
9 条缺陷的共同特征是：**验证脚本与单测都断言在错的层**，所以它们能长期活在
"312 全绿 + 19/19 + 9/9"之下。完整实证见 `docs/VALIDATION.md` §8。

| 修掉的 | 修前实测 |
|--------|---------|
| 冷启动兜底进不了判定层 | 建档期连续 6 天严重异常，`risk_level` **全 0**，一条预警都发不出 |
| `data_quality` 从未落进推理日志 | "降级日跳过持续性统计"这条**写进项目约定的不变量在生产链路里从未生效**（60/60 日志无该字段） |
| 幅度门槛被 7 天窗稀释、且与量纲脱节 | 连续 3 天各 1.45（都越过各自阈值）被 4 个正常天拉到 0.907，只报 L1 |
| 持续性统计不看日历 | 5 条日志跨 22 个日历日（断 17 天）仍数出 `consecutive=5` → L3 |
| 7 天日志窗与 7 天门槛互卡 | circadian 降级模式只要历史含 1 个降级日就永远不可激活 |
| 三处配置段无人读取 | `alert` 整段、`report.*` 均未生效；`ewma.max_freeze_days` 配置里根本没这个键 |
| `requirements.txt` 装不上 | `pyaudio` 缺 portaudio 头文件 → pip 整体中止 → torch 一个都装不上 |
| 周报异常分恒 0.00、周期错配 | 读顶层 `anomaly_score`（双轨后已不存在）；本周无数据时静默拿别的周顶替 |
| 周轨没有入口、MPDD 契约从未产出 | `run_weekly_pipeline` 零调用方；契约定义了测了但没有链路写过 |

> **这一轮最该记住的不是任何一条具体缺陷**，而是：**全绿不等于没问题，要看绿的是什么。**
> 当时三条"通过"的断言分别测了 status 流转、validator 的辅助函数、和手搓的假数据，
> 唯独没测真实链路的输出。

系统验证分三层，证据强度依次递增（完整方案见 `docs/VALIDATION.md`）：

| 层次 | 回答的问题 | 状态 | 说明 |
|------|-----------|------|------|
| **A 代码对不对** | 算法有没有被正确实现 | ✅ 已完成 | 500 单测（23 文件）+ 端到端冒烟（`validate_synthetic.py` 20/20，含双轨信号隔离），每日管道链路已打通 |
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

# 6. 运行测试（23 个测试文件 / 500 用例）
python -m pytest
```

---

## 项目结构

```
mental-health-sense/
├── config/                          # 配置文件
│   ├── settings.yaml                # 全局配置（GRU、训练、EWMA、风险阈值、抑郁通道）
│   └── feature_weights.json         # 特征权重（加权残差计算）
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
│   │   ├── alert_state/             # 预警事件状态（按 elder_id，跨天持久化）
│   │   ├── mpdd_evidence/           # GRU → MPDD 的单向证据契约（JSON）
│   │   ├── depression/              # MPDD → 展示层的抑郁评估契约（JSON，反方向）
│   │   └── weekly_reports/          # 周报（LLM 或规则模板）
│   ├── depression/                  # 抑郁通道的输入
│   │   └── E001/description.txt     # 个人介绍（英文，冻结；改动会改变分数）
│   └── elder_configs.json           # 老人元数据（生成产物，非输入）
│
├── src/                             # 源代码
│   ├── baseline/                    # GRU个人基线模型
│   │   ├── gru_model.py             # PersonalBaselineGRU（7→1天预测）
│   │   ├── trainer.py               # 冷启动（健康门禁+early-stopping）+ 每周微调（剔除偏离天+模型备份）
│   │   ├── inference.py             # 每日推理引擎
│   │   ├── data_health.py           # 训练数据健康门禁（MAD离群筛查，防GRU"学坏"）
│   │   ├── cold_start_fallback.py   # 冷启动兜底（GRU就绪前用中位数/MAD稳健基线检测）
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
│   ├── depression/                  # ★ 抑郁评估旁路（MPDD 群体基线，不参与个人基线）
│   │   ├── status.py                # 状态枚举 + 可展示白名单（单一事实来源）
│   │   ├── contract.py              # 契约 schema、组装与校验
│   │   ├── aggregate.py             # 多片段 → 单次结论（中位数 + 离散度）
│   │   ├── store.py                 # 读写 + 有效期（展示层唯一入口，零重依赖）
│   │   ├── clip_source.py           # ClipSource 抽象：现在传录像 / 将来接实时流
│   │   ├── runner.py                # 编排：取片段 → 逐段推理 → 聚合 → 落盘
│   │   └── mpdd_process.py          # subprocess 封装（环境钉死 + 超时 + 退出码）
│   │
│   ├── risk/                        # 风险判定层
│   │   ├── rules.py                 # 3类风险类型（睡眠稳定性/社会连接/作息节律）
│   │   ├── judge.py                 # 4级风险判定（每轨各自算等级取较高者 + 方向闸门）
│   │   ├── alert.py                 # 预警推送（按事件去重；推送本身仍只写日志字符串）
│   │   └── alert_state.py           # 事件状态机（开始/升级/新类型/缓解 才通知）
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
│       ├── io.py                    # 文件读写、路径管理、原子写
│       ├── status.py                # 推理状态枚举与"可评估"判据（单一事实来源）
│       ├── continuity.py            # 日历回溯（判级与判型共用，两侧曾不一致）
│       ├── seeding.py               # 确定性种子派生（crc32，绝不用内置 hash()）
│       ├── logger.py                # 日志配置
│       └── metrics.py               # 评估指标
│
├── scripts/                         # 可执行脚本
│   ├── generate_simulation_data.py  # 生成60天双轨模拟数据（睡眠异常40-46 / 社交异常50-58，错开）
│   ├── train_all_baselines.py       # 双轨建档（两轨各一个 GRU/scaler/残差统计/EWMA）
│   ├── run_daily_pipeline.py        # 手动触发每日推理
│   ├── run_weekly_pipeline.py       # 手动触发周轨（微调 + 周报）
│   ├── run_depression_assessment.py # 抑郁评估（事件驱动，不挂日管道）
│   ├── ack_alert.py                 # 确认预警事件，转静默追踪
│   ├── validate_synthetic.py        # 【范围1】合成数据端到端跑通验证（60天，层次A）
│   └── validate_discriminative.py   # 【范围2】合成数据判别力验证（10场景+混淆项，层次B）
│
├── tests/                           # 单元测试（23 个测试文件 / 500 用例，确定性可复现）
│   ├── conftest.py                  # 共享 fixture
│   ├── test_aggregator.py           # 数据聚合测试
│   ├── test_data_health.py          # 训练数据健康门禁（MAD 离群筛查）测试
│   ├── test_ewma.py                 # EWMA基线测试
│   ├── test_gru_model.py            # GRU模型测试
│   ├── test_imputer.py              # 缺失值处理（前向填充）测试
│   ├── test_validator.py            # 数据质量校验测试
│   ├── test_regression_2026_07_31.py # ★ 第二轮走查缺陷回归（全部打在真实链路输出上）
│   ├── test_depression_*.py         # 抑郁通道：状态/契约/聚合/存储/隔离守卫/编排
│   ├── test_risk_judge.py           # 风险判定测试（含低/高幅度连续偏离回归）
│   ├── test_risk_rules.py           # 风险规则测试
│   ├── test_alert.py                # 预警动作映射测试
│   ├── test_alert_events.py         # ★ 事件模型测试（含「升级穿透抑制」守门员）
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
| **每日趋势轨** | 每日03:00 | 读取累积原始数据 → 双轨特征聚合 → 每轨独立趋势分析 → 预警出口 | 每轨 GRU 预测 + 该轨 EWMA 动态阈值 |
| **周报轨** | 周日04:00 | 双轨微调 + 汇总一周趋势生成周报 | LLM（fallback: 规则模板） |

> 注意"双轨"在本文档有两个不同含义，别混：这张表说的是**调度轨**（每日 / 每周）；
> 下文的**睡眠轨 / 社交轨**说的是特征与模型的拆分。二者互相独立。

**核心理念**：系统是**每日批处理管道**，没有独立的实时采集运行时。各传感器适配器把原始数据落盘到 `data/raw/{sleep,activity,camera}/`，每日趋势轨在累积数据上做判定后统一发出预警。特征与模型按信号来源拆成**睡眠轨（8 维）/ 社交轨（5 维）**，两轨各有独立的 GRU、scaler、残差统计与 EWMA，互不共享——这是切断跨类型"串味"的关键。所有心理风险预警仅在**连续多日偏离**后触发，避免因单日波动过度敏感、频繁打扰家人。

### 数据流架构

```
【每日趋势轨：批处理，每天凌晨03:00触发】

data/raw/sleep/{date}.json      ─┐   小贝壳无感睡眠监测仪
data/raw/activity/{date}.json   ─┼─► 萤石 T1C 人体移动传感器
data/raw/camera/{date}.json     ─┘   萤石 C6c（边缘人形检测，不出图/不采音）
                                          ↓ 按自然日聚合（无数据则标 missing）
                                      缺失填充 / 数据质量校验
                              （四态 valid/degraded/insufficient/offline
                                → 按轨随推理结果落盘，供持续性统计判断这天算不算数）
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
                 （按自然日回溯，缺日/降级日跳过但连续跳过 >3 天即打断）
                          └───────────────┬───────────────┘
                                          ↓
                    风险类型分类（读 signed_z，方向性判定）
                    睡眠稳定性 / 社会连接减弱 / 作息节律紊乱
                                          ↓
                    Level 0/1/2/3 判定（每轨各自算，取较高者）
                    + 幅度门槛：severity = 分数/自身阈值，只算在这段 streak 上
                    + 方向闸门：升级到 L2/L3 需有"朝坏方向"的证据
                                          ↓
                    ┌─────────────────────┴─────────────────────┐
                    ↓                                           ↓
            预警推送（统一出口）                  MPDD-AVP 单向证据契约
            L1 日志 / L2 子女 / L3 +网格员         mpdd_evidence/*.json
            按事件去重（见「预警按事件去重」）      （旁路，失败不影响主链路）
```

> **两轨绝不跨轨比较绝对分**：8 维与 5 维的权重和不同、残差尺度不同，
> 取最大或求平均都没有统计意义，只能各自与自己的阈值比、再比较等级。

### 四级风险（全部基于趋势）

| 等级 | 名称 | 触发条件 | 响应措施 |
|------|------|----------|----------|
| 0 | 正常 | 无偏离 | 无 |
| 1 | 关注 | 单日偏离、间歇偏离，或 7 天窗内有单日高峰（`max_severity > high_spike_severity`） | 记录日志 |
| 2 | 提醒 | 连续3天偏离 **且** 该段平均幅度超标（`avg_severity > sustained_severity`）**且**过方向闸门 | 推送子女App |
| 3 | 严重 | 连续5天偏离 **且** 平均幅度超标（同提醒级幅度门槛）**且**过方向闸门 | 短信 + 社区网格员 + 强制响铃 |

> 等级按**每轨各自计算、取较高者**（`judge.py`），不跨轨比较绝对分。
> "连续 N 天"按**自然日**数，不按日志记录条数（见下文"持续性统计的日历语义"）。

> **严重级为何也要幅度门槛？** Level 3 会触发社区网格员介入 + 强提醒，代价高。若仅凭连续天数升级，长达数天但每天只"擦线"越过动态阈值的低幅度偏离也会直冲最高级，与提醒级（带幅度门槛）判定不一致，且过度打扰家人和社区。因此严重级与提醒级共用同一条幅度门槛。

> **幅度门槛只算在"驱动本次升级的那段连续偏离"上**，不是整个 7 天窗。用整窗均值会被
> 窗口里的正常天稀释：实测连续 3 天各 1.45（三天都实打实越过了各自的动态阈值）
> 被前面 4 个正常天拉到 0.907，卡在门槛下只报 L1。方向闸门早已只看这段 streak，
> 两个门槛必须看同一段数据。详见 `docs/VALIDATION.md` §8.3。

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
| 幅度门槛（severity） | 光有天数不够，那段偏离的平均幅度还要高出自身阈值 15% 以上，挡住"连续多天擦线" |
| 方向闸门 | L2/L3 需有"朝坏方向"的正面证据，防止全面好转被报成严重风险 |
| 降级日/缺日跳过 | 传感器抖一下既不清零已攒的偏离段，也不凭空算成偏离；但连续跳过 > 3 天就打断 |
| 日历连续性校验 | "连续 N 天"按自然日数，设备离线前后的两段偏离不会被拼成一段 |
| ~~冷启动观察期~~（**已取消**） | 曾为"训练后 7 次推理仅记录不报警"。建档期延长到 35 天后，EWMA 初始已有 28 个样本、动态阈值直接稳定，观察期不再必要——留着只是白白多 7 天不报警的盲区。配置项 `cold_start_observation_days` 保留但置 0，代码路径仍在（真机数据若发现阈值预热更慢，可随时调回） |
| 每日批处理统一出口 | 预警仅在累积数据上连续偏离后发出，不做单日实时报警，避免单点波动打扰家人 |

### 持续性统计的日历语义

"连续 N 天"这句话看着简单，实现上有三个坑，都踩过：

1. **按记录条数数 ≠ 按自然日数**。`load_daily_results` 取的是最近 N 个**文件**；
   而两轨全不可用那天 `daily_job` 直接跳过推理、**连日志都不生成**。两者叠加，
   设备离线一段时间后断裂两端的偏离日会被当成连续日拼起来——实测 5 条日志
   跨越 22 个日历日（中间断 17 天）仍数出 `consecutive=5` → L3。
   现在统一按自然日逐日回溯（`rules._walk_back`）。

2. **"跳过"必须有上限**。缺日与降级日走同一条路径（都是"这天我们不知道"），
   既不累加也不打断；但连续跳过超过 `risk.continuity.max_skip_days`（默认 3）
   就打断连续段——否则一次长时间离线又会把两段无关的偏离粘起来。

3. **加载窗口要够宽**。`circadian_disruption` 的降级模式要求连续 7 天达标，
   若只加载 7 条日志（今天 + 6 天历史），历史里有一个降级日被跳过，计数上限
   就掉到 6，规则在**数学上永远不可激活**。而降级模式正是因为睡眠轨不可用才
   进入的，恰恰是最容易伴随数据质量问题的场景。现在加载天数由
   `rules.required_history_days()` 从各规则门槛推导（当前 25 天）。

> 等级判定窗口仍固定为最近 **7 个自然日**——"最近状况有多严重"这个问题只该看最近一周；
> 只有风险类型的持续性统计才回溯更远。

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

### 3. 冷启动兜底（消除建档期盲区）

**问题**：GRU 需要 35 天建档期才可靠，此前系统若完全不检测，等于头五周是盲区。

**做法**（`src/baseline/cold_start_fallback.py`）：GRU 基线就绪前，用**中位数 / MAD 稳健基线**
的加权 z-score 做基础离群检测。`daily_job` 在 `daily_inference` 返回 `cold_start` 时自动切换到
兜底检测。sigma 默认 **3.0**，比 GRU 轨更保守——建档期滑动基线本身不稳，宁可漏报也不要
一开始就误报动摇信任。GRU 一就绪，兜底自动停用。

改用中位数/MAD 而非均值/标准差：建档期只有十几个样本，混入 1-2 个异常天就会把
均值和标准差拽走（异常天撑大 std → 后续真异常反而检不出）。

> ⚠️ **这一层曾经形同虚设，值得单独说清楚。**
>
> 兜底确实算出了偏离、给了带方向的 `signed_z`，但 `judge` 与 `rules` 的状态白名单
> 只收 `("success","observation")`，**兜底轨的结果整个被丢掉**。实测建档期注入
> 连续 6 天严重睡眠恶化（`anomaly_score` 166→58，阈值 3.0，`is_deviation` 6/6），
> `risk_level` **全是 0**，一条预警都发不出——盲区在检测层补上了，预警层原封不动。
>
> 现在 `cold_start_fallback` 与 `success`/`observation` 一起由 `src/utils/status.py`
> 统一定义为"可评估状态"。**加新状态时只改那一个文件**：此前白名单以字面量散在
> 五处，正是同一个缺陷反复出现的温床。同一场景修复后的升级曲线是
> `L1 → L1 → L2 → L2 → L3 → L3`。
>
> 注意接线不是简单加白名单：兜底的分是稳健 |z|（阈值 3.0，正常天可达 1.8），
> 与 GRU 轨的残差尺度不可比，所以幅度门槛必须同时改成 severity 相对量——
> 否则建档期会天天误报 L1。详见 `docs/VALIDATION.md` §8.2 / §8.3。

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
> 社会接触改由 C6c 边缘侧人形检测的 `copresence_min` 承担。代价是拿不到“是否在交谈”，
> 只能知道“是否有人同处一室”。
>
> **⚠️ 隐私边界已于 2026-07-31 重新划定，务必区分两件事：**
>
> - **GRU 管线仍然完全不使用音频**。社交轨的 5 维一维未变，`copresence_min`
>   只来自边缘侧人形检测的逐秒人数，视频不出户、只上报当日聚合分钟数。
> - **设备会录制短音视频片段**，供独立的抑郁评估模块（MPDD 旁路通道）使用。
>   片段落盘在 `data/raw/video_clips/`，只在检测到“有正脸 + 有连续语音”时导出。
>
> 这两条在知情同意书上是**不同条目**，不能合并成一句“我们现在录音了”。

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

# 周轨：双轨微调 + 周报
python scripts/run_weekly_pipeline.py
python scripts/run_weekly_pipeline.py --no-retrain   # 只出周报，不动基线（排查周报时用）
```

**每日推理会产出三样东西**：

| 产物 | 路径 | 说明 |
|------|------|------|
| 特征行 | `data/features/{id}/features_{track}.csv` | 同一 `day_key` 重复写入时**幂等覆盖**，补算不会产生重复行 |
| 推理日志 | `data/logs/daily_inference/{id}_{date}.json` | 双轨分数/阈值/残差 + 数据质量 + 写回的 `risk_level` |
| 证据契约 | `data/logs/mpdd_evidence/{id}_{date}.json` | 给外部抑郁模型的单向输入（旁路） |

> **补算与重跑是安全的**，三层各有各的挡法：
>
> - **特征行**：同 `day_key` 幂等覆盖，不产生重复行。
> - **EWMA**：按 `day_key` 去重（`day_key <= last_day_key` 直接拒收），重跑同一天
>   不会把分喂进基线两次、也不会让冻结计数多加一次。乱序补算历史日同样被挡住——
>   把早已过去的分当成最新观测喂进指数加权，权重完全错位。
> - **阈值**：重跑复用首跑落盘的三个阈值，保证 `is_deviation` 判定幂等；但只在
>   **同一条产出路径**之间复用（兜底轨与 GRU 轨量纲不可比，见
>   `inference._load_prior_thresholds`）。
> - **预警事件**：通道游标 `last_processed_day` 跨事件保留，**早于它的历史日
>   一律不动状态、不通知**。这一条 2026-08-02 才补上——此前事件一缓解，
>   `apply()` 就把游标清空，补算一个历史高危日会真的推送给子女
>   （见 `docs/VALIDATION.md` §11.1）。

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
# 所有单元测试（23 个测试文件 / 500 用例）
python -m pytest

# 更详细输出
python -m pytest -v
```

> ⚠️ 若本机装了 ROS（`/opt/ros/*` 进了系统 `PYTHONPATH`），pytest 会在**收集阶段**
> 把它导进来并崩在 `ModuleNotFoundError: lark`，一条用例都跑不到。
> 用 `env -u PYTHONPATH python -m pytest` 绕开。同一个污染源也要求 MPDD 子进程
> 剔除系统 `PYTHONPATH`（见 `src/depression/mpdd_process.py`）。

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
    "data_quality": "valid",
    "status": "success"
  },
  "social": {
    "track": "social",
    "anomaly_score": 0.9538,
    "dynamic_threshold": 2.3347,
    "is_deviation": false,
    "ewma_pool": "weekday",
    "data_quality": "degraded",
    "status": "success"
  },
  "track_quality": { "sleep": "valid", "social": "degraded" },
  "is_deviation": false,
  "consecutive_deviation_days": 0,
  "risk_type_qualifies": { "sleep_stability": false, "social_decline": false,
                           "circadian_disruption": false },
  "risk_level": 0
}
```

> 说明：推理层输出的是**加权残差异常分数（anomaly_score）与多档阈值**，而非直接预测值。
>
> - **三套残差**：`abs_residuals` 用于合成 `anomaly_score`（方向无关）；`signed_z` 保号，
>   供 `rules.py` 做方向性判定与 `judge.py` 的方向闸门。二者分工不能混——
>   用 abs 判方向会让"好转"和"恶化"无法区分（见 `docs/VALIDATION.md` 缺陷⑤）。
> - **`ewma_pool`**：社交轨分 `weekday`/`weekend` 两池（周末有子女探访效应）；睡眠轨单池 `default`。
> - **`ewma_frozen`**：当天是否因判偏离而冻结了基线更新（见 `docs/VALIDATION.md` 缺陷④）。
> - **`data_quality`（按轨）**：该轨当天的数据质量，`valid` / `degraded` / `insufficient` / `offline`。
>   **持续性统计靠它决定这天算不算数**，所以必须随推理结果一起落盘。
>   曾经它只写进 `features_{track}.csv`、没进推理日志（实测 60/60 个日志都没有该字段），
>   于是"降级日既不累加也不打断"这条设计在生产链路里从未生效——读不到字段就一律
>   按 valid 计入，传感器抖动照样能攒成预警。判定按**轨**读：用单一顶层值会让睡眠轨
>   降级压住社交轨的计数，等于在判定层把双轨的故障隔离又粘回去。
> - `status` 取值：`success` / `observation` / `cold_start_fallback`（GRU 未就绪，走稳健滑动基线兜底）
>   —— 这三个是**可评估状态**，能参与判级判型；`cold_start` / `data_insufficient` 不参与。
>   判据集中在 `src/utils/status.py`，不要在别处再写字面量白名单。
> - **`risk_type_qualifies` / `risk_level`** 由 `quick_judge` 判定后**写回今天这个文件**，
>   次日的持续性统计才读得到。破坏这条写回链会让风险类型永远激活不了。
> - 风险**等级**与风险**类型**由 `src/risk/judge.py` 在两轨推理结果之上单独判定，
>   **每轨各自算等级、取较高者**，绝不跨轨比较绝对分。

> 某自然日社交数据缺失时，聚合链路返回中性默认值并标记 `data_quality="missing"`，交由每日轨的校验/填充链路据实降级，而非用假的"正常值"喂进 GRU 掩盖真实偏离。

### MPDD-AVP 单向证据契约（mpdd_evidence/*.json）

每日判定后额外产出一份给外部抑郁模型（MPDD-AVP）的证据块：

```json
{
  "schema_version": "2.1.0",
  "elder_id": "E001", "day_key": "2026-08-29", "timezone": "Asia/Shanghai",
  "sleep_evidence":     { "anomaly_score": 1.03, "is_deviation": false,
                          "consecutive_days": 0, "signed_z": {...}, "quality": "valid" },
  "circadian_evidence": { "signed_z": {"rar_amplitude": ..., "rar_iv": ...},
                          "is_disrupted": false, "quality": "valid" },
  "social_evidence":    { "anomaly_score": 0.95, "is_deviation": false,
                          "consecutive_days": 0, "signed_z": {...}, "quality": "valid" },
  "risk_level": 0,
  "note": "单向契约：GRU → MPDD-AVP。本系统不消费 MPDD-AVP 输出。"
}
```

**契约约束（两侧都要遵守）**：

- MPDD-AVP 可把这些块当**先验/辅助特征**，但不得让本系统的偏离分直接决定抑郁标签；
- 本系统**不消费** MPDD-AVP 的任何输出——防止个人基线被抑郁判定反向污染。
  这与"每周微调只用 `is_deviation=False` 的正常天"是同一条防污染原则；
- `signed_z` 的符号约定是 `observed − predicted`：负值 = 低于个人基线；
- `quality` 为 `cold_start` 表示那天走的是兜底基线，证据比 `valid` 弱但**不是** `missing`。

> 产出失败只记错误日志，不影响已算好的风险判定——证据契约是旁路输出。

---

## 预警按事件去重

**一段连续的 L2/L3 是一个事件，不是 N 个。** 实现在 `src/risk/alert_state.py`。

### 为什么必须这么做

`alert.py` 原本对每个 `risk_level >= 1` 的日子无条件发一次通知，没有任何记忆。
实测：

| | 修复前 | 修复后 |
|---|---|---|
| E001 60 天 L2+ 推送 | 12 次 | **4 次** |
| E001 60 天 L3 强制响铃 | 8 次 | **2 次** |
| `PERM_step` 91 天 L3（永久性衰退） | **91 次** | **4 次** |

发出的 4 次恰好对应 4 个真实事件。

**后果不是"吵"，是系统最终变哑**：家属关掉通知后，真正的**新**变化也收不到了。
方向闸门、幅度门槛、持续性统计——前面所有为压低误报做的工作，
在通知被关掉的那一刻全部失去出口。

**检测层是对的**，缺口在预警策略层：把“状态持续”当成了“每天都是新事件”。
所以这次修复没有碰 `judge.py` / `rules.py` 一行。

### 通知时机

| 时刻 | 发不发 | transition |
|---|---|---|
| 首次升到 L2 | **发** | `started` |
| 同级持续 | 不发（进周报回执） | `cooldown` |
| L2 → L3 | **发** ★ 穿透一切抑制 | `escalated` |
| 出现新的风险类型 | **发**（受 3 天短冷却） | `new_type` |
| L3 → L2（仍有风险） | 不发 | `improved` |
| 回到 L0 | **发**（可配） | `resolved` |
| 已人工确认 | 不发 | `acknowledged` |
| 超过 `repeat_days` | **发** | `repeat` |

### ★ 升级必须穿透一切抑制

纯粹按“同一等级 X 天内最多一次”做冷却会出现：

```
D1  L2 → 发通知，进入冷却
D3  L2 → 冷却中，不发   ✓ 对
D5  L3 → 冷却中，不发   ✗ 恶化被冷却窗吃掉
```

**这是拿误报换漏报，比完全没有冷却更危险。** 所以升级判定排在**所有**抑制
判据之前（含冷却、已确认、同日幂等）。`tests/test_alert_events.py` 的
`TestEscalationBypassesEverything` 是这条防线的专职守门员。

### 两条独立的事件流

```
channel="baseline"     GRU 双轨（睡眠障碍 / 孤独）
channel="depression"   MPDD 群体基线（抑郁）
```

各自计数、各自冷却、**互不压制**——理由同“绝不跨轨比较绝对分”：
它们不是同一件事，不该共用一个计数器。基线在冷却中，不该顺带静默抑郁通道。

### 事件边界

复用 `risk.continuity.max_skip_days`（默认 3）：与上次可信观测断开超过它，
就算新事件而非续接——设备离线一段时间后，不该把两段无关的风险期粘成一个。
与 `continuity.walk_back_days` 的断段语义保持一致。

### 状态持久化

`data/logs/alert_state/{elder_id}.json`，原子写。

必须落盘的理由与 EWMA 的 `freeze_streak` 完全相同：**这是每日批处理，
进程每天起一次就退，内存态活不过今天**。容错沿用 `ewma` 的四条：
状态另存小文件 / 原子写 / 缺文件按默认起算 / 损坏则重置该通道并继续
（不抛、不连坐另一条流）。

### 人工确认

```bash
python scripts/ack_alert.py --elder E001 --show     # 看当前事件
python scripts/ack_alert.py --elder E001            # 转静默追踪
python scripts/ack_alert.py --elder E001 --undo     # 撤销
```

确认后同级持续不再通知，但**升级与新风险类型仍会破静默**。
语义是“我知道了，别再提醒我这件事”，而不是“关掉这个老人的所有告警”。

> 真实推送通道接入后，App 侧的“已知晓”按钮回写的就是这同一个状态，
> 本脚本是它的手工等价物。

### 周报回执（配套的告知义务）

通知层静默后，周报若也一字不提，“系统安静”与“系统挂了”就无法区分——
与“长期降级和长期正常必须可区分”是同一条原则。所以持续中的事件每周在周报里
露一次面，低强度、不推送、不响铃，并明确写出“静默不代表已缓解”：

```markdown
## 预警回执

- **睡眠 / 社会连接**：睡眠稳定性偏离、社会连接减弱 持续中，第 16 天（2026-08-08 起）
  - 已通知 5 次，最近一次 2026-08-23

> 以下事件仍在持续。系统按**事件**去重，同一等级不重复推送，
> **静默不代表已缓解**。等级上升或出现新的风险类型时会立即通知。
```

`_empty_report` 也带上——无数据周恰恰最需要它：设备掉线导致本周没有判定，
但上一个事件可能还开着，静默 + 空周报 = 两层遮蔽。

---

## 抑郁评估旁路通道（depression/*.json）

第三条监测线。**与两条 GRU 轨完全独立**，实现在 `src/depression/`。

### 为什么是旁路而不是第三条轨

| | GRU 睡眠轨 / 社交轨 | MPDD 抑郁通道 |
|---|---|---|
| 基线类型 | **个人相对**（跟你自己平时比） | **群体绝对**（跟别人比） |
| 节律 | 每日批处理 | 事件驱动（有合格录像才评估） |
| 判据 | 残差 / EWMA / 连续天数 / 方向闸门 | 单次多模态分类 + PHQ-9 回归 |
| 产出 | risk_level 0~3 | 等级 + PHQ-9 估计 |

两者**尺度不可比**，这与本项目"绝不跨轨比较绝对分"是同一条不变量——
睡眠轨和社交轨这两条自家的轨都不敢比（维度数与权重和不同），何况训练目标
完全不同的外部模型。

更要紧的是**防正反馈**。若抑郁分能标记异常日，会形成自我强化的闭环：

```
抑郁判高 → 该日标为异常 → 异常日被排除出每周微调
        → 个人基线越缩越窄 → 更容易判偏离 → 偏离又被拿去佐证抑郁 → …
```

每一步单独看都"合理"，合起来是系统自己证明自己。这与"系统会习惯异常"
（VALIDATION 缺陷④⑤⑥）是同一族病的镜像版本。

### 四条硬约束（由 `tests/test_depression_isolation.py` 静态守卫）

1. `src/baseline/`、`src/risk/`、`src/data_pipeline/`、`src/scheduler/` **不得导入** `src.depression`
2. 不写 `features_*.csv`、不碰 `residual_stats` / `ewma` / `weekly_retrain`
3. 输出不进入 `judge_risk_level` 的 `risk_level` 与 `per_track`
4. MPDD 的依赖（transformers / librosa / av / cv2）**不进 `requirements.txt`**

第 4 条靠**进程边界**实现：`runner.py` 用 `depression.python_bin` 指定的独立
解释器 subprocess 调用 MPDD 仓。一旦那些依赖进了主链路的 import 图，
任何一个 import 失败都会让睡眠和社交监测一起挂——这与"单轨失败是正常降级、
两轨同时不可用才算整体失败"是同一条原则。

**验收标准**：把整个 MPDD 仓改名，`python -m pytest` 必须全过、
`run_daily_pipeline` 必须正常出结果。

### 分层（决定了谁能 import 谁）

```
status / contract / aggregate / store   零重依赖，只用标准库
clip_source / runner / mpdd_process     碰 subprocess 与外部仓
```

`weekly_report.py` **只准 import `store`**。MPDD 环境挂掉时周报照常渲染，
那一节显示"暂无最新评估"。

### 用法

```bash
# 前置：个人介绍文本（英文，一旦确定必须冻结）
#   data/depression/{elder_id}/description.txt

# 设备到货前：手工传录像
python scripts/run_depression_assessment.py --elder E001 --date 2026-08-12 \
    --video /path/to/footage.mp4
```

链路：`extract_clips_from_surveillance.py`（VAD + OpenFace 筛片段）
→ 逐段 `infer_elder_depression.py` → 中位数聚合 → 落契约 → 周报展示。

> **刻意不用 MPDD 的 `--run_infer`**：它内部是 `subprocess.run(..., check=False)`，
> 退出码被丢弃、失败不打印、manifest 里也不留痕迹。若每段推理都崩了，父进程
> 仍然 exit 0，磁盘上只有 mp4 没有 JSON——静默全失败。我们自己逐段驱动，
> 才能看见每一个退出码。

### 契约字段（`data/logs/depression/{id}_{date}.json`）

```json
{
  "schema_version": "1.0.0", "elder_id": "E001", "day_key": "2026-08-12",
  "assessed_at": "2026-08-12T00:00:00+08:00", "valid_until": "2026-09-10",
  "status": "low_confidence",
  "result": { "level": "正常", "phq9_median": 3.684, "phq9_spread": 0.0,
              "class_probs": {"正常": 0.927, "轻度": 0.032, "重度": 0.040} },
  "low_confidence_reasons": ["片段数 1 < 2，单段结论不可靠"],
  "evidence": { "n_clips": 1, "n_rejected": 0, "source_duration_sec": 21.989,
                "clips": [{"start": 0.0, "end": 21.989, "phq9": 3.684,
                           "face_ratio": 0.978, "frontal_ratio": 1.0}] },
  "provenance": { "checkpoint": "...", "description_sha256": "0bad8737...",
                  "device": "cuda", "batch_size": 64, "frame_sample_rate": 1,
                  "mpdd_git_rev": "5769083" },
  "calibration": { "validated_on_site": false, "warning": "..." },
  "note": "群体基线绝对评估，与个人基线偏离分不可比、不得合并或相互印证。非临床诊断。"
}
```

**`provenance` 里三个字段不是装饰**，缺一个就会出现"分数莫名其妙变了但查不出原因"：

- `description_sha256` —— MPDD 吃的是个人介绍文本的 1024 维 RoBERTa 嵌入。
  **这份文本改一个字分数就变，而与老人的实际状态毫无关系。** 必须像 scaler
  一样冻结并留指纹。
- `device` / `batch_size` / `frame_sample_rate` —— 实测同机同版本下推理逐位可复现
  （无 RNG、全程 `.eval()`、无反向传播），但这三项任一改变数值就会变。
  注意 `infer_elder_depression.py` 在 CUDA 不可用时会**静默降级到 CPU**，
  所以记录的是**实际生效**的那个（从 stderr 捞）。

### 状态机

| status | 含义 | 显示分数？ |
|---|---|---|
| `assessed` | 正常出分 | ✅ |
| `low_confidence` | 片段数不足，或段间分歧过大 | ✅（加"证据不足"提示） |
| `stale` | 超过 `valid_days` | ❌ |
| `no_clip` | 有录像但一段都没通过筛选 | ❌ |
| `no_source` | 当天没有录像 | ❌ |
| `failed` | 截取或推理报错 | ❌ |

判据集中在 `src/depression/status.py`，**别处不得再写字面量白名单**——
`cold_start_fallback` 那次教训（三处白名单漏了它，建档期 35 天一条预警发不出）
就是这么来的。

**过期一律显示"暂无最新评估"，绝不拿旧分顶今天。** 抑郁评估天然稀疏
（要"有正脸 + 有连续语音"的片段，独居老人可能几周才有一次），比周报本身
更容易踩这个坑。

### ⚠️ 当前模型未校准，不接预警出口

`depression.alert` 配置项刻意置 `false`。原本有两个独立原因，现在只剩第二个：

1. ~~**`alert.py` 没有冷却 / 去重 / 抑制机制**~~ —— **已解决**（见上方「预警按事件去重」）。
   当时的顾虑是：`PERM_step` 实测连续 91 天
   每天报 L3 = 91 次短信 + 强制响铃 + 网格员介入（VALIDATION §7.3）。
   再加一个预警源只会加速家属关掉通知，而通知一关，真正的新变化也收不到了。
2. **checkpoint 在其验证集上对全部 9 个样本预测同一类别**
   （混淆矩阵 `[[6,0,0],[2,0,0],[0,1,0]]`，Macro-F1 0.286，Kappa 0.129），
   且未在本机位 / 本人身上校准。

所以周报里那一节带**未校准警示**，措辞是"情绪状态评估（研究性，未校准）"，
并明确写"结论不可作为任何判断依据"。对外口径仍是行为观察，不下诊断。

### 展示：为什么周报里要排"历次"时间线

大众基线对个体差异**没有免疫力**——天生表情少、语速慢、口音重的老人可能
常年被判同一等级。只看单次绝对等级会把这类人固定误报。把历次评估排成序列
看"变没变"而不是"高不高"，能把这种固定偏置降成背景噪音。

这只是展示层排序，**不涉及任何拟合、不产生个人基线**（真做个人基线就违反了
上面第 2、3 条约束，而且抑郁评估样本太稀疏，也根本估不出来）。

### 设备到货后：一路流，两个消费者

摄像头到货后要接实时流，且**同一路流必须同时供给 MPDD 与 GRU 社交轨**。
设计已在 `src/depression/clip_source.py` 的 docstring 里写明：

```
        C6c RTSP ──► 采集进程 ──┬── 1fps 帧 → 人形检测 → data/raw/camera/{id}/{date}.json → GRU
                                └── 音视频环缓 → 触发器 → data/raw/video_clips/{id}/{date}/ → MPDD
```

- 扇出点必须在**解码之后、处理之前**（RTSP 并发有限，两个进程各拉一路等于解码两遍）
- GRU 那一支**零代码改动**：`camera.py` 的去抖 / ROI 过滤 / 共处时长计算都是现成的，
  `data/raw/camera/{id}/{date}.json` 就是 `daily_job` 现在读的路径与格式
- 不做全量录像：环形缓冲只留最近 ~90 秒，触发才落盘（全天 1080p ≈15~30 GB/天 → 几十 MB/天）
- 实时触发器**不能用 OpenFace**（逐图片 shell-out，跑不了实时），只做轻量粗筛，
  夜间再用现成的 `extract_clips` 那套硬筛选精挑
- ★ **采集进程不许产出任何结论**，只产出上面两样东西。判定全部留在日批处理里——
  本仓删过一次实时运行时（`0dddad1`），教训是"三条路径各写各的，实时采集到的
  特征和每日轨读取的不一定是同一份"

`ClipSource` 抽象隔开了"现在传录像"与"将来接实时流"：`runner` 只认片段目录，
不关心片段怎么来的。设备到货时换 `StreamClipSource` 的实现，下游一行不改。

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

# 冷启动兜底（GRU 就绪前用中位数/MAD 稳健基线检测，消除建档期盲区）
cold_start:
  fallback_enabled: true
  fallback_min_days: 5     # 至少积累N天有效数据才启用兜底
  fallback_lookback: 14    # 滑动基线回看窗口（天）
  fallback_sigma: 3.0      # 加权z-score超过此值判为偏离（比GRU轨更保守）

ewma:
  alpha: 0.05              # EWMA平滑系数
  min_samples_for_dynamic: 20        # 启用动态阈值的最小样本数
  min_samples_for_dynamic_weekend: 8 # 周末池单独降低（周末只占 2/7，攒 20 个要 70 天）
  max_freeze_days: 14      # 偏离日冻结上限：连续冻结这么多天后强制恢复喂入

risk:
  sigma_multiplier: 2.5    # 静态阈值倍数
  # ★ 幅度门槛的单位是 severity = anomaly_score / 当日 dynamic_threshold，不是绝对分
  anomaly_score_thresholds:
    high_spike_severity: 1.5   # 单日峰值：即使已恢复也保持关注的门槛
    sustained_severity: 1.15   # 连续偏离段的平均 severity（配合连续天数判 L2/L3）
  consecutive:
    attention: 1           # Level 1（关注）
    warning: 3             # Level 2（提醒）
    severe: 5              # Level 3（严重）
  continuity:
    max_skip_days: 3       # 持续性统计里连续跳过（缺日/降级日）超过几天就打断
  cold_start_observation_days: 0  # 训练后观察期（已取消：建档期 35 天已让阈值稳定）

alert:                     # 由 alert.trigger_alert 读取，缺项回落内置默认
  level_1: { action: "log_only",           notify: [] }
  level_2: { action: "push_notification",  notify: ["children"],
             repeat_days: 0 }              # 0 = 同级持续不重复推送
  level_3: { action: "force_notification", notify: ["children", "community_worker"],
             repeat_days: 30 }             # L3 惊动网格员，月度重提而非彻底静默
  events:                  # 事件抽象的行为参数（见「预警按事件去重」）
    notify_on_resolve: true       # 缓解也通知
    new_type_cooldown_days: 3     # 新风险类型的最短通知间隔
    weekly_receipt: true          # 持续期每周一条回执（进周报，不推送）

report:
  model: "claude-sonnet-5" # 周报正文的 LLM；anthropic 未安装时自动回落规则模板
  max_tokens: 400
```

#### 为什么幅度门槛的单位是 severity 而不是绝对分

系统里有**两套量纲完全不同**的异常分：

| 轨 | `anomaly_score` 的含义 | 正常天典型值 | 阈值 |
|----|----------------------|------------|------|
| GRU 轨（基线就绪后） | 加权归一化残差 | 0.5 ~ 1.0 | 1.3 ~ 1.6（动态） |
| 冷启动兜底轨（建档期） | 稳健加权 \|z\|（中位数/MAD） | 0.8 ~ 1.8 | 3.0（`fallback_sigma`） |

同一个绝对常数对两者不是同一件事——`1.5` 对 GRU 轨是"高峰"，对兜底轨是
"再普通不过的一天"。而 `severity = 分数 / 自己当天的阈值` 在两套尺度间可比，
按定义 `severity > 1 ⟺ is_deviation`。

**门槛必须 > 1**：偏离日按定义 severity 就大于 1，若门槛取 1.0 则恒真，
"防止擦线偏离仅凭连续天数升到 L3"这条防线会**静默失效**（看不出报错，只是不再拦截）。

两个门槛取值不同也是有原因的：`sustained_severity` 只作用在**当前这段连续偏离**上；
`high_spike_severity` 作用在整个 7 天窗（含已经恢复的日子），而活着的偏离早已被
`consecutive >= 1` 判成 L1，所以峰值门槛唯一的职责是"已经恢复了，但前几天那个尖峰
高到仍值得保持关注"——bar 理应更高。


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

有两个门槛都可能拦住它，按顺序排查：

**a) 幅度门槛**：那段连续偏离的**平均 severity** 没超过 `sustained_severity`（默认 1.15），
即每天都只是"擦线"越过自己的阈值。这是刻意设计——L3 会触发网格员介入 + 强提醒，
不能仅凭天数升级。

判断方法：`judge.py` 的返回值里每轨带 `avg_severity` / `max_severity`。
手算也行：`anomaly_score / dynamic_threshold`，两个值日志里都有。

**b) 方向闸门**：驱动升级的那段连续偏离**全是朝好方向**（睡得更好、出门更多）。
好转不该发风险提醒。

判断方法：返回值里每轨带 `adverse_direction` 字段（**注意它与 `avg_severity` 都不落盘**，
`daily_inference/*.json` 里没有，需要在代码里取或加日志）。
若你认为该报，检查日志里该轨 `signed_z` 的符号是否与
`config/feature_weights.json` 中该维的 `direction` 一致。

### 问题6：建档期（前 35 天）完全没有预警

先确认这不是"设计如此"。建档期走的是**冷启动兜底**，它**能**出等级——
`status` 应为 `cold_start_fallback` 而非 `cold_start`。

- `status == "cold_start"`：兜底没启动。查 `cold_start.fallback_enabled` 是否为 true，
  以及该轨历史有效数据是否够 `fallback_min_days`（默认 5 天）。
- `status == "cold_start_fallback"` 但 `is_deviation` 恒 false：兜底的 sigma 是 **3.0**，
  比 GRU 轨保守得多，正常波动本来就不该报。
- `is_deviation` 为 true 但 `risk_level` 为 0：**这是 bug**，说明状态白名单又漏了。
  查 `src/utils/status.py` 的 `EVALUABLE_STATUSES`，以及是否有人在别处新写了字面量白名单。

### 问题7：日志有空洞时"连续 N 天"数得不对

先确认预期：缺日与降级日**都被跳过**（既不累加也不打断），但**连续跳过超过
`risk.continuity.max_skip_days`（默认 3）就打断连续段**。

- 断 2 天 → 不打断，前后视为同一段；
- 断 17 天 → 打断，前后是两段无关的偏离。

两轨全不可用那天 `daily_job` **不生成日志文件**，所以磁盘上的空洞就是"那天没测到"。
若你需要更宽松/更严格，调 `max_skip_days`，不要去改计数逻辑。

### 问题8：改了 `config/settings.yaml` 但没生效

本轮之前确实有三处配置是死的（`alert` 整段、`report.*`、以及压根不存在的
`ewma.max_freeze_days`），现已全部接线。若仍不生效：

- 确认改的是 `config/settings.yaml`（本项目唯一的运行配置）——
  无任何代码读取；
- 确认调用链传了 `config`：`trigger_alert(..., config=config)` 不传时用的是
  `alert.py` 内置默认；
- `ewma.max_freeze_days` 对**已建档**的基线不生效——状态文件里存的值优先，
  已有基线保持自己建档时的口径直到重新建档。

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

**最后更新**：2026-07-31  
**项目版本**：v2.1.1

> **v2.1.1 要点（2026-07-31 代码走查，不改算法思路，只把已有设计真正接通）**：
> ①**冷启动兜底接进判定层**——此前 `judge`/`rules` 的状态白名单漏了
> `cold_start_fallback`，整个 35 天建档期一条预警都发不出（实测连续 6 天严重异常
> `risk_level` 全 0）；新增 `src/utils/status.py` 作白名单的单一事实来源。
> ②**幅度门槛改用 severity**（`分数/自身阈值`）并只算在驱动升级的那段 streak 上——
> 绝对常数既会被 7 天窗里的正常天稀释，又无法同时适配 GRU 轨与兜底轨两套量纲。
> ③**持续性统计改按自然日回溯**，缺日与降级日统一跳过但连续跳过 >3 天即打断
> （此前 5 条日志跨 22 个日历日仍数出 consecutive=5 → L3）；加载天数由规则门槛推导，
> 解开"7 天日志窗 vs 7 天门槛"的死锁。
> ④**`data_quality` 按轨落进推理日志**——此前该标记只进 features CSV，
> "降级日跳过持续性统计"这条不变量在生产链路里从未生效。
> ⑤接线三处死配置（`alert` 整段 / `report.*` / 新增 `ewma.max_freeze_days`）；
> 清理 `requirements.txt` 中 6 项零引用依赖（`pyaudio` 会让整条 pip install 中止）。
> ⑥补周轨入口 `scripts/run_weekly_pipeline.py`、产出 MPDD 证据契约、
> 修周报统计恒 0.00 与周期错配。
> 测试 312→332，范围 1 19→20（新增"兜底能出等级"断言），范围 2 保持 9/9。
> 详见 `docs/VALIDATION.md` §8。**预警冷却/去重仍未做**，是当前第一优先级。

> v2.1 要点（在 v1.8 基础上：**单轨 6 维 → 双轨（睡眠 8 维 / 社交 5 维）**，两轨各有独立 GRU 与 scaler，从结构上切断跨类型"串味"；风险类型 2→3（睡眠稳定性 / 社会连接减弱 / 作息节律紊乱），规则改为"必选维 + 可选池"结构，`threshold_ratio` 按规则可配；`social_turns` 随音频链路删除，社会接触改由 C6c 边缘侧人形检测的 `copresence_min` 承担；`hrv_rmssd` 降级为 `night_hr_mean` 弱代理并降权；建档期 21→35 天、取消冷启动观察期；EWMA 增加**偏离日冻结**（上限 14 天），风险判定增加**方向闸门**，`validate_synthetic.py` 补固定 torch 种子——修复了三个一层压一层的缺陷：持续性异常被自适应基线掩盖、好转被判成"严重风险"、以及单维门槛对变异系数不同的维不等价（`copresence_min` 的 z 在门槛上抖动、社交崩塌报不出类型），详见 `docs/VALIDATION.md` 缺陷④⑤⑥。）

> v1.7 要点（在 v1.6 基础上）：①冷启动建档期 14→21 天并提为可配置项 `build_days`，附 20-seed 过拟合对比实验；②修复风险类型分类两处缺陷——"永不激活"（日志不写回 risk_types）与"正常日误激活"（类型判定挂靠 is_deviation）；③新增合成数据验证脚本 `validate_synthetic.py`（范围1跑通）与 `validate_discriminative.py`（范围2判别力，含真实噪声+混淆项）；④新增 `docs/TRAINING.md` GRU 训练详解。⑤修复模拟数据异常注入天数（25-30 → 40-46）与总天数（50 → 60），避开新建档期+观察期。测试 134 全绿。已知待办：跨类型"串味"（共享GRU溢出）、真实传感器采集/推送仍为桩。文档：本 README（总览+现状）、`docs/TRAINING.md`（训练详解）、`docs/TODO.md`、`docs/VALIDATION.md`（三层验证+§7执行记录）

> v1.6 要点：打通统一系统链路（统一调度器真正调用每日管道）；声学按自然日聚合 + 缺失标记；微调与冷启动共用 early-stopping；统一系统快照原子写；save_interval 走配置。

> v1.5 要点：单人系统；实时轨仅采集不报警；GRU 加健康门禁 / 冷启动兜底 / early-stopping / 微调防污染；判定链路修复（前向填充生效 / 观察期按训练后推理计数 / 严重级加幅度门槛）；移除时间编码死代码，特征统一为 10 维 `FEATURE_NAMES`。
