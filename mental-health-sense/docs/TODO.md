# 待修清单（实现落差与同步隐患）

> **当前状态（v2.2：双轨 GRU + MPDD 抑郁旁路）**
>
> 测试 **475 通过**；范围 1（链路正确性）**20/20**；范围 2（判别力）**9/9**。
>
> 下方 2026-07-31 及更早的条目均**属历史归档**，保留备查。

## 2026-07-31 第二轮走查：20 个缺陷（已修）

集成 MPDD 抑郁通道时顺带做的全链路审计。共同特征仍是“断言打在错的层”——
全部 20 项都活在“332 全绿 + 20/20 + 9/9”之下。实证见 `VALIDATION.md` §9。

| 严重度 | 缺陷 | 后果 |
|--------|------|------|
| 高危 | 补算历史日判成最新日 | 日志覆盖 07-01~08-29 时补算 08-15，判的是 08-29 的等级、按它发预警、写进 08-15 的契约；08-15 永远拿不到 qualifies |
| 高危 | 单轨掉线打断连续段 | 不可用轨既无子字典也不在 track_quality 里，回退全落空后按 valid 计入 → 打断。坏掉的正是双轨故障隔离本身 |
| 高危 | 判级不看 data_quality | degraded 日既累加也能打断 consecutive。5 天偏离中 2 天掉线 → rules 正确不激活，judge 却报 **L3**（短信+强制响铃+网格员） |
| 高危 | imputer 在原始量纲填 0 | `night_hr_mean=0 → z=−15.2`；`copresence_min=0 → z=−1.44` 足以满足 down 方向，而它是 social_decline 的必选维（设计上缺了就不该触发）。缺 1 维时质量仍判 valid，这行 −15σ 还会进 `StandardScaler.fit` |
| 中高 | 周报重判绕过完整判定窗 | 周报说“本周风险类型：无”，与家属当天收到的预警矛盾 |
| 中高 | `except Exception` 吞掉配置错误 | 并连带切断写回链 → 持续性计数永久钉在 1 → 风险类型数学上永不激活 |
| 中高 | 异常被吞后 status 仍 success | cron 与监控全绿而老人当天零监测 |
| 中 | 同日重跑改变 dynamic_threshold | 擦线分数翻转 is_deviation，而它驱动 consecutive / qualifies / 微调排除集 |
| 中 | 周报窗口比数据晚一天 | 每周固定漏掉一天，且标题区间比数据宽一天 |
| 中 | rules 默认门槛 1.2 与配置的 1.0 漂开 | 1.2 是 settings.yaml 论证过的“假绿”；传裁剪配置的调用方静默拿到损坏值 |
| 中 | 就地写非原子 | 整表重写崩一半 → 建档期历史不可恢复；单个日志被截断 → judge/周报/微调全部永久崩溃 |
| 中 | `_get_recent_quality` 把今天算进离线判据 | 同一输入首跑判 insufficient、重跑判 offline |
| 测试 | 周报只测 helper、integration 断言 validator 辅助函数 | 真实链路零回归保护 |
| 测试 | 状态白名单字面量残留 | `inference.py` 那份是 `EVALUABLE_STATUSES` 去掉 `cold_start_fallback` 的拷贝 |
| 次要 ×6 | 见 `VALIDATION.md` §9 | 缺 adverse_direction 键 / EWMA 空池 TypeError / alerted 用原始整数 / 契约硬编码时区 / 不可达的“退回全段估残差”分支 / 微调窗口把不连续自然日当相邻天 |

## 2026-07-31 预警按事件去重（方案 B，已修）

`alert.py` 原本对每个 >=L1 的日子无条件发一次通知，没有任何记忆。
把“状态持续”当成了“每天都是新事件”。

| | 修复前 | 修复后 |
|---|---|---|
| E001 60 天 L2+ 推送 | 12 次 | **4 次** |
| E001 60 天 L3 强制响铃 | 8 次 | **2 次** |
| `PERM_step` 91 天 L3 | **91 次** | **4 次** |

发出的 4 次恰好对应 4 个真实事件。实现见 `src/risk/alert_state.py`。

**最关键的一条不变量：升级穿透一切抑制**（冷却、已确认、同日幂等分支）。
纯粹按“同一等级 X 天内最多一次”做冷却会在 L2 的冷却窗内吃掉 L3——
拿误报换漏报，比修复前更糟。`tests/test_alert_events.py` 的
`TestEscalationBypassesEverything` 是这条防线的守门员。

配套的告知义务：周报新增“预警回执”章节。通知层静默后周报若也一字不提，
“系统安静”与“系统挂了”就无法区分——同“长期降级和长期正常必须可区分”。

---

新增 `tests/test_regression_2026_07_31.py`（14 条，全部打在真实链路输出上）。
**已实测：回退到修复前的代码后 13/14 变红**——这一步是刻意做的，因为本仓的教训
就是绿色本身不构成证据。

> **已知未解决**（如实记录，非阻断）：
>
> - ~~**永久性变化会无限期每日报警**~~ —— **已修（2026-07-31，方案 B 事件模型）**
>   见本文档下方“预警按事件去重”一节与 VALIDATION §10。
>
> - **MPDD checkpoint 未在本机位校准**：当前 Track1 ternary checkpoint 在其验证集上
>   对全部 9 个样本预测同一类别（混淆矩阵 `[[6,0,0],[2,0,0],[0,1,0]]`，
>   Macro-F1 0.286，Kappa 0.129）。管道是通的、结果可复现，但**数值本身不可信**，
>   周报里带未校准警示。设备到货后必须做域验证：真机位（客厅监控视角，
>   非访谈构图）下还能不能截出合格片段——这是唯一可能让整条抑郁线作废的风险。
>
>   ★ `depression.alert` 仍置 `false`，但**阻塞理由已从两条减为一条**：
>   ~~alert.py 无冷却机制~~ 已解决；剩下的是本条（checkpoint 未校准）。
>   代码层的线已经接好——抑郁事件流独立于基线事件流、映射刻意保守
>   （即使判“重度”也只到 L2，不惊动网格员），换到可用 checkpoint 后
>   改一个配置项即可启用。
>
> - **残差统计（`residual_stats`）未按周内分池**，`copresence_min`（周末 ×2.4）、
>   `out_of_home_min` 等有周末效应的维，残差 std 被双峰分布抬高、z 分被系统性压小。
>   根治需远长于 7 天的留出段，留到真机阶段。
>   现以 `social_decline` 的 `threshold_ratio=1.0` 局部补偿（见 VALIDATION 缺陷⑥）。
>
> - **GRU 固定 7 天窗对持续性变化钝感**：异常持续到第 3 天后输入窗填满异常日、
>   预测跟着漂移，残差收缩。EWMA 层已用偏离日冻结缓解（缺陷④），
>   但模型层没有对应机制。属固定窗口预测器的固有取舍。
>
> - **实时流采集尚未实现**（`StreamClipSource` 是 NotImplementedError）：
>   等摄像头设备到货。设计已写进 `src/depression/clip_source.py` 的 docstring：
>   一路 RTSP 解码一次、在帧上扇出给 GRU 社交轨与 MPDD 两个消费者；
>   环形缓冲触发式导出而非全量录像；采集进程**不许产出任何结论**。
>
> - `KM_social_partial`：只共处↓、外出照常的场景必然漏报，是“三项全中”规则
>   换取低误报的代价。

---

## 2026-07-31 代码走查：判定链路缺陷（已修）

这一轮的共同特征是**验证脚本与单测都断言在错的层**，所以缺陷能长期存活在
"312 全绿 + 19/19 + 9/9"之下。详细实证与数据见 `docs/VALIDATION.md` §8。

| 缺陷 | 后果 | 修法 |
|------|------|------|
| 冷启动兜底进不了判定层 | **建档期 35 天完全无预警**。实测连续 6 天 score 166→58（阈值 3.0）、偏离 6/6，`risk_level` 全 0 | 新增 `src/utils/status.py` 作状态白名单的单一事实来源；三处硬编码白名单改用它 |
| `data_quality` 从未落进推理日志 | "degraded 日既不累加也不打断"这条写进 CLAUDE.md 的不变量**在生产链路里从未生效**（实测 60/60 日志无该字段） | 按轨透传并落盘；`_counts_toward_consecutive` 改按轨判定 |
| 幅度门槛被 7 天窗稀释、且与量纲脱节 | 3 天各 1.45（都越过各自阈值）被 4 个正常天拉到 0.907 < 1.0，只报 L1 | 改算在连续偏离段上；门槛换成 severity = 分数/自身阈值 |
| 持续性统计不看日历 | 5 条日志跨 22 个日历日（断 17 天）仍数出 consecutive=5 → L3 | `_walk_back` 按自然日回溯；缺日与 degraded 同样跳过，但连续跳过 > `max_skip_days` 即打断 |
| 7 天日志窗与 7 天门槛互卡 | circadian 降级模式只要历史里有 1 个 degraded 日就永远不可激活 | 加载天数由 `required_history_days` 从规则门槛推导 |
| 三处配置段无人读取 | `alert` 整段、`report.model/max_tokens` 均未生效；`ewma.max_freeze_days` 配置里根本没这个键 | 全部接线 |
| `requirements.txt` 装不上 | `pyaudio` 缺 portaudio 头文件 → pip 整体中止 → torch 一个都装不上，照文档走的新克隆跑不起来 | 删 6 个零引用依赖 |
| 周报异常分恒 0.00 / 周期错配 | 读顶层 `anomaly_score`（双轨后已不存在）；本周无数据时静默拿别的周顶替，标题与内容对不上 | 按轨统计；无数据出"数据不足"周报 |
| 周轨没有入口、MPDD 契约从未产出 | `run_weekly_pipeline` 零调用方；`build_mpdd_evidence` 定义了测了但没有链路写过 | 新增 `scripts/run_weekly_pipeline.py`；契约接进 `daily_job` |

> **已知未解决**（如实记录，非阻断）：
> - **永久性变化会无限期每日报警（优先级最高，本轮有意未做）**：`PERM_step` 诊断场景实测
>   连续 91 天每天判 L3，冻结按 `max_freeze_days=14` 正常释放、阈值也回升
>   （1.18→1.91），但 score 4.5 远超其上，重新基线化追不上。而 L3 = 短信 +
>   网格员 + 强制响铃，且 `alert.py` 仍**无任何冷却/去重/抑制机制**。
>   检测层是对的（老人确实持续处于低效睡眠），缺口在**预警策略层**：
>   把"状态持续"当成了"每天都是新事件"。见 VALIDATION §7.3。
>   本轮已把 `max_freeze_days` 提到配置（此前硬编码、调都没法调），
>   但那只是缓解阈值追赶速度，**不是**这个问题的解法。
> - **残差统计（`residual_stats`）未按周内分池**，`copresence_min`（周末 ×2.4）、
>   `out_of_home_min` 等有周末效应的维，残差 std 被双峰分布抬高、z 分被系统性压小。
>   根治需远长于 7 天的留出段，留到真机阶段。
>   现以 `social_decline` 的 `threshold_ratio=1.0` 局部补偿（见 VALIDATION 缺陷⑥）。
> - **GRU 固定 7 天窗对持续性变化钝感**：异常持续到第 3 天后输入窗填满异常日、
>   预测跟着漂移，残差收缩。EWMA 层已用偏离日冻结缓解（缺陷④），
>   但模型层没有对应机制——长期异常的 z 分仍会随时间衰减。
>   属固定窗口预测器的固有取舍，真机阶段需评估是否值得引入更长窗或双时标。
> - `KM_social_partial`：只共处↓、外出照常的场景必然漏报，是"三项全中"规则
>   换取低误报的代价。
>
> ---
>
> 本清单记录代码走查中发现的"实现落差"与"同步隐患"。区别于 bug：这些多是
> "设计对、但代码没接完"或"边界场景会出错"的地方，直接影响系统能否声称"端到端跑通"。
> 按优先级排序：P0 阻断端到端跑通，P1 真实部署下会出错，P2 一致性/健壮性。

| 优先级 | 含义 |
|--------|------|
| **P0** | 阻断"统一系统端到端真正跑通"，答辩/演示前必修 |
| **P1** | 真实部署（老人夜间静默 + 凌晨触发）下会产出错误数据 |
| **P2** | 一致性、健壮性、可维护性，不紧急但该收口 |

## 修复状态（2026-07-26）

全部 7 项已修复，133 个测试通过（原 121 + 新增 12）。

| 项 | 状态 | 关键改动 |
|----|------|---------|
| P0-1 | ✅ | `UnifiedScheduler._run_daily_inference` 真正调用 `run_daily_pipeline`；删除 mock getter |
| P0-2 | ✅ | 声学统一经 `UnifiedDataManager` 自然日聚合；`daily_job.load_raw_sensors` 供三路传感器 |
| P1-1 | ✅（方案B） | 新增按自然日持久化 utterance（JSONL）+ `aggregate_natural_day`；滑动窗口读取加墙上时钟过滤 |
| P1-2 | ✅ | `get_daily_acoustic_with_quality` 无数据返回 `data_quality="missing"` |
| P2-1 | ✅ | `RealtimeMonitor.save_interval` 从 `realtime_config.yaml` 读取 |
| P2-2 | ✅ | 抽出 `_train_loop`，冷启动/微调共用 early-stopping+回滚；顺带修 best_state 与 best_loss 差一梯度步的 off-by-one |
| P2-3 | ✅ | 快照原子写（临时文件 + `os.replace`）；utterance JSONL append-only |

> **补充说明（2026-07-29，已超越）**：其中所有**声学相关的数据流水项**——
> `UnifiedDataManager` 声学聚合/快照、`sensevoice_engine` 的窗口与 utterance 逻辑、
> `get_daily_acoustic_data` 的默认常量兜底等（主要涉及 **P0-2 / P1-1 / P1-2**）——
> 已随"移除抑郁判断 + SenseVoice 声学子系统（10→6 维）"整体删除，不再适用。
> 下方历史条目仅保留备查，不代表当前代码结构。

> 下方为原始问题记录，保留备查。

---

## P0-1　UnifiedScheduler 的每日推理是 TODO 桩，没真正调 GRU

**位置**：`src/unified_scheduler.py:167-171`、`_get_sleep_data/_get_activity_data/_get_social_data`（176-196）

**现象**：
- `_run_daily_inference` 里 `daily_inference(...)` 被注释掉，只 `print("GRU推理完成（待实现）")`。
- 睡眠/活动/社交三路数据全是写死的模拟常量（`sleep_efficiency: 0.85` 等）。

**后果**：走 `start_unified_system.py` 这条"生产路径"时，**GRU 根本没跑、风险判定没发生、预警不会触发**。真正完整实现的是 `src/scheduler/daily_job.py::run_daily_pipeline`，两个入口不一致。

**修法**：让 `_run_daily_inference` 直接调用 `run_daily_pipeline(elder_id, date, raw_data=...)`，把实时轨聚合的 acoustic_data 通过 `raw_data` 传入；删掉三个 mock getter 或改成真正的传感器读取适配层。

---

## P0-2　实时轨与每日轨的数据衔接没打通（三个入口写不同文件）

**位置**：`src/realtime/monitor.py:167`、`src/unified_data_manager.py:82/117`、`src/scheduler/daily_job.py:263-285`

**现象**：三条路径各写各的，文件名/结构不统一：
- `RealtimeMonitor._save_current_features` → `features_{date}.json`（records 数组，一天多条）
- `UnifiedDataManager._save_snapshot` → `snapshot_{date}.json`（单条 acoustic_data）
- `daily_job._load_raw_and_aggregate` → 从 `data/raw/acoustic/{id}/{date}.json` 读，**根本不碰上面两个**

**后果**：实时采集到的声学特征，和每日轨实际读取的声学特征，**不一定是同一份**——取决于用哪个脚本启动。生产链路（采集→每日推理）事实上没闭环。

**修法**：确定唯一衔接契约——建议实时轨只经 `UnifiedDataManager` 落 `snapshot_{date}.json`，每日轨统一从 `UnifiedDataManager.get_daily_acoustic_data` 取；`daily_job` 的 raw 读取改为可注入，纯传感器数据（睡眠/活动/社交）走 `data/raw/`，声学走实时轨快照。与 P0-1 一起改。

---

## P1-1　24 小时窗口用"数据时间"裁剪，每日轨用"墙上时钟"取数，两者错位

**位置**：`src/realtime/sensevoice_engine.py`（`_cleanup_old_data` / `get_current_features`）、`unified_data_manager.py:71/98`

**根因**：窗口清理只在 `add_utterances` 里发生，`cutoff = 最近一句话的时间戳 - 24h`。**没人说话就不清理，窗口冻结**。而 `get_current_features()` 取数时**完全不按时间过滤**，直接把 buffer 现存全部拿来算。

**三个具体后果**：
1. **夜间静默 → 读到陈旧数据**：每日轨凌晨 02:00 触发，老人最后说话可能是前晚 21:00。此时窗口冻结在 21:00，读到的是 `[前天21:00, 昨天21:00]` 这个错位区间，且缺昨晚 21:00 之后的时段——它代表的**不是"昨天一整天"**。
2. **文件名 vs 内容时间系统不一致**：`_save_snapshot` 用 `datetime.now()`（墙上时钟）决定 `snapshot_{date}.json` 的文件名，但内容按数据时间戳裁。跨午夜时会给旧窗口贴上新一天的标签，每日轨按 date 取数就张冠李戴。
3. **静默期跨日污染**：连续少言时 buffer 残留 >24h 旧数据又不被清理，前天的话被算进今天特征。

**修法（三选一，推荐方案 B）**：
- A（最小改动）：`get_current_features` / `_save_snapshot` 取数时用 `time.time()` 再做一次 cutoff 过滤，不依赖说话事件触发的清理。
- **B（最干净）：每日轨改为"按自然日聚合"**——实时轨只持续存带时间戳的原始 utterance，每日轨凌晨捞"昨天 00:00–23:59"自然日区间自行聚合。语义与 GRU"一天一条"天然对齐，彻底甩掉滑动窗口 vs 墙上时钟的错位。
- C：清理改定时驱动，周期性调 `_cleanup_old_data(time.time())`。

> 说明：滑动窗口本身没错，它适合"实时看当前状态"。错在**拿它去满足每日轨"某自然日汇总"这个不同语义的需求**。

---

## P1-2　`get_daily_acoustic_data` 对"今天"和"过去"用两套数据源

**位置**：`src/unified_data_manager.py:70-94`

**现象**：`date == today` 时从内存聚合器实时取；`date` 是过去则读磁盘快照；都没有则返回一组默认常量（`sad_ratio: 0.05` 等）。

**后果**：
- 每日轨在凌晨 02:00 跑的是"昨天"的数据（date=昨天），走的是"读磁盘快照"分支——如果昨天的快照因 P1-1 错位或进程重启没落盘，会**静默落到默认常量**，即用一组假的正常值喂进 GRU，可能掩盖真实偏离。
- "默认值兜底"没有任何日志或质量标记，上层无法区分"真的正常"还是"根本没数据"。

**修法**：默认值分支必须打标记（如返回 `data_quality: "missing"`），让每日轨据此把该天标成 insufficient 而非 valid；与 validator/imputer 的缺失处理链路对齐。

---

## P2-1　save_interval 有多个不一致默认值，且不读配置

**位置**：`monitor.py:39`（默认 1800s）、`monitor.py:236`（示例传 180s）、`config/realtime_config.yaml`（无此项）

**现象**：保存间隔硬编码在实例化处，`realtime_config.yaml` 里没有对应配置项，代码也没去读。三处默认值不统一（30 分钟 vs 3 分钟）。

**修法**：把 `save_interval` 提到 `realtime_config.yaml`（如 `aggregator.save_interval_seconds`），`RealtimeMonitor` 从配置读取，去掉硬编码。

---

## P2-2　每周微调缺 early-stopping，与冷启动训练策略不一致

**位置**：`src/baseline/trainer.py:353-365`（`weekly_retrain`）

**现象**：冷启动 `train_initial_baseline` 有 early-stopping + 回滚最优权重（159-193 行），但微调只记录 `best_loss`、既不提前停止也不回滚到 best 权重，固定跑满 50 epoch。

**后果**：风险不大（低 lr + 已有备份 `gru.prev.pth` 可回滚），但两条训练路径的过拟合抑制策略不统一，微调仍可能在小样本上轻微过拟合。

**修法**：给 `weekly_retrain` 补上与冷启动一致的 early-stopping + best_state 回滚逻辑（可抽成公共函数复用）。

---

## P2-3　快照写入非原子，进程中断可能损坏 JSON

**位置**：`monitor.py:160-182`（读-改-写追加）、`unified_data_manager.py:96-119`

**现象**：`_save_current_features` 先读整份 JSON、append 一条、再整体覆盖写。写入过程中断电/被 kill 会留下半截文件，下次读直接抛异常。

**修法**：写临时文件再 `os.replace` 原子替换；或改用 append-only 的行式格式（JSONL）。

---

## 建议修复顺序

1. **P0-1 + P0-2 一起改**（同一处衔接逻辑）→ 先让统一系统真正端到端跑通，这是演示/答辩前提。
2. **P1-1（选方案 B）+ P1-2** → 保证喂进 GRU 的日级特征时间语义正确、缺失可追踪。这直接决定 VALIDATION.md 里"层次 A 端到端冒烟测试"的结论可不可信。
3. **P2 三项** → 收口一致性与健壮性，不阻断主线。

> 关联（已解决）：P0/P1 修复前，`docs/VALIDATION.md` 层次 A"端到端链路冒烟测试"实际只覆盖 `daily_job` 单入口，而非"统一系统"生产链路。现 P0 已打通——`UnifiedScheduler` 真正调用同一个 `run_daily_pipeline`，冒烟结论已覆盖统一系统链路（VALIDATION.md 2.2 节已同步说明）。

