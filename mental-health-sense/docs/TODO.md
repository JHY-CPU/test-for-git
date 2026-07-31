# 待修清单（实现落差与同步隐患）

> **当前状态（v2.1 双轨）**：本文档下方记录的 7 项已全部收口，**属历史归档**。
> 其中声学相关项已随音频链路删除而不再适用。
> 当前测试与验证状态：**312 个单元测试通过**；范围 1（链路正确性）**19/19**；
> 范围 2（判别力）**9/9**。
>
> 本轮双轨重构中**新发现并修复**的三个缺陷记录在 `docs/VALIDATION.md`（缺陷④⑤⑥），
> 不在本清单内——它们不是"代码没接完"，而是设计缺陷：
> EWMA 学会异常把持续性异常掩盖掉、以及好转被判成"严重风险"。
>
> **已知未解决**（如实记录，非阻断）：
> - **残差统计（`residual_stats`）未按周内分池**，`copresence_min`（周末 ×2.4）、
>   `out_of_home_min` 等有周末效应的维，残差 std 被双峰分布抬高、z 分被系统性压小。
>   根治需远长于 7 天的留出段，留到真机阶段。
>   现以 `social_decline` 的 `threshold_ratio=1.0` 局部补偿（见 VALIDATION 缺陷⑥）。
> - **永久性变化会无限期每日报警（新，优先级最高）**：`PERM_step` 诊断场景实测
>   连续 91 天每天判 L3，冻结按 `max_freeze_days=14` 正常释放、阈值也回升
>   （1.18→1.91），但 score 4.5 远超其上，重新基线化追不上。而 L3 = 短信 +
>   网格员 + 强制响铃，且 `alert.py` 无任何冷却/去重/抑制机制。
>   检测层是对的（老人确实持续处于低效睡眠），缺口在**预警策略层**：
>   把"状态持续"当成了"每天都是新事件"。见 VALIDATION §7.3。
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

