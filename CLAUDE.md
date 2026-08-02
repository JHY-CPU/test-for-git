# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库结构

Git 仓库根是 `test-for-git/`，项目实际在子目录 `mental-health-sense/`——**所有命令都从该子目录执行**（脚本用 `sys.path.insert` 挂项目根，`data/` 路径由 `src/utils/io.py:get_project_root()` 推出，均以它为基准）。

- 主文档是 `mental-health-sense/docs/README.md`（不在项目根），另有 `docs/TRAINING.md`（GRU 训练详解）、`docs/VALIDATION.md`（三层验证框架 + §7 执行记录 + §8 代码走查，代码里的"缺陷④⑤⑥"指 §7）、`docs/TODO.md`（已知未解决问题）。

## 常用命令

环境：本机只有 `python3`（3.10），仓库未带 venv，torch/pytest 默认未安装。

```bash
cd mental-health-sense
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

> 核心链路只需 numpy / pandas / scikit-learn / torch / joblib / PyYAML / loguru / pytest；`openai`（周报正文走 DeepSeek，OpenAI 兼容格式）只用于周报正文，未装或未设 `DEEPSEEK_API_KEY` 则自动回落规则模板。`funasr` / `modelscope` / `pyaudio` / `scipy` / `opencv-python` / `APScheduler` 已于 2026-07-31 从 `requirements.txt` 移除（零引用；其中 `pyaudio` 缺 portaudio 头文件会让整条 pip install 中止，torch 一个都装不上）。

> ⚠️ 本机跑 pytest 必须剔除系统 `PYTHONPATH`：`/opt/ros/humble` 会被 pytest 的
> 收集阶段导进来，直接 `python -m pytest` 会崩在 `ModuleNotFoundError: lark`，
> 一条用例都跑不到。用 `env -u PYTHONPATH python -m pytest`。
> 这与"MPDD 子进程必须剔除系统 `PYTHONPATH`"是同一个污染源（见抑郁旁路一节）。

```bash
# 测试（26 文件 / 517 用例）
env -u PYTHONPATH python -m pytest                 # pytest.ini 已设 testpaths=tests 与 -v
python -m pytest tests/test_risk_judge.py
python -m pytest tests/test_risk_judge.py::TestJudgeRiskLevel::test_xxx
python -m pytest -m "not slow"                     # 已注册 marker: slow / integration

# 完整工作流（单人系统，默认 elder_id=E001）
python scripts/generate_simulation_data.py         # 60 天双轨模拟数据
python scripts/train_all_baselines.py              # 双轨建档（--track sleep 只训一轨）
python scripts/run_daily_pipeline.py --date 2026-08-15
python scripts/run_weekly_pipeline.py              # 周轨：双轨微调 + 周报（--no-retrain 只出周报）

# 抑郁评估（MPDD 旁路通道，事件驱动，**不挂在日管道里**）
python scripts/run_depression_assessment.py --elder E001 --date 2026-08-12 \
    --video /path/to/footage.mp4

# 预警事件：查看 / 确认（确认后同级持续静默，但升级仍会破静默）
python scripts/ack_alert.py --elder E001 --show
python scripts/ack_alert.py --elder E001

# 端到端验证（单测不覆盖判定链，改算法/配置后必须重跑）
python scripts/validate_synthetic.py               # 范围1：链路跑通 + 双轨信号隔离，20 断言（--keep 保留 V001 数据）
python scripts/validate_discriminative.py          # 范围2：10 场景判别力，当前 9/9（KM 不计分）
python scripts/validate_discriminative.py --only TP_sleep
python scripts/validate_discriminative.py --drift  # 长周期慢坡诊断 4×80~140 天，很慢，只记录不计分
```

## 架构

面向单个居家老人的**每日批处理**趋势预警系统，没有实时运行时。链路：
`scripts/run_daily_pipeline.py` → `src/scheduler/daily_job.py:run_daily_pipeline`（按轨：聚合 → 填充 → 校验 → 存 CSV）→ `src/baseline/inference.py:daily_inference`（按轨 GRU 残差 + EWMA 阈值）→ `src/risk/judge.py:quick_judge`（等级 + 类型）→ `src/risk/alert.py:trigger_alert`。周轨 `src/scheduler/weekly_job.py` 做微调 + 周报。

### 双轨是核心不变量

睡眠轨（8 维，源小贝壳）与社交轨（5 维，源萤石 C6c+T1C）各有**独立**的 GRU、scaler、残差统计、EWMA 池，互不共享——这是 v2.1 从结构上切断跨类型"串味"的手段。

- **绝不跨轨比较绝对分**：两轨维度数与权重和不同，残差尺度不可比。等级是"每轨各自算、取较高者"（`judge.py:judge_risk_level`）。
- 特征表在 `src/baseline/scaler_utils.py`（`SLEEP_FEATURES` / `SOCIAL_FEATURES`），**列表顺序即向量下标**，必须与 `config/feature_weights.json` 中的顺序严格一致。改顺序会让已存的模型/scaler/残差统计全部错位，必须删基线重训。
- 磁盘布局按轨后缀：`data/baselines/{id}/gru_{track}.pth`、`scaler_{track}.pkl`、`residual_stats_{track}.pkl`、`ewma_{track}*.pkl`（`baseline_meta.json` 两轨共用，按 track 分键）；`data/features/{id}/features_{track}.csv`。原始数据在 `data/raw/{sleep,activity,camera}/{id}/{date}.json`。

### 双残差契约（最容易写错的地方）

`signed = observed − predicted`（判方向）、`abs = |signed|`（打幅度分），两套统计各自独立存在 `residual_stats` 里。

- `anomaly_score = Σ(abs_residual·w)/Σw`，方向无关；风险规则的方向性判定必须读 `signed_z`（`signed_residual / signed_std`，只除 std 不减均值以保号）。
- 用 abs 判方向会让"睡眠好转"与"睡眠恶化"无法区分——这正是方向闸门要解决的历史缺陷。
- **残差统计只能在留出段估计**（`trainer.py`：35 天 = 7 窗口 + 21 训练 + 7 留出）。用训练集残差估阈值会让 std 趋零 → 阈值分母趋零 → 建档期一过疯狂误报。

### 阈值与判级

- `static_threshold = 加权(abs_mean) + sigma·加权(abs_std)`；`dynamic_threshold = min(static, ewma)`，取 min 是为了防老人缓慢衰退后系统"习以为常"。
- **幅度门槛的单位是 `severity = anomaly_score / 当日 dynamic_threshold`，不是绝对分**。GRU 轨（归一化残差，阈值 1.3~1.6）与冷启动兜底轨（稳健加权 |z|，阈值 = `fallback_sigma` = 3.0，正常天可达 1.77）量纲根本不同，同一个绝对常数对两者不是同一件事。门槛必须 > 1：偏离日按定义 `severity > 1`，取 1.0 等于恒真，防线会静默失效。
- **`cold_start_fallback` 是可评估状态**，与 `success`/`observation` 一起由 `src/utils/status.py` 统一定义。曾因三处白名单漏了它，导致整个 35 天建档期一条预警都发不出（`VALIDATION.md` §8.2）。加新状态时改那一个文件。
- **重跑复用首跑阈值只在同源路径之间**（`inference._load_prior_thresholds`）。兜底轨落盘的三个阈值都等于 `fallback_sigma`（稳健 |z| 量纲），GRU 轨是归一化残差；建档完成后补算建档期内某天会同时满足"基线已就绪"与"已喂过 → 判为重跑"，把 3.0 当成 GRU 阈值 → `is_deviation` 恒 False。与"幅度门槛必须用 severity 而非绝对分"是同一条量纲不可比。
- EWMA **偏离日冻结**（上限 `ewma.max_freeze_days`，默认 14）：判为偏离的当天不喂给基线，否则持续性异常两三天就被自己的历史掩盖。社交轨分 weekday/weekend 两池（周末子女探访效应），周末池样本稀疏，`min_samples_for_dynamic_weekend=8`。喂入按 `day_key` 去重（`day_key <= last_day_key` 拒收），补算/重跑同一天不会把分喂进基线两次。
- L2/L3 除连续天数外还要过**幅度门槛**（连续偏离段的平均 `severity > sustained_severity`）与**方向闸门**（`judge.py:_has_adverse_movement`：驱动升级的那段连续偏离里至少一半天数朝坏方向，否则封顶 L1）。拿不到方向元数据或 `signed_available=False` 时不压等级——"测不到"不等于"好转"。

### 风险类型与跨日累积

`src/risk/rules.py` 用"必选维 + 可选池"结构定义三条规则（`sleep_stability` / `social_decline` / `circadian_disruption`），门槛全部从 `config/settings.yaml` 的 `risk.risk_rules` 读。三处非直觉设计：

- `social_decline` 用 **7 天滚动窗取 5 天**而非严格连续（严格连续必跨周末，会被周末的高社交打断）；其 `threshold_ratio` 单独降到 1.0，因为三个必选维的变异系数差一个量级，安全性由"三项 AND + 滚动窗"的结构提供而非单维门槛。
- `circadian_disruption` 是**跨轨**规则（RA/IV 在社交轨、`sleep_onset_clock` 在睡眠轨）；睡眠轨不可用时走降级路径，门槛从 5 天提到 7 天。
- **跨日累积依赖写回**：`quick_judge` 把当天的 `risk_type_qualifies` 写回今天的 `daily_inference/*.json`，次日的持续性统计才读得到。`rules._history_before_today` 刻意排除今天（今天的日志此刻还没写回），今天的达标信号用现算值。破坏这条链会让风险类型永远激活不了——这个 bug 修过一次。

### 数据质量：缺失 ≠ 正常

`validator.py` 的四态 `valid / degraded / insufficient / offline` 贯穿全链路。degraded/insufficient 的日子在持续性统计中被**跳过**（既不累加也不打断，见 `rules._counts_toward_consecutive`），**缺日同理**（两轨全不可用那天不生成日志）；但连续跳过超过 `risk.continuity.max_skip_days`（默认 3）即打断，否则一次长时间离线会把两段无关的偏离粘成一段。质量标记按轨判定：`data_quality` 随推理结果逐轨落盘，单一顶层值会让睡眠轨降级压住社交轨的计数。`copresence_min` 禁止前向填充（"今天有没有人来"取决于子女安排，用昨天填等于伪造社会接触）。单轨失败是正常降级场景，两轨同时不可用才算整体失败。

**`insufficient` 有两个产出点，离线升级判据必须共用一份**：`_process_track` 的
`except DataInsufficientError` 分支（聚合阶段就缺 ≥3 维，**走不到**
`validate_daily_data`）与 `validate_daily_data` 自己那条。判据统一在
`validator.escalate_if_offline`，两边都调它。只写在后者里的后果是
`QUALITY_OFFLINE` **在生产链路里不可达**——实测连跑 7 天全缺仍只记
`insufficient`，四态退化成三态（VALIDATION §11.3）。两份会漂的离线判据比没有
更危险，同 `imputer` 删掉 `check_offline_status` 的理由。

### 预警是事件驱动的，不是每日状态

`judge` 每天判等级，但 `alert.trigger_alert` **只在状态变化时通知**：
开始 / 升级 / 出现新风险类型 / 缓解。同级持续不重复推送，改由周报的
「预警回执」承担告知义务（静默与"系统挂了"必须可区分）。

修复前：E001 60 天 12 次推送对应 4 个真实事件；`PERM_step` 91 天 91 次。
修复后：4 次 / 4 次。事件模型在 `src/risk/alert_state.py`。

改动这一层时必须守住四条：

- **★ 升级穿透一切抑制**（冷却、已确认、同日幂等分支）。纯粹按"同一等级 X 天内
  最多一次"做冷却会在 L2 的冷却窗内吃掉 L3——拿误报换漏报，比没有冷却更糟。
  `decide()` 里升级判定排在所有抑制判据**之前**，
  `tests/test_alert_events.py::TestEscalationBypassesEverything` 是守门员。
- **★ 通道观测游标 `last_processed_day` 跨事件保留**：`apply()` 的**每一条**返回
  路径（含 `RESOLVED`）都要把它带出去，且只增不减。它与事件内的 `last_seen_day`
  是两个东西——前者回答"这条通道处理到哪天了"（据此挡住往回补算），后者回答
  "当前事件断没断"。两者合并过一次，代价是事件一关闭通道就失忆，此后
  **补算任何一个历史高危日都会真的推送给子女**，还把 `started_day` 锚在过去
  （2026-08-02 实测，见 VALIDATION §11.1）。
- **两条事件流独立**：`baseline`（GRU 双轨）与 `depression`（MPDD）各自计数、
  各自冷却、互不压制——理由同"绝不跨轨比较绝对分"。
- **`trigger_alert` 必须收到 `day_key`**。不传会把"今天"盖到历史判定上，
  冷却窗与事件边界全算错——同 `judge_risk_level` 被迫加 `today_key`。

事件边界（隔多久没观测就算两段无关的事件）**按 channel 取**，由
`alert._event_gap_days` 决定，刻意不给独立配置键：
`baseline` 用 `risk.continuity.max_skip_days`（与 `continuity.walk_back_days`
的断段语义一致）；`depression` 用 `depression.assessment.valid_days`——那条通道是
稀疏事件驱动的，几周才有一次合格片段，套用 3 天会让**每次评估都判成新事件、
去重完全失效**。

状态落在 `data/logs/alert_state/{id}.json`（原子写）——
每日批处理的内存态活不过今天，同 `ewma` 的 `freeze_streak`。

新增 `alert:` 段的配置键时必须同步扩 `alert.py:_actions_for` 的**白名单元组**，
否则会被静默丢弃。

`tests/conftest.py` 有一个 **autouse fixture** 把事件状态隔离到 tmp。
删掉它会让测试往真实 `data/logs/` 写状态并跨会话残留——第二次跑同一条用例
就变成"冷却中"，制造不可复现的绿色。

### 抑郁评估是旁路，不是第三条轨

`src/depression/`（MPDD 群体基线）与两条 GRU 个人基线轨**完全独立**。四条硬约束，
由 `tests/test_depression_isolation.py` 静态守卫：

1. `src/baseline/`、`src/risk/`、`src/data_pipeline/`、`src/scheduler/` **不得导入** `src.depression`
2. 不写 `features_*.csv`、不碰 `residual_stats` / `ewma` / `weekly_retrain`
3. 不进 `judge_risk_level` 的 `risk_level` 与 `per_track`
4. MPDD 的依赖（transformers / librosa / av / cv2）**不进 `requirements.txt`**

理由有两层。**尺度不可比**：一个是群体绝对分，一个是个人相对分，
与“绝不跨轨比较绝对分”同理。**更要紧的是防正反馈**：若抑郁分能标记异常日，
异常日会被排除出每周微调 → 个人基线越缩越窄 → 更容易判偏离 → 又佐证抑郁，
自己证明自己。

分层：`status` / `contract` / `aggregate` / `store` 零重依赖；`clip_source` / `runner` /
`mpdd_process` 才碰 subprocess。**周报只准 import `store`**——MPDD 环境挂掉时
周报必须照常渲染。第 4 条靠进程边界实现（`depression.python_bin` 指定独立解释器）。

几个实现约束：
- **不用 MPDD 的 `--run_infer`**：它内部 `check=False`，静默吞掉全部推理失败而父进程
  仍 exit 0。自己逐段驱动才能看见退出码。
- **MPDD 仓不可移动/改名**：OpenFace 的 RUNPATH 是编译进去的绝对路径。
- 子进程必须钉死 `HF_HUB_OFFLINE` / `TORCH_HOME` / `TMPDIR` / `NUMBA_CACHE_DIR`，
  并剔除系统 `PYTHONPATH`（本机 `/opt/ros/humble` 会污染 MPDD 解释器）。
- **★ `_maybe_alert` 对"评估判正常"（`risk_level == 0`）也必须调 `trigger_alert`**，
  不能提前 return——那是**缓解**，要靠它关掉活跃事件并发"过去了"。提前返回的
  后果与 `daily_job` 修过的那条完全同型：事件永远停在最后一次 L2/L3 上，周报的
  「预警回执」无限期显示"情绪状态评估 持续中，第 N 天"。
  只有 `is_displayable == False`（评估失败/无片段/过期）才该原地返回：
  那时"测不到"不等于"好转"，既不开事件也**不关**事件。
- `depression.alert` 刻意为 `false`：当前 checkpoint 在其验证集上对全部样本
  预测同一类别（Macro-F1 0.286），未在本机位校准。**冷却/去重那条阻塞理由
  已解除**（2026-07-31 事件模型 + 2026-08-02 补齐本通道的缓解与断段窗口）。

### 持续性统计的日历回溯是共用的

`src/utils/continuity.py:walk_back_days` 同时被 `rules`（判型）与 `judge`（判级）使用。
两侧曾经不一致——degraded 日在判型时被跳过、判级时却被计入，导致同一段数据
“不激活风险类型却报 L3”。改动持续性语义时必须两侧一起看。

### 实现成熟度

算法层是真的；硬件与外部服务是桩：`adapters/{xiaobeike,camera,ezviz_events}.py` 的 `_read_raw` 全部 `raise NotImplementedError`（只有 mock/file 模式可跑），`alert.py` 只写日志字符串。`config/settings.yaml` 是唯一的运行配置（`paths:` / `scheduler:` 等死配置段已删）。

## 约定与坑

- **确定性优先**：两个验证脚本都固定了随机种子（数据用 `src/utils/seeding.py` 的 crc32，GRU 用 `torch.manual_seed`）。**绝不用内置 `hash()` 派生种子**（受 PYTHONHASHSEED 随机化影响）。不可复现的绿色比红色更危险——曾因未固定 torch 种子给出假的 19/19，掩盖了一个真实的灵敏度缺口。
- **断言要打在真实链路的输出上**。2026-07-31 走查查出的一组缺陷全都活在"全绿"之下，原因是断言打错了层：验证脚本数的是 `status` 流转而非能否出等级；集成测试断言的是 validator 的辅助函数而非 rules 层的实际行为；单测手搓的字典带着生产链路根本没写过的字段。**全绿不等于没问题，要看绿的是什么**（`VALIDATION.md` §8）。
- **注释写"为什么"**：本仓库的注释与 docstring 大量记录"这行代码是被哪条失效链逼出来的"（配置文件里也是）。改动这些区域时保持同样密度，并在文档里同步结论。全仓库文档/注释/提交信息均为中文。
- 提交信息格式：`类型：描述`（修复/文档/重构/测试/新增/配置/数据/脚本/删除）。
- 文档与代码的一致性：`docs/README.md` 的"项目结构"章节已于 2026-07-31 校准到实际文件树；若再次改动目录结构，同步更新该章节。
- 已知未解决（`docs/TODO.md`）：①**MPDD checkpoint 未在本机位校准**（在其验证集上对全部样本预测同一类别，Macro-F1 0.286），故 `depression.alert` 仍置 false，设备到货后必须做域验证；②残差统计未按周内分池，有周末效应的维 z 分被系统性压小；③GRU 固定 7 天窗对持续性变化钝感，异常持续到第 3 天后输入窗被异常日填满、残差收缩。
