"""
抑郁评估旁路通道（MPDD 群体基线）

本包把外部专用模型 MPDD-AVG（`/home/zhousenyu/project/MPDD-AVG-2026/`）接进来，
提供第三条监测线：从一段录像里截取"有正脸 + 有连续语音"的片段，跑多模态推理，
给出抑郁等级与 PHQ-9 估计。

这个口子是当初留的：2026-07-29 提交 563b38c「健康特征 10→6 维，移除抑郁风险类型」
的提交信息写的就是"抑郁判断改由外部专用模型接入"，`judge.build_mpdd_evidence`
留下的单向证据契约（GRU → MPDD）也是为它准备的。

★★ 硬约束：本包是单向只读的旁路输出，不参与个人基线的任何环节 ★★

    1. `src/baseline/`、`src/risk/`、`src/data_pipeline/`、`src/scheduler/`
       **不得导入本包**（由 tests/test_depression_isolation.py 静态守卫）
    2. 不写 `features_*.csv`、不碰 `residual_stats` / `ewma` / `weekly_retrain`
    3. 输出不进入 `judge_risk_level` 的 `risk_level`、`per_track`
    4. 分数绝不与 `anomaly_score` 做任何算术

为什么要立这四条：

  **尺度不可比**。MPDD 是群体绝对基线（"跟别人比你有多抑郁"），两条 GRU 轨是
  个人相对基线（"跟你自己平时比"）。把它们相加/平均/互相印证，与本仓库
  "绝不跨轨比较绝对分"是同一类错误——睡眠轨和社交轨这两条自家的轨都不敢比
  （维度数与权重和不同，残差尺度不可比），何况一个训练目标完全不同的外部模型。

  **更要紧的是防正反馈**。若抑郁分能标记异常日，就会形成自我强化的闭环：
      抑郁判高 → 该日标为异常 → 异常日被排除出每周微调 → 个人基线越缩越窄
      → 更容易判偏离 → 偏离又被拿去佐证抑郁 → ……
  这是"系统会习惯异常"那一族缺陷（VALIDATION 缺陷④⑤⑥）的镜像版本，只是方向
  相反：那边是基线学会异常导致漏报，这边是基线被判定污染导致误报自证。
  `judge.build_mpdd_evidence` 的契约注释里写的"本系统不消费 MPDD-AVP 输出"，
  防的就是这个。

分层（决定了谁能 import 谁）：

    status / contract / aggregate / store   零重依赖，只用标准库
    clip_source / runner                    碰 subprocess 与外部仓

  `src/report/weekly_report.py` **只准 import 前四个**。MPDD 环境（transformers /
  librosa / av / cv2 / OpenFace 二进制，几个 G 的权重）挂掉时周报必须照常渲染——
  这与"单轨失败是正常降级、两轨同时不可用才算整体失败"是同一条原则：
  抑郁这条线不该有能力拖垮另外两条。

  同理，MPDD 的依赖**刻意不进 `requirements.txt`**。本仓刚因为 pyaudio 缺
  portaudio 头文件导致 `pip install -r requirements.txt` 整体中止、torch 一个都
  装不上（见 requirements.txt 尾部注释），教训是新鲜的。隔离靠的是进程边界：
  `runner.py` 用 `config.depression.python_bin` 指定的独立解释器 subprocess 调用。
"""
