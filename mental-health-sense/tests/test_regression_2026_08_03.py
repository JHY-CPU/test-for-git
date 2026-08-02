"""2026-08-03 第四轮走查：周报与预警出口的三个「代码与自身设计意图不符」缺陷。

这一轮的缺陷是**核对文档时发现的**，不是测试发现的——因为 501 条用例全绿。
共同特征仍是断言打在错的层（同 VALIDATION §8/§9/§11），但这次的形态更具体：

  **守门用例不传 config，于是只测到了内置默认，测不到真实配置下的行为。**

    - `test_regression_2026_08_01.py:88` 断言缓解会通知社区网格员，
      但它 `trigger_alert(...)` 不传 config → `_resolve_actions_for` 的配置合并段
      整段跳过 → 绿。而生产侧两个调用方都传 `load_config()`，走的是有缺陷的分支。
    - `test_integration.py:511` 是全仓唯一调 `generate_weekly_report` 的用例，
      它传了 `config=load_config()` → 恰好绕开了"脚本入口不传 config"这条真实路径。

  第三个缺陷（周报声称做过周环比）则是**根本没有任何测试看过产物文案**。

所以本文件的断言全部落在：
  - 生产入口 `generate_weekly_report(config=None)` 与 `scripts/run_weekly_pipeline.py`
  - 生产入口 `trigger_alert(config=load_config())`
  - **产出物本身**（周报 Markdown 全文）

预警状态由 conftest 的 autouse fixture 隔离到 tmp；需要真实 data/ 布局的用例
用 conftest 的 `sandbox`（它同时把 data/logs/ 与 config/ 一起隔离）。
"""

import json
from datetime import datetime, timedelta

import pytest

ELDER = "W001"
WEEK_START = "2026-08-02"
WEEK_END = "2026-08-08"

# 周报里凡是出现这些词，就是在声称做了本系统根本没做的周环比
WEEK_OVER_WEEK_WORDS = ("上周",)


def _day_result(day_key: str, *, sleep_z: float = -0.2, social_z: float = -0.3) -> dict:
    """一条结构与生产链路一致的每日推理日志。

    字段照 `inference.infer_track` 的真实返回构造——本仓 VALIDATION §8 记过教训：
    单测手搓的字典带着生产链路根本没写过的字段，断言全绿而缺陷活得好好的。
    """
    def track(z: float, names: list[str]) -> dict:
        return {
            "anomaly_score": 0.82,
            "static_threshold": 1.40,
            "ewma_threshold": 1.35,
            "dynamic_threshold": 1.35,
            "is_deviation": False,
            "signed_residuals": {n: z for n in names},
            "abs_residuals": {n: abs(z) for n in names},
            "signed_z": {n: z for n in names},
            "signed_available": True,
            "data_quality": "valid",
            "status": "success",
        }

    return {
        "elder_id": ELDER,
        "day_key": day_key,
        "sleep": track(sleep_z, ["sleep_efficiency", "waso_min", "sleep_onset_clock"]),
        "social": track(social_z, ["copresence_min", "activity_counts", "rar_amplitude"]),
        "track_quality": {"sleep": "valid", "social": "valid"},
        "track_statuses": {"sleep": "success", "social": "success"},
        "is_deviation": False,
        "consecutive_deviation_days": 0,
        "status": "success",
    }


def _seed_week(elder_id: str = ELDER) -> None:
    """把整周 7 天的推理日志写进（已被 sandbox 重定向的）data/logs/。"""
    from src.utils.io import save_daily_result

    start = datetime.strptime(WEEK_START, "%Y-%m-%d")
    for offset in range(7):
        day_key = (start + timedelta(days=offset)).strftime("%Y-%m-%d")
        save_daily_result(elder_id, day_key, _day_result(day_key))


def _seed_assessment(elder_id: str = ELDER, day_key: str = "2026-08-05") -> None:
    """存一份"已评估"的抑郁契约。sandbox 已把 get_log_dir 重定向到 tmp，
    所以这里走真实的路径推导，不额外 monkeypatch——路径推导必须只有一处。"""
    from src.depression.contract import build_contract
    from src.depression.store import save_assessment

    save_assessment(build_contract(
        elder_id=elder_id, day_key=day_key, status="assessed",
        aggregated={"level": "轻度", "phq9_median": 7.5, "phq9_spread": 1.2,
                    "class_probs": {"正常": 0.3, "轻度": 0.6, "重度": 0.1}},
        evidence={"n_clips": 3, "n_rejected": 0,
                  "source_duration_sec": 3600, "clips": []},
        provenance={"checkpoint": "x.pth", "description_sha256": "abc",
                    "device": "cuda", "batch_size": 64,
                    "frame_sample_rate": 1, "mpdd_git_rev": "deadbee"},
        valid_days=30,
    ))


# ===== 缺陷一：周报的「情绪状态评估」整节静默消失 =====

class TestDepressionSectionSurvivesMissingConfig:
    """★ 修复前实测：

        generate_weekly_report(..., config=load_config()) → 有「情绪状态评估」小节
        generate_weekly_report(..., config=None)          → **整节返回 ''**

    而 `scripts/run_weekly_pipeline.py` 恰恰不传 config。被吞掉的不只是分数，
    还有 `contract.py:113-118` 明令"字段在，周报就必须渲染"的那句校准警示语——
    模型在其验证集上对全部样本预测同一类别，家属却看不到这个前提。

    这与本仓"测不到不等于正常"是同一条原则的展示层版本：
    配置读不到，不该让整节**静默消失**成"这周没评估"。
    """

    def test_不传config时抑郁节仍须渲染并带警示语(self, sandbox):
        from src.report.weekly_report import generate_weekly_report

        _seed_week()
        _seed_assessment()

        report = generate_weekly_report(
            elder_id=ELDER, week_start=WEEK_START, week_end=WEEK_END,
            use_llm=False, config=None,
        )

        assert "情绪状态评估" in report, "config 为 None 时抑郁小节被静默吞掉"
        assert "未在本机位" in report or "不可作为任何判断依据" in report, \
            "校准警示语必须随小节一起出现，否则家属会把未校准的分当结论"

    def test_周轨脚本跑出的周报必须含抑郁小节(self, sandbox, monkeypatch):
        """从最外层入口进。断言打在**磁盘上的产物**而不是函数返回值——
        这个缺陷的全部危害就在于家属最终读到的那份文件里少了一节。"""
        import sys

        import scripts.run_weekly_pipeline as entry
        from src.utils.io import get_log_dir

        _seed_week()
        _seed_assessment()

        # 周轨的窗口终点是"昨天"，把时钟钉在 WEEK_END 的次日，让窗口正好覆盖这一周
        import src.scheduler.weekly_job as weekly_job

        class _FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.strptime(WEEK_END, "%Y-%m-%d") + timedelta(days=1)

        monkeypatch.setattr(weekly_job, "datetime", _FixedDatetime)
        monkeypatch.setattr(sys, "argv",
                            ["run_weekly_pipeline.py", "--elder", ELDER, "--no-retrain"])

        entry.main()

        report_path = get_log_dir("weekly_reports") / f"{ELDER}_{WEEK_START}.md"
        assert report_path.is_file(), "周轨脚本没有产出这一周的周报"
        assert "情绪状态评估" in report_path.read_text(encoding="utf-8")


# ===== 缺陷二：3 级事件缓解时，社区网格员收不到"结束了" =====

class TestResolveNotifiesPeakRecipients:
    """★ 修复前实测（`_resolve_actions_for` 是纯函数，可直接对拍）：

        prev_level=3, config=None          → ['children', 'community_worker']
        prev_level=3, config=load_config() → ['children']          ← 缺陷

    根因是顺序：先按峰值等级定收件人，随后被 `alert.events.resolve.notify`
    无条件覆盖，而 settings.yaml 里恰好写了和默认相同的 ["children"]
    （照配置注释"要覆盖时写全 action / notify"抄的，显然不是有意压掉网格员）。

    后果正是该函数注释里明写要防的那个场景：
    **网格员被叫来过，但没人告诉他结束了。**
    """

    def test_L3事件缓解要通知曾惊动过的社区网格员(self):
        from src.risk.alert import trigger_alert
        from src.utils.io import load_config

        config = load_config()

        started = trigger_alert(ELDER, 3, [{"risk_key": "sleep_stability"}],
                                config=config, day_key="2026-08-10")
        assert "push_to_community_worker" in started["actions"]

        resolved = trigger_alert(ELDER, 0, [], config=config, day_key="2026-08-11")

        assert resolved["transition"] == "resolved"
        assert resolved["alerted"] is True
        assert "push_to_children" in resolved["actions"]
        assert "push_to_community_worker" in resolved["actions"], \
            "L3 惊动过网格员，缓解就必须也告诉他，不能被 resolve.notify 覆盖掉"
        # 缓解是好消息：降一档强度，绝不响铃
        assert "force_ring" not in resolved["actions"]

    def test_L2事件缓解不得惊动网格员(self):
        """反向守卫：并集只并入**峰值等级**的收件人，不是无脑加 community_worker。"""
        from src.risk.alert import trigger_alert
        from src.utils.io import load_config

        config = load_config()
        trigger_alert(ELDER, 2, [{"risk_key": "sleep_stability"}],
                      config=config, day_key="2026-08-10")
        resolved = trigger_alert(ELDER, 0, [], config=config, day_key="2026-08-11")

        assert "push_to_children" in resolved["actions"]
        assert "push_to_community_worker" not in resolved["actions"]

    def test_峰值等级取自peak_level而非最后一次通知等级(self):
        """L3 之后降到 L2 又报一次新类型，`last_notified_level` 会被覆盖成 2，
        只有 `peak_level` 记得曾经到过 3。收件人必须跟着 peak 走。"""
        from src.risk.alert import trigger_alert
        from src.utils.io import load_config

        config = load_config()
        trigger_alert(ELDER, 3, [{"risk_key": "sleep_stability"}],
                      config=config, day_key="2026-08-10")
        # 降到 L2 且出现新类型 → 会再发一次通知，把 last_notified_level 覆盖成 2
        trigger_alert(ELDER, 2,
                      [{"risk_key": "sleep_stability"}, {"risk_key": "social_decline"}],
                      config=config, day_key="2026-08-14")
        resolved = trigger_alert(ELDER, 0, [], config=config, day_key="2026-08-15")

        assert "push_to_community_worker" in resolved["actions"]

    def test_损坏的peak_level不得让缓解崩溃(self, _isolate_alert_state):
        """`load_state` 的合并只过滤键名、不校验值域（alert_state.py:186），
        手改过或别的版本写的状态文件里出现 peak_level=7 时，
        `AlertLevel(7)` 会抛 ValueError，而 alert.py:296 那行没有第二层 try。

        `peak_level` 此前全仓零测试覆盖——对比 `last_processed_day` 的兜底
        有 test_regression_2026_08_02.py:98 专门守着。
        """
        from src.risk.alert import trigger_alert
        from src.utils.io import load_config

        (_isolate_alert_state / f"{ELDER}.json").write_text(json.dumps({
            "schema_version": "1.2.0",
            "elder_id": ELDER,
            "channels": {
                "baseline": {
                    "active": True, "level": 3, "risk_keys": ["sleep_stability"],
                    "started_day": "2026-08-10", "last_seen_day": "2026-08-10",
                    "last_processed_day": "2026-08-10",
                    "last_notified_day": "2026-08-10", "last_notified_level": 3,
                    "notify_count": 1, "acknowledged": False, "acknowledged_at": None,
                    "peak_level": 7,
                },
                "depression": {},
            },
        }), encoding="utf-8")

        resolved = trigger_alert(ELDER, 0, [], config=load_config(),
                                 day_key="2026-08-11")

        assert resolved["transition"] == "resolved"
        assert "push_to_community_worker" in resolved["actions"], \
            "越界的 peak_level 应被钳到 L3，而不是让整条日管道崩掉"


# ===== 缺陷三：周报声称做过周环比，实际从没加载过上一周 =====

class TestReportDoesNotClaimWeekOverWeek:
    """`_judge_vs_baseline` 算的是"本周 signed_z 均值 vs 个人基线（±1.0）"，
    `_judge_trend` 算的是"周内前半周 vs 后半周"。两者都与上一周无关，
    而周报表格列名、LLM 系统提示词、用户模板段落标题、规则兜底正文
    四处都在说"与上周相比"。

    提示词那一处最隐蔽也最有害：它**命令模型**"指出本周相比上周的变化趋势"，
    于是 LLM 正文会凭空写出周环比结论，而措辞不固定、没人对得上。
    """

    def test_周报正文与表格都不得出现上周字样(self, sandbox):
        from src.report.weekly_report import generate_weekly_report

        _seed_week()
        report = generate_weekly_report(
            elder_id=ELDER, week_start=WEEK_START, week_end=WEEK_END,
            use_llm=False, config=None,
        )

        for word in WEEK_OVER_WEEK_WORDS:
            assert word not in report, f"周报声称做了周环比（出现『{word}』），但系统从没加载过上一周"

    def test_LLM提示词不得命令模型做周环比(self):
        """规则模板是 fallback，设了 DEEPSEEK_API_KEY 时走的是 LLM 那条路——
        提示词里的谎言不会出现在本仓的测试产物里，但会出现在**用户的周报**里。"""
        from src.report.templates import (
            WEEKLY_REPORT_SYSTEM_PROMPT,
            WEEKLY_REPORT_USER_TEMPLATE,
            fill_prompt,
        )

        for word in WEEK_OVER_WEEK_WORDS:
            assert word not in WEEKLY_REPORT_SYSTEM_PROMPT
            assert word not in WEEKLY_REPORT_USER_TEMPLATE
            # 默认文案同样会被送进模型
            assert word not in fill_prompt(WEEKLY_REPORT_USER_TEMPLATE)

    def test_基准日早于事件起始日时不得渲染负天数(self, sandbox):
        """端到端验证时看到的第四个问题：补生成历史周报时，活跃事件可能是
        **之后**才开始的，`event_age_days` 于是算出负数，周报写着
        "持续中，第 -7 天"。负天数对家属没有意义，应当整段略去。"""
        from src.report.weekly_report import _format_alert_receipts
        from src.risk.alert import trigger_alert
        from src.risk.alert_state import event_age_days
        from src.utils.io import load_config

        config = load_config()
        trigger_alert(ELDER, 2, [{"risk_key": "sleep_stability"}],
                      config=config, day_key="2026-08-08")

        # 基准日早于事件起始日
        assert event_age_days({"started_day": "2026-08-08"}, "2026-07-31") is None
        # 同日算第 1 天、次日第 2 天（含首日的语义不变）
        assert event_age_days({"started_day": "2026-08-08"}, "2026-08-08") == 1
        assert event_age_days({"started_day": "2026-08-08"}, "2026-08-09") == 2

        receipts = _format_alert_receipts(ELDER, "2026-07-31", config)
        assert "持续中" in receipts
        assert "第 -" not in receipts, "周报渲染出了负的事件天数"

    def test_规则兜底正文不得出现上周字样(self):
        from src.report.templates import generate_rule_based_report

        for deviation_days, social_trend in ((0, "平稳"), (1, "上升"), (5, "下降")):
            report = generate_rule_based_report(
                elder_id=ELDER, week_start=WEEK_START, week_end=WEEK_END,
                social_trend=social_trend, sleep_trend="平稳", activity_trend="平稳",
                deviation_days=deviation_days, risk_label="正常", risk_types=[],
            )
            for word in WEEK_OVER_WEEK_WORDS:
                assert word not in report
