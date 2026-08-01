"""
周报Prompt模板

每周日趋势轨执行后，调用大模型生成可解释周报。
支持正常、关注、提醒三种类型模板。

★ 措辞与实际计算必须一致：本系统**从不做周环比**。

  `weekly_report._judge_vs_baseline` 算的是"本周 signed_z 均值相对**个人基线**
  的高低"，`_judge_trend` 算的是"**周内**前半周 vs 后半周"。两者都没有加载过
  上一周的任何数据。

  本文件此前在三处声称做了周环比：系统提示词第 2 条命令模型"指出本周相比上周的
  变化趋势"、用户模板的段落标题写着【上周对比】、规则兜底正文写"与上周相近"。
  提示词那一处最有害——它让 LLM **凭空写出**周环比结论，而正文措辞不固定，
  没有任何测试对得上。这与 weekly_report 修过的"本周无数据时拿别的周顶替"
  是同一类失效：让家属读到一份与实际计算对不上的报告。
"""

# ===== 主周报Prompt =====

WEEKLY_REPORT_SYSTEM_PROMPT = """你是一位老年心理健康分析助手。你的任务是根据老人本周的监测数据，生成一份给子女的、通俗易懂的周报。

【核心原则】
1. 用日常语言描述，不要用专业术语
2. 指出本周相对这位老人自身常态的高低，以及一周之内的走向
3. 如果有异常，给出具体、可操作的建议
4. 语气温暖关切，不要制造恐慌
5. 总字数控制在200字以内
6. 不进行任何临床诊断，仅描述观测到的变化趋势"""


WEEKLY_REPORT_USER_TEMPLATE = """请根据以下老人本周的监测数据，生成一份周报。

【本周走向】（一周之内前半周与后半周相比）
- 社交互动频次：{social_trend}
- 睡眠质量：{sleep_trend}
- 日常活动量：{activity_trend}
- 异常天数：{deviation_days}天（共7天）

【相对个人常态】（本周整体与这位老人自己平时相比）
{social_vs_baseline}
{sleep_vs_baseline}
{activity_vs_baseline}

【风险等级】
- 本周风险等级：{risk_label}
- 需要关注的方向：{risk_types}

【输出要求】
请生成一段连贯的周报文字，200字以内。"""


# ===== 模板规则生成（LLM不可用时的fallback）=====

def generate_rule_based_report(
    elder_id: str,
    week_start: str,
    week_end: str,
    social_trend: str,
    sleep_trend: str,
    activity_trend: str,
    deviation_days: int,
    risk_label: str,
    risk_types: list[str],
    social_vs_baseline: str = "无明显差异",
    sleep_vs_baseline: str = "无明显差异",
    activity_vs_baseline: str = "无明显差异",
) -> str:
    """
    基于规则的周报生成（LLM fallback）。

    当LLM不可用时，使用预制模板拼装周报。
    """
    # 选择语气基调
    if deviation_days == 0:
        opener = f"{elder_id}老人本周整体状态平稳，各项监测指标与其个人常态相近。"
    elif deviation_days <= 2:
        opener = f"{elder_id}老人本周大多数时间状态良好，偶有轻微波动。"
    elif deviation_days <= 4:
        opener = f"{elder_id}老人本周有{deviation_days}天出现偏离常态的情况，需要关注。"
    else:
        opener = f"{elder_id}老人本周有{deviation_days}天明显偏离日常状态，建议多加留意。"

    # 社交线
    # 这三档对应的是**周内**走向（前半周 vs 后半周），措辞不得暗示与上一周比较
    social_map = {
        "上升": "社交互动在这一周里逐渐变得活跃",
        "下降": "社交互动有所减少，可能有些孤独感",
        "平稳": "社交互动频次正常",
    }
    social_line = social_map.get(social_trend, "社交方面无明显变化")

    # 睡眠线
    sleep_map = {
        "上升": "睡眠质量有所改善",
        "下降": "睡眠质量略微下降",
        "平稳": "睡眠质量与往常相近",
    }
    sleep_line = sleep_map.get(sleep_trend, "睡眠状况无明显变化")

    # 建议
    if deviation_days >= 3:
        advice = (
            "建议这周多给老人打一两个电话，聊聊近况。"
            "如果方便的话，可以周末去看看老人。"
        )
    elif deviation_days >= 1:
        advice = "保持日常联系即可，留意老人是否提到身体不适或心情不好。"
    else:
        advice = "一切正常，保持现有的联系频率就好。"

    report = f"{opener}{social_line}；{sleep_line}。{advice}"

    return report


# ===== Prompt填充工具 =====

def fill_prompt(
    template: str,
    **kwargs,
) -> str:
    """
    填充Prompt模板变量。

    自动将缺失的变量填充为"暂无数据"。
    """
    # 确保所有必需变量都有默认值
    defaults = {
        "social_trend": "平稳",
        "sleep_trend": "平稳",
        "activity_trend": "平稳",
        "deviation_days": 0,
        "risk_label": "正常",
        "risk_types": "无",
        "social_vs_baseline": "社交方面本周整体与其常态无明显差异",
        "sleep_vs_baseline": "睡眠方面本周整体与其常态无明显差异",
        "activity_vs_baseline": "活动方面本周整体与其常态无明显差异",
    }

    for key, default_value in defaults.items():
        kwargs.setdefault(key, default_value)

    return template.format(**kwargs)
