"""
周报Prompt模板

每周日趋势轨执行后，调用大模型生成可解释周报。
支持正常、关注、提醒三种类型模板。
"""

# ===== 主周报Prompt =====

WEEKLY_REPORT_SYSTEM_PROMPT = """你是一位老年心理健康分析助手。你的任务是根据老人本周的监测数据，生成一份给子女的、通俗易懂的周报。

【核心原则】
1. 用日常语言描述，不要用专业术语
2. 指出本周相比上周的变化趋势
3. 如果有异常，给出具体、可操作的建议
4. 语气温暖关切，不要制造恐慌
5. 总字数控制在200字以内
6. 不进行任何临床诊断，仅描述观测到的变化趋势"""


WEEKLY_REPORT_USER_TEMPLATE = """请根据以下老人本周的监测数据，生成一份周报。

【本周数据】
- 情绪低落的日占比趋势：{sad_trend}
- 社交互动频次变化：{social_trend}
- 睡眠质量变化：{sleep_trend}
- 日常活动量变化：{activity_trend}
- 异常天数：{deviation_days}天（共7天）

【上周对比】
{sad_week_change}
{social_week_change}
{sleep_week_change}
{activity_week_change}

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
    sad_trend: str,
    social_trend: str,
    sleep_trend: str,
    activity_trend: str,
    deviation_days: int,
    risk_label: str,
    risk_types: list[str],
    sad_week_change: str = "无明显变化",
    social_week_change: str = "无明显变化",
    sleep_week_change: str = "无明显变化",
    activity_week_change: str = "无明显变化",
) -> str:
    """
    基于规则的周报生成（LLM fallback）。

    当LLM不可用时，使用预制模板拼装周报。
    """
    # 选择语气基调
    if deviation_days == 0:
        opener = f"{elder_id}老人本周整体状态平稳，各项监测指标与上周相近。"
    elif deviation_days <= 2:
        opener = f"{elder_id}老人本周大多数时间状态良好，偶有轻微波动。"
    elif deviation_days <= 4:
        opener = f"{elder_id}老人本周有{deviation_days}天出现偏离常态的情况，需要关注。"
    else:
        opener = f"{elder_id}老人本周有{deviation_days}天明显偏离日常状态，建议多加留意。"

    # 情绪线
    sad_map = {
        "上升": "情绪低落的天数有所增加",
        "下降": "情绪状态相比上周有好转",
        "平稳": "情绪状态总体平稳",
    }
    sad_line = sad_map.get(sad_trend, "情绪状态无明显变化")

    # 社交线
    social_map = {
        "上升": "社交互动比上周更活跃",
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

    report = f"{opener}{sad_line}；{social_line}；{sleep_line}。{advice}"

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
        "sad_trend": "平稳",
        "social_trend": "平稳",
        "sleep_trend": "平稳",
        "activity_trend": "平稳",
        "deviation_days": 0,
        "risk_label": "正常",
        "risk_types": "无",
        "sad_week_change": "情绪方面与上周相比无明显变化",
        "social_week_change": "社交方面与上周相比无明显变化",
        "sleep_week_change": "睡眠方面与上周相比无明显变化",
        "activity_week_change": "活动方面与上周相比无明显变化",
    }

    for key, default_value in defaults.items():
        kwargs.setdefault(key, default_value)

    return template.format(**kwargs)
