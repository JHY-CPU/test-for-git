"""持续性统计的日历回溯（判级与判型共用的单一实现）

★ 为什么抽出来

  "从今天往前逐个自然日回溯、跳过不可信的日子、连续跳过太久就打断"这套语义，
  原本只存在于 rules.py（风险**类型**的持续天数）。风险**等级**那一侧
  （judge._judge_single_track 数 consecutive）压根没有这套逻辑，只是把窗口里的
  记录从后往前数，遇到第一个非偏离日停止。

  两者不一致的后果是实打实的误报，而且落在代价最高的那一级：

    连续 5 天社交偏离，其中第 2、4 天 C6c 掉线 → copresence_min 缺失 → degraded。
      rules 正确跳过 D2/D4 → 滚动窗只数到 3 < 5 → 「社会连接减弱」不激活 ✔
      judge 把 5 天全数上 → consecutive=5 → avg_severity 过门槛 → **L3 严重** ✘
    L3 = 短信 + 强制响铃 + 社区网格员介入。系统里代价最高的动作，恰好由设计上
    明令"不计入"的那些天凑出来；而且因为没有活跃风险类型，建议文案还会退化成
    通用的"连续5天异常，建议尽快联系老人"。

  反向的漏报同样存在：真实的 5 天睡眠恶化里夹一个 degraded 日，consecutive 在
  那天归零，L3 永远到不了。

  所以两侧必须用同一个回溯器。这里只放"怎么走"，"这天算不算数"由调用方以
  counts_fn 传入——两侧判据不同（rules 按规则必需的轨判，judge 按单轨自己判），
  但走法必须一致。
"""

from datetime import datetime, timedelta
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")

DATE_FMT = "%Y-%m-%d"


def walk_back_days(
    index: dict[str, T],
    today_key: str,
    max_steps: int,
    max_skip: int,
    counts_fn: Callable[[T], bool],
    include_today: bool = False,
) -> Iterator[T]:
    """从 today_key 往前逐个自然日回溯，yield 计入统计的那些日子（降序）。

    Args:
        index: {day_key: 记录}
        max_steps: 最多回溯几个自然日（不含今天）
        max_skip: 连续跳过多少天后打断
        counts_fn: 该记录是否计入统计（False → 跳过，既不累加也不打断）
        include_today: 是否先 yield 今天自己（judge 需要，rules 不需要——
            rules 的今天用现算值，日志此刻还没写回 risk_type_qualifies）

    ★ 为什么必须按自然日而不是按记录序号

      load_daily_results 取的是最近 N 个**文件**，不是最近 N 个**自然日**；
      两轨全不可用那天 daily_job 直接跳过推理、连日志都不生成。两者叠加：
      设备离线一段时间后，断裂两端的偏离日会被当成连续日拼起来。
      实测 5 条日志跨越 22 个日历日（中间断 17 天）仍数出 consecutive=5 → L3。

    ★ 缺日与降级日走同一条路径

      两者都是"这天我们不知道"：缺日是压根没测，降级是测得不可信。既不累加
      也不打断，与 validator 的四态设计一致。但不能无上限地跨过去——连续跳过
      超过 max_skip 天就打断，否则一次长时间离线又会把两段无关的偏离粘起来。
      max_skip 默认 3：超过三天没有可信数据，就不该再假装这是同一段状态。
    """
    try:
        cursor = datetime.strptime(today_key, DATE_FMT)
    except (TypeError, ValueError):
        return

    if include_today:
        today = index.get(today_key)
        if today is not None and counts_fn(today):
            yield today

    skipped_in_a_row = 0
    for _ in range(max_steps):
        cursor -= timedelta(days=1)
        key = cursor.strftime(DATE_FMT)
        day_result = index.get(key)

        if day_result is None or not counts_fn(day_result):
            skipped_in_a_row += 1
            if skipped_in_a_row > max_skip:
                return          # 连续跳过太久，不再认为是同一段状态
            continue

        skipped_in_a_row = 0
        yield day_result
