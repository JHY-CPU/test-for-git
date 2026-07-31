"""推理状态枚举与"可评估"判据（单一事实来源）

每日推理对每一轨给出一个 status，判定层据此决定该轨当天能不能参与判级与判型。

    success               GRU 基线已就绪，正常出分
    observation           同上，但仍在冷启动观察期内（当前配置已取消，保留兼容）
    cold_start_fallback   GRU 基线未就绪，走中位数/MAD 稳健滑动基线兜底
    cold_start            基线文件缺失且兜底也无法进行（历史数据不足）
    data_insufficient     当日或近 window 天特征缺失，无法出分

★ 为什么要抽出这个模块

  "哪些 status 算可评估"此前以字面量元组的形式散在 judge.py（两处）、rules.py、
  daily_job.py（两处）里，五份各写各的。结果是 cold_start_fallback 只被加进了
  daily_job 的两处，judge 与 rules 都漏了——兜底轨算出的偏离进不了判定层，
  建档期 35 天完全没有预警能力：实测连续 6 天 anomaly_score 166→58（阈值 3.0）、
  is_deviation 6/6，risk_level 全为 0。检测层是对的，缺口在白名单不一致。

  放在 utils 而不是 baseline/inference.py：rules.py 与 judge.py 都要用它，
  而从 baseline.inference 导入会把 torch 拖进整个 risk 层。本模块零依赖。
"""

STATUS_SUCCESS = "success"
STATUS_OBSERVATION = "observation"
STATUS_COLD_START_FALLBACK = "cold_start_fallback"
STATUS_COLD_START = "cold_start"
STATUS_DATA_INSUFFICIENT = "data_insufficient"

# 该轨当天能参与判级/判型的状态。
#
# cold_start_fallback 必须在内：兜底是为了"消除建档期监测盲区"而写的，
# 它返回带方向的 signed_z 与 is_deviation，判定层没有理由拒收。
# 注意兜底的 anomaly_score 是稳健 z（阈值 = fallback_sigma），与 GRU 轨的残差
# 尺度不可比——所以幅度门槛必须用 severity（分数/自身阈值）而非绝对分，
# 见 judge._severity。
EVALUABLE_STATUSES = frozenset({
    STATUS_SUCCESS,
    STATUS_OBSERVATION,
    STATUS_COLD_START_FALLBACK,
})

# 基线尚未就绪的状态（对外呈现"证据强度较弱"时用）
COLD_START_STATUSES = frozenset({
    STATUS_COLD_START,
    STATUS_COLD_START_FALLBACK,
})


def is_evaluable(status: str | None) -> bool:
    """该轨当天是否可参与风险判定。"""
    return status in EVALUABLE_STATUSES
