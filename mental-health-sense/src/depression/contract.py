"""抑郁评估契约：schema 定义、组装与校验

落盘位置：`data/logs/depression/{elder_id}_{day_key}.json`

这是与 `data/logs/mpdd_evidence/` 对称的**反方向**产物：

    GRU  → mpdd_evidence/   （judge.build_mpdd_evidence，已有）
    MPDD → depression/      （本模块，新增）

两个方向都是单向文件契约，谁都不 import 谁。反方向的消费者只有周报，而且**只读不用**
——不进判级、不进模型、不进微调（见 src/depression/__init__.py 的四条硬约束）。

★ provenance 里三个字段不是可选装饰，缺一个就会出现"分数莫名其妙变了但查不出原因"：

  `description_sha256`
      MPDD 的 A-V+P 模型吃的是"老人介绍"文本的 1024 维 RoBERTa 嵌入
      （infer_elder_depression.py:208-235）。**这份文本改一个字，分数就变，而与
      老人的实际状态毫无关系。** 它必须像 scaler 一样冻结，并留指纹供事后核对。
      这与"scaler 建档后永久冻结"是同一条防线：把评估基准钉死，否则今天和上个月
      的分不可比。

  `device` / `batch_size` / `frame_sample_rate`
      实测：同机同版本下 MPDD 推理逐位可复现（无 RNG、全程 .eval()、无反向传播），
      但这三项任意一个变了数值就会变——batch_size 改变 cuDNN 的 kernel 选择，
      frame_sample_rate 改变插值到 target_t=128 之前的序列长度，CPU/GPU 则是
      两套浮点实现。注意 infer_elder_depression.py:79-81 在 CUDA 不可用时会
      **静默降级到 CPU**（只打一行 WARN 到 stderr），所以不能假设配置里写的
      device 就是实际用的——必须记录真实生效的那个。

      本仓库"确定性优先、不可复现的绿色比红色更危险"这条约定在这里同样适用。
"""

from datetime import datetime, timedelta

SCHEMA_VERSION = "1.0.0"

# 与 judge.build_mpdd_evidence 的 note 对称：两侧都把约束写进产物本身，
# 这样即使有人只拿到一个 JSON 文件、没读代码，也知道它不能拿去做什么。
CONTRACT_NOTE = (
    "群体基线绝对评估，与个人基线偏离分不可比、不得合并或相互印证。非临床诊断。"
)

DATE_FMT = "%Y-%m-%d"

REQUIRED_TOP_LEVEL = (
    "schema_version",
    "elder_id",
    "day_key",
    "assessed_at",
    "valid_until",
    "status",
    "result",
    "evidence",
    "provenance",
    "calibration",
    "note",
)


def compute_valid_until(day_key: str, valid_days: int) -> str:
    """有效期终点（含当日）。

    过期后展示层一律显示"暂无最新评估"，**绝不拿旧分顶替今天**——这正是
    weekly_report 修过的那个缺陷（本周无数据时静默拿别的周顶替，标题与内容对不上）
    在抑郁通道上的同型风险。抑郁评估天然稀疏（要"有正脸 + 有连续语音"的片段，
    独居老人可能几周才有一次），比周报更容易踩这个坑。
    """
    if valid_days < 1:
        raise ValueError(f"valid_days 必须 ≥ 1，得到 {valid_days}")
    base = datetime.strptime(day_key, DATE_FMT)
    return (base + timedelta(days=valid_days - 1)).strftime(DATE_FMT)


def build_contract(
    elder_id: str,
    day_key: str,
    status: str,
    aggregated: dict | None,
    evidence: dict,
    provenance: dict,
    valid_days: int,
    assessed_at: str | None = None,
    low_confidence_reasons: list[str] | None = None,
    calibration_warning: str | None = None,
) -> dict:
    """组装契约字典。

    Args:
        status: src.depression.status 里的状态常量
        aggregated: aggregate_clip_results 的 "result" 块；无结论时为 None
        evidence: {"n_clips", "n_rejected", "source_duration_sec", "clips": [...]}
        provenance: {"checkpoint", "description_sha256", "device",
                     "batch_size", "frame_sample_rate", "mpdd_git_rev"}
        assessed_at: 片段的**实际时间**（ISO8601）。默认取 day_key 当天。
            刻意不用 datetime.now()：跑推理的时刻可能是次日凌晨批处理，
            甚至是几天后的补算，用它会让"最近一次评估"的时间轴整体漂移。
    """
    from src.depression.status import validate_status

    validate_status(status)

    return {
        "schema_version": SCHEMA_VERSION,
        "elder_id": elder_id,
        "day_key": day_key,
        "assessed_at": assessed_at or f"{day_key}T00:00:00+08:00",
        "valid_until": compute_valid_until(day_key, valid_days),
        "status": status,
        "result": aggregated,
        "low_confidence_reasons": low_confidence_reasons or [],
        "evidence": evidence,
        "provenance": provenance,
        "calibration": {
            # 真机位校准做完之前恒为 false。这个字段存在的意义是**强制展示层
            # 面对它**：当前 checkpoint 在验证集上对全部 9 个样本预测同一类别
            # （混淆矩阵 [[6,0,0],[2,0,0],[0,1,0]]，Macro-F1 0.286），
            # 拿它当结论会误导家属。字段在，周报就必须渲染警示语。
            "validated_on_site": False,
            "warning": calibration_warning or (
                "模型在其验证集上对全部样本预测同一类别，且未在本机位/本人身上校准，"
                "结论不可作为任何判断依据"
            ),
        },
        "note": CONTRACT_NOTE,
    }


def validate_contract(payload: dict) -> dict:
    """校验契约结构，缺字段立即报错。

    为什么要主动校验而不是等下游 KeyError：契约是跨进程、跨仓库的产物，
    写入方（runner）和读取方（周报）在不同的时间、可能不同的版本下运行。
    结构漂了要在**读到的那一刻**炸，而不是在周报渲染到一半时抛一个
    看不出所以然的 KeyError。
    """
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in payload]
    if missing:
        raise ValueError(f"抑郁契约缺少字段：{missing}")

    from src.depression.status import validate_status

    validate_status(payload["status"])

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        # 主版本不同直接拒绝；同主版本的小升级放行（新增字段应当向后兼容）。
        if str(version).split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(
                f"抑郁契约主版本不兼容：文件 {version}，当前 {SCHEMA_VERSION}"
            )

    return payload
