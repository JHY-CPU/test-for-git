"""
边缘人形共处检测适配器（copresence_min 的唯一来源）

职责：C6c 视频流 → 逐秒人数 n(t) → copresence_min（画面中人形数 ≥2 的累计分钟）

★ 这是社会连接轨**唯一**的社会接触指标，也是本方案最大的单点风险。
其余 4 维（离家、RA、IV、活动量）测的都不是"跟人在一起"。因此：
  - 该维缺失时整轨降级，且**禁止前向填充**（见 aggregator.NO_FORWARD_FILL_FEATURES）
    ——"今天有没有人来"取决于子女安排，用昨天填今天等于凭空伪造社会接触。
  - 云端人形检测抽样校准从"锦上添花"变成必须执行的日常质检。
  - 摄像头画面 SSIM 突变要触发 ROI 标定失效提醒。

隐私边界（2026-07-31 重新划定，务必区分两件事）：

  **本适配器（GRU 社交轨）：**
  - 视频原始数据**不出户**，边缘设备只上报当日聚合的分钟数
  - **不取音频流、不解码、不落盘** —— 社交轨的 5 维一维未变，
    `copresence_min` 只来自逐秒人数，与声音无关

  **同一台设备上的抑郁评估模块（src/depression/，旁路通道）：**
  - **会录制短音视频片段**并落盘到 `data/raw/video_clips/`，供 MPDD 多模态
    推理使用（它需要音频 MFCC + 画面 ResNet 特征）
  - 只在检测到"有正脸 + 有连续语音"时导出片段，不做全天录像

  ★ 这两条在知情同意书上是**不同条目**，不能合并成一句"我们现在录音了"。
    区别是实质性的：睡眠/孤独的监测**完全不需要**声音，抑郁评估才需要；
    家属若只同意前者，抑郁通道就该整个关掉（`depression.enabled: false`），
    而两条 GRU 轨照常工作。这正是把抑郁做成旁路而非第三条轨的附带收益之一。

30 s 中位数去抖的理由：单帧人形检测在遮挡、侧身、逆光下漏检率不低，
逐帧判定会把一次连续来访切成几十段。中位数滑窗要求"半分钟里多数时间看得见两个人"，
这与"社会接触"的语义也更贴合——擦身而过不算接触。

两类必须承认的误差：
  1. 漏检（低估）：老人与来访者一同走出客厅时共处仍在继续，但摄像头看不到。
     单摄像头无解，文案里只能表述为"客厅可见共处时长"。
  2. 误检（高估）：电视画面里的人、照片、镜面反射被计为人形 → 由 ROI 过滤处理。
"""

import numpy as np

from src.data_pipeline.adapters import SensorAdapter

# 共处判据：画面中人形数 ≥ 此值
COPRESENCE_MIN_PERSONS = 2

# 去抖窗口（秒）与采样率（fps）
SMOOTH_WINDOW_SEC = 30
SAMPLE_FPS = 1

# 云端抽样校准的一致率告警线。
# 已删除（2026-07-31）：CLOUD_CALIBRATION_MAX_FRAMES = 50，零引用——
# 抽样帧数上限属于 _read_raw 接入真机时的实现细节，写在这里只是个未被消费的常量。
CLOUD_CALIBRATION_MIN_AGREEMENT = 0.90

# 访客日判据：共处时长超过此值标 has_visitor
VISITOR_DAY_THRESHOLD_MIN = 30


def smooth_person_counts(per_second_counts, window_sec: int = SMOOTH_WINDOW_SEC) -> np.ndarray:
    """
    对逐秒人数做滑动中位数去抖。

    窗口中心对齐；边界处窗口自动收窄（不做 padding——padding 会在开头结尾
    引入不存在的人数，而来访往往正好发生在时段边缘）。
    """
    counts = np.asarray(per_second_counts, dtype=np.float64).flatten()
    if counts.size == 0:
        return counts
    if np.any(counts < 0):
        raise ValueError("person counts must be non-negative")

    half = max(window_sec // 2, 1)
    smoothed = np.empty_like(counts)
    for i in range(counts.size):
        lo = max(0, i - half)
        hi = min(counts.size, i + half + 1)
        smoothed[i] = np.median(counts[lo:hi])
    return smoothed


def compute_copresence_minutes(
    per_second_counts,
    window_sec: int = SMOOTH_WINDOW_SEC,
    fps: int = SAMPLE_FPS,
) -> dict:
    """
    逐秒人数 → copresence_min。

    Returns:
        {
            "copresence_min": float,
            "copresence_segments": int,   # 分几次共处（周报用）
            "max_persons": int,
            "sampled_seconds": int,
            "has_visitor": bool,
        }
    """
    counts = np.asarray(per_second_counts, dtype=np.float64).flatten()
    if counts.size == 0:
        return {
            "copresence_min": 0.0, "copresence_segments": 0,
            "max_persons": 0, "sampled_seconds": 0, "has_visitor": False,
        }

    smoothed = smooth_person_counts(counts, window_sec)
    is_copresent = smoothed >= COPRESENCE_MIN_PERSONS

    seconds_per_sample = 1.0 / max(fps, 1)
    total_min = float(is_copresent.sum() * seconds_per_sample / 60.0)

    # 段数：从 False 翻到 True 的次数（含开头即为 True 的情形）
    segments = int(np.sum(is_copresent[1:] & ~is_copresent[:-1])) + int(is_copresent[0])

    return {
        "copresence_min": round(total_min, 3),
        "copresence_segments": segments,
        "max_persons": int(counts.max()),
        "sampled_seconds": int(counts.size * seconds_per_sample),
        "has_visitor": total_min > VISITOR_DAY_THRESHOLD_MIN,
    }


def filter_tv_roi_detections(
    detections: list[dict],
    tv_roi: tuple[float, float, float, float] | None,
) -> list[dict]:
    """
    丢弃落在电视屏幕 ROI 内的人形框。

    电视画面里的人、照片、镜面反射会被计为人形，直接抬高 copresence。
    这是 v2.0 电视门控唯一被保留的部分——只需一次静态标定，
    不涉及音频停顿结构那套脆弱判别。

    判据用框中心是否落在 ROI 内，而非 IoU：电视里的人往往只占屏幕一部分，
    IoU 阈值难定且需调参，中心点判据稳定。

    Args:
        detections: [{"bbox": (x1, y1, x2, y2)}]，坐标归一化到 [0,1]
        tv_roi: (x1, y1, x2, y2)；None 表示未标定，不过滤
    """
    if not tv_roi:
        return list(detections or [])

    rx1, ry1, rx2, ry2 = tv_roi
    kept = []
    for det in detections or []:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            kept.append(det)
            continue
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            continue
        kept.append(det)
    return kept


def check_cloud_calibration(edge_counts: list[int], cloud_counts: list[int]) -> dict:
    """
    边缘模型 vs 云端 human/analysis/detect 的一致率校验。

    copresence_min 是社会接触的唯一来源，这条校准是**必须执行的日常质检**，
    不是可选项。一致率 <90% 即触发模型复查告警。

    Returns:
        {"agreement": float, "n_frames": int, "passed": bool, "alert": str | None}
    """
    if len(edge_counts) != len(cloud_counts):
        raise ValueError(
            f"edge/cloud 帧数不匹配: {len(edge_counts)} vs {len(cloud_counts)}"
        )
    n = len(edge_counts)
    if n == 0:
        return {
            "agreement": 0.0, "n_frames": 0, "passed": False,
            "alert": "无校准样本，无法验证边缘模型"
                     "（copresence 是唯一社会接触来源，必须校准）",
        }

    agree = sum(1 for e, c in zip(edge_counts, cloud_counts) if e == c)
    agreement = agree / n

    return {
        "agreement": round(agreement, 4),
        "n_frames": n,
        "passed": agreement >= CLOUD_CALIBRATION_MIN_AGREEMENT,
        "alert": None if agreement >= CLOUD_CALIBRATION_MIN_AGREEMENT else (
            f"边缘人形检测一致率 {agreement:.1%} < {CLOUD_CALIBRATION_MIN_AGREEMENT:.0%}，"
            f"需复查模型或重标电视 ROI"
        ),
    }


class CameraAdapter(SensorAdapter):
    """
    C6c 边缘人形共处检测 → copresence_min。

    Usage:
        adapter = CameraAdapter(mode="mock")
        features = adapter.extract(source="", date="2026-08-01")
    """

    FEATURE_NAMES = ["copresence_min"]

    def __init__(
        self,
        mode: str = "mock",
        tv_roi: tuple[float, float, float, float] | None = None,
    ):
        super().__init__(mode)
        self.tv_roi = tv_roi

    def _read_raw(self, source: str, date: str) -> dict:
        """
        【接入真实设备时实现】边缘取流 → 人形检测 → 当日聚合。

            POST /api/lapp/device/capacity                     → 能力集自检
            POST /api/lapp/v2/live/address/get                 → 取流地址（本地网络）
            POST /api/lapp/intelligence/human/analysis/detect  → 云端校准（每日 ≤50 帧）

        实现要点：
          1. **只解码视频轨，不解码音频轨** —— v2.1 硬性约束
          2. 1 fps 抽帧 → YOLOv8n 或同级轻量模型 → 逐秒人数
          3. filter_tv_roi_detections 丢弃电视画面里的人形
          4. compute_copresence_minutes 做 30 s 中位数去抖
          5. 只向云侧写出当日聚合分钟数，**视频不出户**
          6. 每日 check_cloud_calibration，一致率 <90% 告警
        """
        raise NotImplementedError(
            "边缘人形检测 live 模式尚未接入。依赖 §11 待决问题："
            "边缘算力选型、电视 ROI 现场标定、C6c 是否支持 app_human_detect 实测。"
            "已就绪的部分：去抖、ROI 过滤、共处时长计算、云端校准比对逻辑。"
        )

    def _generate_mock(self, date: str) -> dict:
        """
        生成逐秒人数序列并走真实聚合逻辑。

        周末造更长的共处时段（子女探访），使 mock 数据体现周末效应——
        这是社交轨必须按 is_weekend 分池的原因。
        """
        import random
        from datetime import datetime

        rng = random.Random(f"{date}-camera")
        is_weekend = datetime.strptime(date, "%Y-%m-%d").weekday() >= 5

        n_samples = 16 * 3600                          # 16 小时日间窗口，1 fps
        counts = np.ones(n_samples, dtype=np.float64)  # 平时画面里只有老人

        n_visits = rng.randint(1, 3) if is_weekend else rng.randint(0, 1)
        for _ in range(n_visits):
            duration = rng.randint(1800, 7200) if is_weekend else rng.randint(600, 2400)
            start = rng.randint(0, max(1, n_samples - duration))
            counts[start:start + duration] = 2
            # 掺入单帧漏检，用于验证去抖确实起作用
            for _ in range(duration // 40):
                counts[start + rng.randint(0, duration - 1)] = 1

        return {"copresence_min": compute_copresence_minutes(counts, fps=SAMPLE_FPS)["copresence_min"]}
