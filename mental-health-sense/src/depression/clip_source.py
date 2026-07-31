"""候选片段来源：现在传录像文件，将来接实时流

★ 这个抽象是"现在"和"以后"的接缝，是整个抑郁通道里最需要提前定对的一层。

摄像头设备到货前只能手工传录像；到货后要接 C6c 的实时流。两种情况下**下游完全
一样**——runner 只认一个目录，里面是若干 mp4 加一份 index.json，它不关心片段是
怎么来的。于是设备到货那天做的是换实现，不是重写。

    ClipSource（本模块）
      ├── FileClipSource     现在：给一个 mp4 → 调 extract_clips 截片段
      └── StreamClipSource   以后：环形缓冲 + 实时触发器直接吐片段

这跟适配器层的 mock / file / live 三态是同一个模式（见 adapters/__init__.py），
只是抽象的粒度从"特征值"挪到了"片段"。

—— 设备到货后的采集架构（写在这里备查，实现时照此办理）——

一路 RTSP 解码一次，在帧上扇出给两个消费者：

        C6c RTSP ──► 采集进程 ──┬── 1fps 帧 → 人形检测 → data/raw/camera/{id}/{date}.json → GRU 社交轨
                                └── 音视频环缓 → 触发器 → data/raw/video_clips/{id}/{date}/ → MPDD

  - 扇出点必须在**解码之后、处理之前**：C6c 的 RTSP 并发连接数有限，两个进程
    各拉一路等于解码两遍，白白吃 CPU。
  - GRU 那一支的计算全是现成的：camera.py 里的 smooth_person_counts（30s 中位数
    去抖）、filter_tv_roi_detections（电视 ROI 过滤）、compute_copresence_minutes
    都已实现并测过，且 data/raw/camera/{id}/{date}.json 就是 daily_job 现在读的
    路径与格式（{"copresence_min": ..., "timestamp": ...}）——**GRU 侧零改动**。
  - 不要全量录像：环形缓冲只保留最近 ~90 秒，触发才落盘。全天 1080p 录像
    ≈15~30 GB/天，而环缓方案是几十 MB/天，隐私面也小得多。
  - 触发器**不能用 OpenFace**：FaceLandmarkImg 是逐图片 shell-out 的，跑不了实时。
    实时只做粗筛（轻量人脸检测 + 能量 VAD，宁可多留），夜间再用现成的
    extract_clips 那套 OpenFace 硬筛选精挑。
  - ★ 采集进程**不许产出任何结论**，只产出上面两样东西。判定全部留在日批处理里。
    这条是硬的：本仓删过一次实时运行时（提交 0dddad1），TODO P0-2 记的教训是
    "三条路径各写各的，实时采集到的特征和每日轨读取的不一定是同一份"。
    采集进程是**数据源**，不是第二个决策入口。
"""

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from src.depression.mpdd_process import (
    EXIT_OK,
    MpddInvocationError,
    describe_failure,
    run_script,
)
from src.utils.io import atomic_write_json, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXTRACT_SCRIPT = "extract_clips_from_surveillance.py"


def get_clip_dir(elder_id: str, day_key: str) -> Path:
    """候选片段目录：data/raw/video_clips/{elder_id}/{day_key}/"""
    return get_project_root() / "data" / "raw" / "video_clips" / elder_id / day_key


class ClipSource(ABC):
    """候选片段来源的统一接口。"""

    @abstractmethod
    def collect(self, elder_id: str, day_key: str, tmpdir: Path) -> Path | None:
        """产出片段目录（含 *.mp4 与 index.json）。

        Returns:
            片段目录路径；无可用片段时返回 None（由调用方判 no_clip / no_source）。
        """
        ...


class FileClipSource(ClipSource):
    """从一段本地录像里挖片段（设备到货前的模式）。

    调用 MPDD 的 extract_clips_from_surveillance.py，但**不使用它的 --run_infer**：

      该选项内部是 `subprocess.run(command, check=False)`，退出码被丢弃、失败不打印、
      manifest 里也不留痕迹。若每个片段的推理都崩了（OOM、描述文件不对、HF 挂住），
      父进程仍然 exit 0，磁盘上只有 mp4 没有 JSON——**静默全失败**。
      我们自己逐个驱动推理，才能看见每一个退出码（见 runner.py）。

    另一个要处理的坑：manifest.json 是在片段全部导出**之后**才写的。所以
    "有 mp4 但没 manifest" = 导出中途崩了，目录是脏的，必须整个清掉重来，
    不能拿残缺的片段集去推理（片段数会影响 min_clips 判定）。
    """

    def __init__(self, video_path: str | Path, dep_cfg: dict):
        self.video_path = Path(video_path).expanduser().resolve()
        self.dep_cfg = dep_cfg

    def collect(self, elder_id: str, day_key: str, tmpdir: Path) -> Path | None:
        if not self.video_path.is_file():
            logger.warning(f"  └─ 录像不存在: {self.video_path}")
            return None

        out_dir = get_clip_dir(elder_id, day_key)
        if out_dir.exists():
            # 重跑同一天：先清干净。留着旧片段会让 n_clips 混入上一次的结果，
            # 而 n_clips 直接决定 low_confidence 判定。
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        clip_cfg = self.dep_cfg.get("clip", {}) or {}
        args = [
            "--video", str(self.video_path),
            "--output_dir", str(out_dir),
            "--top_k", str(clip_cfg.get("top_k", 3)),
            "--min_dur", str(clip_cfg.get("min_dur", 10.0)),
            "--max_dur", str(clip_cfg.get("max_dur", 45.0)),
            "--crop", str(clip_cfg.get("crop", "face")),
        ]

        proc = run_script(
            self.dep_cfg, EXTRACT_SCRIPT, args, tmpdir,
            timeout_sec=int(self.dep_cfg.get("timeout_sec", 1800)),
        )

        manifest_path = out_dir / "manifest.json"

        if proc.returncode != EXIT_OK:
            # 退出码非 0 有两类：一类是"这段录像没有语音/没有可用帧"（正常的
            # no_clip，不是故障），一类是真出错。两者的区分靠 manifest 在不在：
            # 前者在写 manifest 之前就 sys.exit(1) 了，后者也一样——所以统一
            # 清理目录，由调用方按"没产出片段"处理，同时把 stderr 记进日志。
            logger.warning(f"  └─ {describe_failure(proc, EXTRACT_SCRIPT)}")
            shutil.rmtree(out_dir, ignore_errors=True)
            return None

        if not manifest_path.exists():
            logger.error(
                f"  └─ {EXTRACT_SCRIPT} 退出码 0 但没有 manifest.json，"
                f"目录状态不可信，已清理"
            )
            shutil.rmtree(out_dir, ignore_errors=True)
            return None

        clips = self._normalize_manifest(manifest_path, out_dir)
        if not clips:
            logger.info("  └─ 没有片段通过筛选（人脸/正脸/语音占比不达标）")
            return out_dir  # 目录留着，index.json 记录了 0 段，供排查

        logger.info(f"  └─ 截取到 {len(clips)} 个候选片段")
        return out_dir

    @staticmethod
    def _normalize_manifest(manifest_path: Path, out_dir: Path) -> list[dict]:
        """把 MPDD 的 manifest.json 转成统一的 index.json。

        为什么要转一层而不是直接用 manifest：manifest 的结构属于 MPDD 仓，
        它随时可能变（`settings` 字段直接是 `vars(args)`，加个 CLI 参数就会变）。
        index.json 是**我们自己的**契约，StreamClipSource 将来也产出同样的结构，
        runner 只认它。这样上游换实现、MPDD 升级，都影响不到下游。
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        clips = []
        for entry in manifest.get("clips", []):
            path = entry.get("path")
            if not path or not Path(path).is_file():
                continue
            clips.append({
                "path": path,
                "start": entry.get("start"),
                "end": entry.get("end"),
                "duration": entry.get("duration"),
                "speech_ratio": entry.get("speech_ratio"),
                "face_ratio": entry.get("face_ratio"),
                "frontal_ratio": entry.get("frontal_ratio"),
                "score": entry.get("score"),
            })

        atomic_write_json(out_dir / "index.json", {
            "source": manifest.get("source_video"),
            "source_duration_sec": manifest.get("duration"),
            "n_speech_segments": len(manifest.get("speech_segments", [])),
            "clips": clips,
        })
        return clips


class StreamClipSource(ClipSource):
    """【设备到货后实现】从 C6c 实时流的环形缓冲里触发式导出片段。

    实现要点见本模块 docstring 的"采集架构"一节。产出必须与 FileClipSource
    **完全一致**：同一个目录布局、同一份 index.json 结构——这是本抽象存在的
    全部意义。
    """

    def __init__(self, dep_cfg: dict):
        self.dep_cfg = dep_cfg

    def collect(self, elder_id: str, day_key: str, tmpdir: Path) -> Path | None:
        raise NotImplementedError(
            "实时流片段采集尚未接入，等待摄像头设备到货。依赖待决项："
            "边缘算力选型、环形缓冲时长、实时触发器的人脸/VAD 模型选型、"
            "以及与 GRU 社交轨共用一路解码的扇出实现。"
            "已就绪的部分：片段目录契约、聚合、推理驱动、结果落盘与展示。"
        )
