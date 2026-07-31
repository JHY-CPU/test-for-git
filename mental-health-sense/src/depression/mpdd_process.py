"""MPDD 子进程调用的底层封装（环境钉死 + 超时 + 退出码检查）

★ 这是本项目里唯一允许启动外部进程的地方（clip_source / runner 都经由它）。

为什么必须用子进程而不是直接 import MPDD 的代码：

  MPDD 要拖进来 transformers + librosa + av + cv2 + OpenFace 二进制 + 几个 G 的
  权重。一旦它们进了主链路的 import 图，**任何一个 import 失败，睡眠和社交监测
  就跟着一起挂**。本仓刚因为 pyaudio 缺 portaudio 头文件导致
  `pip install -r requirements.txt` 整体中止、torch 一个都装不上，教训是新鲜的。

  进程边界把这个风险挡在外面：MPDD 环境炸了，日管道照常跑，只是那天没有抑郁评估。

四个必须钉死的环境变量（每一个都对应一种实测过的挂法）：

  HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE
      即使 HF 缓存已预热（third_party/huggingface/ 有 1.4 GB 的 roberta-large），
      transformers 仍会去 hub 发一次未认证请求。网络不通时那是一次**无上限的等待**，
      而我们的调用方是每日批处理，挂住就是当天没结果且没有任何征兆。

  TORCH_HOME
      ResNet50 的 ImageNet 权重走 torch.hub，落在 ~/.cache/torch/hub/checkpoints/，
      **不受 infer_elder_depression.py 的 --model_cache_dir 控制**（那个只管 HF）。
      不钉死的话，换个用户/容器跑就会突然去下载 103 MB。

  NUMBA_CACHE_DIR / TMPDIR
      infer 脚本在 import 时就 setdefault 了 NUMBA_CACHE_DIR 到 $TMPDIR；
      而 extract 脚本会把采样帧以**全分辨率 JPEG** 落到 $TMPDIR
      （--scan_fps 2.0 下一小时 1080p 视频轻松几个 GB）。默认 /tmp 常常是
      tmpfs 或小分区，撑爆了报的错和真正的原因离得很远。

另一条硬约束：**MPDD 仓不可移动或改名**。`third_party/OpenFace/build/bin/FaceLandmarkImg`
的 RUNPATH 是编译进二进制的绝对路径
（/home/zhousenyu/project/MPDD-AVG-2026/third_party/OpenBLAS/install/lib），
移动后启动即报 `libopenblas.so.0: cannot open shared object file`。
所以配置里的 mpdd_root 必须是那个绝对路径，本模块启动前会校验它存在。
"""

import os
import subprocess
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

# infer_elder_depression.py 的退出码约定（实测）：
#   0 = 成功，JSON 已完整写出（该脚本从不产生半截 JSON）
#   1 = 运行期错误，traceback 在 stderr
#   2 = argparse 参数错误
EXIT_OK = 0
EXIT_CLI_ERROR = 2


class MpddInvocationError(RuntimeError):
    """MPDD 子进程调用失败（含超时、非零退出、可执行文件缺失）。"""


def _resolve_paths(dep_cfg: dict) -> tuple[Path, Path]:
    """解析并校验 mpdd_root 与 python_bin，缺失立即报错。

    早报错是有意的：真正的失败点（OpenFace 找不到动态库、解释器不存在）产生的
    错误信息离原因很远，不如在入口处直接说清是哪个路径不对。
    """
    root = Path(dep_cfg.get("mpdd_root", "")).expanduser()
    python_bin = Path(dep_cfg.get("python_bin", "")).expanduser()

    if not root.is_dir():
        raise MpddInvocationError(
            f"mpdd_root 不存在：{root}。注意 OpenFace 的 RUNPATH 是编译进去的绝对"
            f"路径，MPDD 仓不能移动或改名。"
        )
    if not python_bin.is_file():
        raise MpddInvocationError(
            f"python_bin 不存在：{python_bin}。MPDD 需要独立解释器"
            f"（transformers/librosa/av/cv2），刻意不装进本项目的 venv。"
        )
    return root, python_bin


def build_env(dep_cfg: dict, tmpdir: Path) -> dict:
    """构造子进程环境变量（在当前环境基础上覆盖关键项）。"""
    root, _ = _resolve_paths(dep_cfg)
    tmpdir.mkdir(parents=True, exist_ok=True)

    torch_home = dep_cfg.get("torch_home") or str(Path.home() / ".cache" / "torch")

    env = dict(os.environ)
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(root / "third_party" / "huggingface"),
        "TORCH_HOME": torch_home,
        "TMPDIR": str(tmpdir),
        "NUMBA_CACHE_DIR": str(tmpdir / "numba"),
    })
    # ROS 等系统级 PYTHONPATH 会把不兼容的 site-packages 注入 MPDD 解释器
    # （本机 /opt/ros/humble 就在 PYTHONPATH 里，实测会让 pytest 都起不来）。
    # 子进程必须用它自己解释器的干净路径。
    env.pop("PYTHONPATH", None)
    return env


def run_script(
    dep_cfg: dict,
    script_name: str,
    args: list[str],
    tmpdir: Path,
    timeout_sec: int = 1800,
) -> subprocess.CompletedProcess:
    """在 MPDD 仓里跑一个脚本，返回 CompletedProcess（不自动抛错）。

    不设 cwd：两个脚本都以 `Path(__file__).resolve().parent` 锚定自身路径，
    `from models import ...` 靠 sys.path[0] = 脚本目录解析，与工作目录无关。

    Raises:
        MpddInvocationError: 超时、或可执行文件/脚本缺失
    """
    root, python_bin = _resolve_paths(dep_cfg)
    script = root / script_name
    if not script.is_file():
        raise MpddInvocationError(f"MPDD 脚本不存在：{script}")

    command = [str(python_bin), str(script), *args]
    logger.info(f"  └─ 调用 MPDD: {script_name} {' '.join(args[:6])}...")

    try:
        return subprocess.run(
            command,
            env=build_env(dep_cfg, tmpdir),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        raise MpddInvocationError(
            f"{script_name} 超时（{timeout_sec}s）。首次运行需要加载 roberta-large "
            f"与 ResNet50；若网络不通且未设 HF_HUB_OFFLINE 会无限等待。"
        ) from e


def describe_failure(proc: subprocess.CompletedProcess, script_name: str) -> str:
    """把子进程失败压成一行可读诊断（stderr 只留尾部，traceback 的根因在最后）。"""
    tail = (proc.stderr or "").strip().splitlines()
    tail_text = " | ".join(tail[-3:]) if tail else "(无 stderr)"
    hint = "（参数错误，检查配置）" if proc.returncode == EXIT_CLI_ERROR else ""
    return f"{script_name} 退出码 {proc.returncode}{hint}: {tail_text}"


def mpdd_git_rev(dep_cfg: dict) -> str:
    """取 MPDD 仓的当前 commit，写进 provenance 供事后追溯。

    取不到就返回 "unknown"——这是溯源信息，不该因为 git 不可用而让整次评估失败。
    """
    try:
        root, _ = _resolve_paths(dep_cfg)
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError, MpddInvocationError):
        return "unknown"
