"""架构不变量守卫：抑郁模块不得被核心链路导入

★ 这不是风格检查，是**架构不变量的守卫**。

  哪天有人"顺手"在 judge.py 里读一下抑郁分来佐证风险等级，功能上一切正常、
  周报照出、预警照发，**不会有任何测试变红**。这正是 VALIDATION §8 反复强调的
  "全绿不等于没问题，要看绿的是什么"——那一批缺陷全都活在 312 全绿 + 19/19 + 9/9
  之下，因为断言打错了层。所以要有一条专门盯着这条边界的断言。

  它防的是一个具体的失效链（见 src/depression/__init__.py）：
      抑郁判高 → 该日标为异常 → 异常日被排除出每周微调 → 个人基线越缩越窄
      → 更容易判偏离 → 偏离又被拿去佐证抑郁 → ……
  自我强化，而且每一步单独看都"合理"。

第二条守卫是依赖方向：MPDD 的依赖（transformers / librosa / av / cv2）绝不能进
requirements.txt。本仓刚因 pyaudio 缺 portaudio 头文件导致整条 pip install 中止、
torch 一个都装不上，教训是新鲜的。
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 核心链路：这些包里的任何文件都不得依赖抑郁模块
PROTECTED_PACKAGES = ("baseline", "risk", "data_pipeline", "scheduler")

# MPDD 专属依赖，绝不允许出现在 requirements.txt
FORBIDDEN_REQUIREMENTS = ("transformers", "librosa", "av", "opencv", "opensmile")


def _imported_modules(py_file: Path) -> set[str]:
    """静态解析一个文件 import 了哪些模块（不执行代码）。"""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def _protected_files() -> list[Path]:
    files: list[Path] = []
    for pkg in PROTECTED_PACKAGES:
        files.extend(sorted((PROJECT_ROOT / "src" / pkg).rglob("*.py")))
    return files


class TestCoreDoesNotImportDepression:
    def test_守卫范围非空(self):
        """防止这条测试因为目录改名而静默失效——扫不到文件就等于没测。"""
        files = _protected_files()
        assert len(files) >= 15, f"受保护文件只扫到 {len(files)} 个，路径可能已变更"

    @pytest.mark.parametrize(
        "py_file", _protected_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT))
    )
    def test_核心链路不导入抑郁模块(self, py_file):
        offenders = {m for m in _imported_modules(py_file) if m.startswith("src.depression")}
        assert not offenders, (
            f"{py_file.relative_to(PROJECT_ROOT)} 导入了 {offenders}。\n"
            f"抑郁评估是群体绝对基线的旁路输出，不得进入个人基线的任何环节，"
            f"否则会形成自我强化闭环：抑郁判高 → 该日标为异常 → 异常日被排除出"
            f"每周微调 → 个人基线越缩越窄 → 更容易判偏离 → 偏离又被拿去佐证抑郁。"
            f"详见 src/depression/__init__.py。"
        )

    def test_动态导入也拦住(self):
        """ast 已覆盖 import / from-import（含函数内的延迟导入），
        但 importlib.import_module("src.depression.x") 这类字符串形式绕得过去。
        这里补一遍文本扫描，只匹配模块路径本身，不误伤普通注释。"""
        for py_file in _protected_files():
            text = py_file.read_text(encoding="utf-8")
            for pattern in ("src.depression", "src/depression"):
                assert pattern not in text, (
                    f"{py_file.relative_to(PROJECT_ROOT)} 出现 {pattern!r}，"
                    f"疑似以动态方式引用抑郁模块"
                )


class TestDependencyIsolation:
    def test_requirements_不含_MPDD_依赖(self):
        req = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        # 只看未注释的行——文件尾部的说明性注释里会提到这些包名
        active = [
            line.split("#")[0].strip().lower()
            for line in req.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        for forbidden in FORBIDDEN_REQUIREMENTS:
            hits = [line for line in active if line.startswith(forbidden)]
            assert not hits, (
                f"requirements.txt 出现 MPDD 专属依赖 {hits}。"
                f"它们必须留在 MPDD 自己的 conda 环境里，靠 subprocess 进程边界隔离——"
                f"一旦进了主链路的 import 图，任何一个 import 失败都会让"
                f"睡眠和社交监测一起挂。"
            )


class TestReportOnlyUsesLightModules:
    def test_周报不导入_runner_或_clip_source(self):
        """周报必须在 MPDD 环境完全不可用时照常渲染。

        store / aggregate / status / contract 是零重依赖的；
        runner / clip_source / mpdd_process 会拉起 subprocess 与外部仓。
        """
        report_pkg = sorted((PROJECT_ROOT / "src" / "report").rglob("*.py"))
        heavy = {"src.depression.runner", "src.depression.clip_source",
                 "src.depression.mpdd_process"}
        for py_file in report_pkg:
            offenders = _imported_modules(py_file) & heavy
            assert not offenders, (
                f"{py_file.relative_to(PROJECT_ROOT)} 导入了 {offenders}；"
                f"周报只允许 import src.depression.store"
            )
