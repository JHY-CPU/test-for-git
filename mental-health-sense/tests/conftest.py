"""pytest配置和共享fixtures"""

import sys
from pathlib import Path

import pytest

# 确保项目根目录在Python路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """把项目根目录重定向到 tmp_path，所有产物落在临时目录里。

    放在 conftest 而不是各测试文件里：需要真实 data/ 布局的用例不止一处
    （test_integration 的全链路、test_regression_* 的按轨质量与阈值复用），
    各写一份会漂——与 imputer 删掉 `check_offline_status`、
    utils/status.py 收拢状态白名单是同一条理由。

    配置文件按原样拷进去：`load_config` 走的是 get_project_root()/config，
    不拷的话 tmp 里没有 settings.yaml，链路会在读配置时就炸。
    """
    import src.utils.io as io_mod

    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    for name in ("settings.yaml", "feature_weights.json"):
        (tmp_path / "config" / name).write_text(
            (PROJECT_ROOT / "config" / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    monkeypatch.setattr(io_mod, "get_project_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _isolate_alert_state(tmp_path, monkeypatch):
    """把预警事件状态重定向到 tmp，每个用例一份干净状态。

    ★ 必须 autouse。

      trigger_alert 现在是**有状态**的：它把"这个事件上次什么时候通知过"落到
      data/logs/alert_state/{elder_id}.json（跨天的信息，而每日批处理的进程
      活不过今天）。不隔离的话有两个后果：

        1. 跑一次 pytest 就往**真实数据目录**写状态，污染 E001 的生产数据；
        2. 状态**跨 pytest 会话残留**——第一次跑是"首次通知"，第二次跑同一条
           用例就变成"冷却中"，测试结果取决于跑过几次。

      第 2 条尤其危险：它制造的是不可复现的绿色，而本仓的教训是
      "不可复现的绿色比红色更危险"（曾因未固定 torch 种子给出假的 19/19）。

    用 monkeypatch 换掉 get_alert_state_dir 而不是改环境变量：这是本仓已有的
    模式（tests/test_depression_store.py 用同样方式换掉 store.get_depression_dir）。
    """
    from src.risk import alert_state

    state_dir = tmp_path / "alert_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(alert_state, "get_alert_state_dir", lambda: state_dir)
    return state_dir
