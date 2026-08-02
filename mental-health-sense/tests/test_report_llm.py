"""
周报 LLM 生成路径的回归测试（DeepSeek / OpenAI 兼容格式）

此前全仓没有任何用例触达 `_generate_with_llm` 的 LLM 分支——所有测试都传
`use_llm=False` 走规则模板，于是"换模型、改 base_url、key 读错环境变量名"
这类改动会在全绿之下静默通过，而生产链路每天真的在打 LLM（本仓"全绿不等于
没问题，要看绿的是什么"的教训，VALIDATION §8）。

本文件直接测 `_generate_with_llm`（真实入口的 LLM 段，入参是简单 dict，
无需搭整条周报链路），覆盖三条回落路径（无 key / SDK 未装 / 调用失败）
与成功路径的参数透传。mock 用 pytest 的 monkeypatch（全仓惯例，无
unittest.mock）——把假 openai 模块塞进 sys.modules，`from openai import
OpenAI` 取到的就是假的，不碰真实 SDK 与网络。

★ 不测 `generate_weekly_report` 全链路：那需要构造一周的 daily 日志与风险
判定，而本文件要盯的是"LLM 这段的失败处理"，链路组装由
test_integration / test_regression_2026_08_03（use_llm=False）负责。
"""

import sys
import types

import pytest

from src.report.weekly_report import _generate_with_llm

# ===== 最小入参 =====

_TRENDS = {
    "social_trend": "平稳",
    "sleep_trend": "平稳",
    "activity_trend": "平稳",
    "social_vs_baseline": "无明显差异",
    "sleep_vs_baseline": "无明显差异",
    "activity_vs_baseline": "无明显差异",
}

_RISK_RESULT = {
    "risk_label": "正常",
    "consecutive_deviation": 0,
}

_CFG = {
    "report": {
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "max_tokens": 400,
    }
}

# deviation_days=0 时规则模板的开头（generate_rule_based_report 的零偏离档）
_FALLBACK_MARK = "整体状态平稳"


# ===== 假 openai 模块 =====

class _FakeCompletions:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        self._client.calls.append(kwargs)
        if self._client.raise_on_call:
            raise RuntimeError("模拟 API 故障")
        return _FakeResponse()


class _FakeChat:
    def __init__(self, client):
        self.completions = _FakeCompletions(client)


class _FakeOpenAI:
    """模拟 openai.OpenAI 客户端：记录构造参数与调用参数，可配置抛错。"""

    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.raise_on_call = False
        self.calls = []
        self.chat = _FakeChat(self)


class _FakeMessage:
    content = "这是模型生成的周报正文。"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


@pytest.fixture
def fake_openai(monkeypatch):
    """把假 openai 模块装进 sys.modules。

    返回共享状态 dict：
        state["raise_on_call"]  调用前置位 → create 抛异常（测回落）
        state["client"]         最近一次构造的假客户端（测参数透传）

    monkeypatch 在测试结束后自动还原真 openai 模块，不会污染其他用例。
    """
    state = {"raise_on_call": False, "client": None}

    def _factory(api_key=None, base_url=None):
        client = _FakeOpenAI(api_key=api_key, base_url=base_url)
        client.raise_on_call = state["raise_on_call"]
        state["client"] = client
        return client

    module = types.ModuleType("openai")
    module.OpenAI = _factory
    monkeypatch.setitem(sys.modules, "openai", module)
    return state


# ===== 用例 =====

class TestGenerateWithLlm:

    def test_无key时回落规则模板且不碰SDK(self, monkeypatch):
        """★ API key 只从环境变量读。没设 key 必须直接回落，
        不白打一次 API 等 401——周报不能因为外部服务配置没到位就出不来。"""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        body = _generate_with_llm("E001", _TRENDS, _RISK_RESULT, "无", _CFG)

        assert _FALLBACK_MARK in body
        # 无 key 分支在 import openai 之前就返回，真 SDK 若已装也不该被调用

    def test_openai未安装时回落规则模板(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
        # sys.modules 里值为 None = "该模块导入失败"，from openai import OpenAI
        # 会抛 ImportError → 回落。即使本机装了真 openai 也一样触发。
        monkeypatch.setitem(sys.modules, "openai", None)

        body = _generate_with_llm("E001", _TRENDS, _RISK_RESULT, "无", _CFG)

        assert _FALLBACK_MARK in body

    def test_调用失败时回落规则模板(self, fake_openai, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
        fake_openai["raise_on_call"] = True

        body = _generate_with_llm("E001", _TRENDS, _RISK_RESULT, "无", _CFG)

        assert _FALLBACK_MARK in body
        assert fake_openai["client"].calls, "失败路径也应当发起过调用"

    def test_成功路径返回模型正文(self, fake_openai, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")

        body = _generate_with_llm("E001", _TRENDS, _RISK_RESULT, "无", _CFG)

        assert body == "这是模型生成的周报正文。"

    def test_参数从配置与环境变量透传(self, fake_openai, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
        cfg = {
            "report": {
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com/v1",
                "max_tokens": 300,
            }
        }

        _generate_with_llm("E001", _TRENDS, _RISK_RESULT, "无", cfg)

        client = fake_openai["client"]
        assert client.api_key == "sk-test-123"          # key 从环境变量读
        assert client.base_url == "https://api.deepseek.com/v1"
        call = client.calls[-1]
        assert call["model"] == "deepseek-chat"
        assert call["max_tokens"] == 300
        assert call["messages"][0]["role"] == "system"
        assert call["messages"][1]["role"] == "user"
        # 提示词里填的是真数据，不是空壳
        assert "社交互动频次：平稳" in call["messages"][1]["content"]
        assert "风险等级：正常" in call["messages"][1]["content"]

    def test_配置缺省时用DeepSeek默认值(self, fake_openai, monkeypatch):
        """settings.yaml 的 report 段读不出（或损坏）时也要有默认值，
        与"读不出配置回落空字典而不是抛"是同一层防御。"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")

        _generate_with_llm("E001", _TRENDS, _RISK_RESULT, "无", {})

        client = fake_openai["client"]
        assert client.base_url == "https://api.deepseek.com"
        assert client.calls[-1]["model"] == "deepseek-v4-pro"
        assert client.calls[-1]["max_tokens"] == 400
