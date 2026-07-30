"""确定性随机种子工具

内置 `hash()` 对 str 带随机化盐（PYTHONHASHSEED 默认为 random），同一字符串在
不同进程会得到不同哈希值。若把它掺进随机种子，"固定 seed"就形同虚设——每次运行
生成的数据都不一样，导致验证脚本断言概率性失败、mock 数据无法复现。

本模块用 crc32 替代：确定性、跨进程跨平台稳定、无需加密强度（这里只做种子派生）。
"""

import zlib


def stable_hash(text: str) -> int:
    """跨进程稳定的字符串哈希（非负整数）。

    Args:
        text: 待哈希的字符串

    Returns:
        crc32 校验值，恒为 0 ~ 2**32-1 的非负整数
    """
    return zlib.crc32(text.encode("utf-8"))


def stable_seed(text: str, modulus: int = 2**31) -> int:
    """从字符串派生确定性随机种子。

    Args:
        text: 种子来源字符串（如日期、老人ID）
        modulus: 取模上界，默认 2**31（numpy RandomState 的合法上界）

    Returns:
        0 ~ modulus-1 的种子值
    """
    return stable_hash(text) % modulus
