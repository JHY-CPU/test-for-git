"""pytest配置和共享fixtures"""

import sys
from pathlib import Path

import pytest

# 确保项目根目录在Python路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
