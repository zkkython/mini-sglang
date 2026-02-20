#!/usr/bin/env python3
"""测试mini-sGLang环境是否正常工作"""

import sys
import torch

print("=== Mini-SGLang 环境测试 ===")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 测试是否能导入minisgl模块
try:
    import minisgl
    print("\n✅ minisgl模块导入成功")

    # 测试是否能导入核心模块
    from minisgl.core import SamplingParams
    print("✅ minisgl.core模块导入成功")

    # 创建测试参数
    params = SamplingParams(temperature=0.7, max_tokens=100)
    print(f"✅ SamplingParams创建成功: temperature={params.temperature}, max_tokens={params.max_tokens}")

except ImportError as e:
    print(f"\n❌ 导入失败: {e}")
    print("请确保已运行: uv pip install -e .")
except Exception as e:
    print(f"\n❌ 其他错误: {e}")

print("\n=== 环境测试完成 ===")