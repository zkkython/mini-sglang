#!/bin/bash
# 启动mini-sGLang使用本地模型

echo "=== 启动mini-sGLang服务器（使用本地模型） ==="
echo "开始时间: $(date)"
echo "模型路径: /home/kason/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"

# 激活虚拟环境
source .venv/bin/activate

# 设置环境变量
export LOG_LEVEL=INFO

# 启动服务器
echo -e "\n启动命令:"
echo "python -m minisgl --model-path \"/home/kason/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B\" \\"
echo "  --max-running-requests 2 \\"
echo "  --port 1919"

echo -e "\n=== 服务器输出 ==="
python -m minisgl --model-path "/home/kason/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B" \
  --max-running-requests 2 \
  --attention-backend fa \
  --port 1919

echo -e "\n=== 启动完成 ==="
echo "结束时间: $(date)"