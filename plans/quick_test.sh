#!/bin/bash
# 快速配置参数测试

source .venv-system/bin/activate

MODEL_PATH="/home/kason/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
PORT=1919

echo "=== 测试1: 基础配置 ==="
echo "端口: $PORT"

# 清理
pkill -f "minisgl" 2>/dev/null && echo "清理进程" || true
sleep 2

# 启动服务器
echo "启动服务器..."
python -m minisgl \
  --model-path "$MODEL_PATH" \
  --port $PORT \
  --attention-backend fa \
  2>&1 | tee /tmp/test_baseline.log &

SERVER_PID=$!
echo "服务器PID: $SERVER_PID"

# 等待启动
echo "等待15秒启动..."
sleep 15

# 检查进程
echo "进程状态:"
ps aux | grep "minisgl" | grep -v grep | grep -v quick_test || echo "无进程"

# 检查端口
echo "端口状态:"
ss -tlnp | grep ":$PORT" || echo "端口未监听"

# API测试
echo "API测试..."
python3 -c "
import requests, json, time
try:
    url = 'http://localhost:$PORT/v1/chat/completions'
    data = {
        'model': 'Qwen/Qwen3-0.6B',
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'max_tokens': 10
    }
    start = time.time()
    resp = requests.post(url, json=data, timeout=30)
    elapsed = time.time() - start
    print(f'API响应时间: {elapsed:.2f}秒')
    print(f'状态码: {resp.status_code}')
    if resp.status_code == 200:
        result = resp.json()
        if 'choices' in result:
            content = result['choices'][0]['message']['content']
            print(f'回复: {content[:50]}...')
    else:
        print(f'错误: {resp.text[:200]}...')
except Exception as e:
    print(f'API错误: {e}')
"

# 停止
echo "停止服务器..."
kill $SERVER_PID 2>/dev/null && echo "已停止" || echo "停止失败"
sleep 2

echo "日志文件: /tmp/test_baseline.log"
echo "=== 测试完成 ==="