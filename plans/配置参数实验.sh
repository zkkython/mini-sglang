#!/bin/bash
# mini-sGLang 配置参数实验
# 测试不同配置参数对系统行为的影响

echo "=== mini-sGLang 配置参数实验 ==="
echo "开始时间: $(date)"
echo "模型路径: /home/kason/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
echo

# 激活虚拟环境
source .venv-system/bin/activate

# 全局变量
MODEL_PATH="/home/kason/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
LOG_DIR="/tmp/minisgl_experiments"
mkdir -p $LOG_DIR

# 清理函数
cleanup() {
    echo "清理进程..."
    pkill -f "minisgl" 2>/dev/null && echo "停止minisgl进程" || true
    sleep 2
}

# 测试函数
run_test() {
    local test_name=$1
    local port=$2
    shift 2
    local extra_args="$@"

    echo -e "\n=== 测试: $test_name ==="
    echo "端口: $port"
    echo "额外参数: $extra_args"

    cleanup

    # 启动服务器（后台运行）
    echo "启动服务器..."
    python -m minisgl \
        --model-path "$MODEL_PATH" \
        --port $port \
        $extra_args \
        --attention-backend fa \
        2>&1 | tee "$LOG_DIR/${test_name}.log" &

    SERVER_PID=$!
    echo "服务器PID: $SERVER_PID"

    # 等待服务器启动
    echo "等待10秒服务器启动..."
    sleep 10

    # 检查进程状态
    echo "进程状态:"
    ps aux | grep "minisgl" | grep -v grep | grep -v "配置参数实验" || echo "无minisgl进程"

    # 检查端口监听
    echo "端口监听状态:"
    ss -tlnp | grep ":$port" || echo "端口$port未监听"

    # 简单API测试
    echo "API测试..."
    python3 -c "
import requests, json, time
try:
    url = 'http://localhost:${port}/v1/chat/completions'
    data = {
        'model': 'Qwen/Qwen3-0.6B',
        'messages': [{'role': 'user', 'content': 'Hello, what is 1+1?'}],
        'max_tokens': 5
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
            print(f'回复长度: {len(content)}字符')
            print(f'回复前50字符: {content[:50]}...')
    else:
        print(f'错误: {resp.text[:100]}...')
except Exception as e:
    print(f'API错误: {e}')
"

    # 停止服务器
    echo "停止服务器..."
    kill $SERVER_PID 2>/dev/null && echo "服务器已停止" || echo "停止服务器失败"
    sleep 2

    # 提取关键日志信息
    echo "关键日志提取:"
    grep -E "(Server is ready|max_running_req|memory_ratio|max_extend_length|num_tokenizer|attention_backend|cache_type)" "$LOG_DIR/${test_name}.log" | tail -10

    echo "测试完成: $test_name"
    echo "日志文件: $LOG_DIR/${test_name}.log"
    echo
}

# ==================== 测试用例 ====================

echo "开始执行测试用例..."

# 测试1: 基础配置（默认参数）
run_test "test_01_baseline" 1919 ""

# 测试2: 不同端口和主机
run_test "test_02_port_1920" 1920 ""
run_test "test_03_host_localhost" 1921 "--host 127.0.0.1"

# 测试3: 并发请求数
run_test "test_04_max_req_1" 1922 "--max-running-requests 1"
run_test "test_05_max_req_10" 1923 "--max-running-requests 10"

# 测试4: 内存比例
run_test "test_06_memory_ratio_0.3" 1924 "--memory-ratio 0.3"
run_test "test_07_memory_ratio_0.7" 1925 "--memory-ratio 0.7"

# 测试5: 批处理参数
run_test "test_08_max_extend_128" 1926 "--max-prefill-length 128"
run_test "test_09_max_extend_512" 1927 "--max-prefill-length 512"

# 测试6: Tokenizer数量
run_test "test_10_num_tokenizer_1" 1928 "--num-tokenizer 1"
run_test "test_11_num_tokenizer_2" 1929 "--num-tokenizer 2"

# 测试7: 注意力后端
run_test "test_12_attn_fa" 1930 "--attention-backend fa"
run_test "test_13_attn_fi" 1931 "--attention-backend fi"

# 测试8: 缓存类型
run_test "test_14_cache_naive" 1932 "--cache-type naive"
run_test "test_15_cache_radix" 1933 "--cache-type radix"

# 测试9: 组合参数
run_test "test_16_combo_1" 1934 "--max-running-requests 4 --memory-ratio 0.5 --max-prefill-length 256 --num-tokenizer 1"

# ==================== 实验结果分析 ====================

echo "=== 实验结果汇总 ==="
echo "测试完成时间: $(date)"
echo

echo "各测试日志文件:"
ls -la $LOG_DIR/*.log | head -20

echo -e "\n=== 关键指标比较 ==="
echo "测试名称 | 端口 | 启动成功 | API响应 | 关键配置"
echo "--------|------|----------|----------|----------"

for log_file in $LOG_DIR/*.log; do
    test_name=$(basename $log_file .log)
    port=$(echo $test_name | grep -oE '[0-9]+$' || echo "N/A")

    # 检查启动成功
    if grep -q "Server is ready" "$log_file"; then
        startup="成功"
    else
        startup="失败"
    fi

    # 检查API响应
    if grep -q "API响应时间" "$log_file"; then
        api_time=$(grep "API响应时间" "$log_file" | tail -1 | grep -oE '[0-9]+\.[0-9]+')
        api_status="${api_time}秒"
    else
        api_status="无响应"
    fi

    # 提取关键配置
    config=$(grep -E "(max_running_req|memory_ratio|max_extend_length|num_tokenizer)" "$log_file" | head -3 | tr '\n' ' ')

    echo "$test_name | $port | $startup | $api_status | $config"
done

echo -e "\n=== 清理 ==="
cleanup

echo -e "\n=== 实验完成 ==="
echo "结束时间: $(date)"
echo "所有日志保存在: $LOG_DIR"
echo "使用以下命令查看详细日志:"
echo "  tail -f $LOG_DIR/test_*.log"
echo "  grep -E '(错误|错误|ERROR|error)' $LOG_DIR/*.log"