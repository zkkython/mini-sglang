#!/usr/bin/env python3
"""测试mini-sGLang API服务器"""

import requests
import json
import time

def test_api():
    """测试API端点"""
    url = "http://localhost:1919/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # 测试数据
    data = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "user", "content": "Hello, what is 2+2?"}
        ],
        "max_tokens": 20
    }

    print("=== 测试mini-sGLang API ===")
    print(f"URL: {url}")
    print(f"请求数据: {json.dumps(data, indent=2)}")
    print("-" * 50)

    try:
        # 发送请求
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data, timeout=30)
        elapsed = time.time() - start_time

        print(f"响应时间: {elapsed:.2f}秒")
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"响应: {json.dumps(result, indent=2)}")

            # 提取回复内容
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    print(f"\n模型回复: {content}")

            # 显示使用情况
            if "usage" in result:
                usage = result["usage"]
                print(f"\n使用情况:")
                print(f"  提示token数: {usage.get('prompt_tokens', 'N/A')}")
                print(f"  完成token数: {usage.get('completion_tokens', 'N/A')}")
                print(f"  总token数: {usage.get('total_tokens', 'N/A')}")
        else:
            print(f"错误响应: {response.text}")

    except requests.exceptions.ConnectionError:
        print("连接错误：服务器可能未运行或端口错误")
    except requests.exceptions.Timeout:
        print("请求超时：服务器响应时间过长")
    except Exception as e:
        print(f"其他错误: {e}")

def check_server_status():
    """检查服务器状态"""
    print("\n=== 检查服务器状态 ===")

    # 检查进程
    import subprocess
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    minisgl_processes = [line for line in result.stdout.split('\n') if 'minisgl' in line]

    if minisgl_processes:
        print(f"找到 {len(minisgl_processes)} 个minisgl进程:")
        for proc in minisgl_processes[:3]:  # 只显示前3个
            print(f"  {proc[:80]}...")
    else:
        print("未找到minisgl进程")

    # 检查端口
    print("\n检查端口1919:")
    try:
        result = subprocess.run(['ss', '-tlnp'], capture_output=True, text=True)
        port_lines = [line for line in result.stdout.split('\n') if ':1919' in line]
        if port_lines:
            print("端口1919正在监听:")
            for line in port_lines:
                print(f"  {line}")
        else:
            print("端口1919未监听")
    except:
        print("无法检查端口状态")

if __name__ == "__main__":
    check_server_status()
    print("\n")
    test_api()