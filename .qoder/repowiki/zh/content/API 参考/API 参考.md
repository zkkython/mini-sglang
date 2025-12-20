# API 参考

<cite>
**本文档中引用的文件**  
- [llm.py](file://python/minisgl/llm/llm.py)
- [api_server.py](file://python/minisgl/server/api_server.py)
- [core.py](file://python/minisgl/core.py)
- [scheduler.py](file://python/minisgl/scheduler/scheduler.py)
- [tokenizer.py](file://python/minisgl/message/tokenizer.py)
- [frontend.py](file://python/minisgl/message/frontend.py)
</cite>

## 目录
1. [HTTP API](#http-api)
2. [Python API](#python-api)
3. [使用示例](#使用示例)

## HTTP API

### /v1/chat/completions 端点

`/v1/chat/completions` 端点提供与 OpenAI 兼容的聊天完成功能，支持通过 POST 请求生成文本。

#### 请求格式
- **方法**: POST
- **路径**: `/v1/chat/completions`
- **内容类型**: `application/json`
- **流式响应**: 支持通过 `text/event-stream` 进行流式输出

#### 支持的参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `model` | 字符串 | 是 | 模型标识符，用于指定要使用的模型。 |
| `messages` | 对象数组 | 否 | 包含对话消息的数组，每个消息包含 `role`（角色）和 `content`（内容）。角色可以是 `system`、`user` 或 `assistant`。 |
| `prompt` | 字符串 | 否 | 用于生成补全的提示文本。`messages` 和 `prompt` 至少需要提供一个。 |
| `max_tokens` | 整数 | 否 | 生成的最大 token 数量，默认值为 16。 |
| `temperature` | 浮点数 | 否 | 控制生成文本的随机性，值越高越随机，默认值为 1.0。 |
| `top_p` | 浮点数 | 否 | 核采样参数，控制生成文本的多样性，默认值为 1.0。 |
| `n` | 整数 | 否 | 为每个提示生成的完成数量，默认值为 1。 |
| `stream` | 布尔值 | 否 | 是否启用流式响应，默认值为 `false`。 |
| `stop` | 字符串数组 | 否 | 生成停止的序列列表。 |
| `presence_penalty` | 浮点数 | 否 | 控制新 token 出现的惩罚，默认值为 0.0。 |
| `frequency_penalty` | 浮点数 | 否 | 控制重复 token 的惩罚，默认值为 0.0。 |
| `ignore_eos` | 布尔值 | 否 | 是否忽略结束符（EOS），默认值为 `false`。 |

#### 流式响应处理

当 `stream` 参数设置为 `true` 时，服务器将通过 `text/event-stream` 发送流式响应。每个响应块以 `data: ` 开头，包含 JSON 格式的增量输出。流式响应的处理方式如下：

1. 客户端通过 `StreamingResponse` 接收流式数据。
2. 每个数据块包含增量输出，客户端可以实时显示生成的文本。
3. 流式响应以 `data: [DONE]` 结束，表示生成完成。

```python
async def stream_chat_completions(self, uid: int):
    first_chunk = True
    async for ack in self.wait_for_ack(uid):
        delta = {}
        if first_chunk:
            delta["role"] = "assistant"
            first_chunk = False
        if ack.incremental_output:
            delta["content"] = ack.incremental_output

        chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()

        if ack.finished:
            break

    end_chunk = {
        "id": f"cmpl-{uid}",
        "object": "text_completion.chunk",
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(end_chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"
```

#### 错误码

| 错误码 | 描述 |
|-------|------|
| 400 | 请求参数无效或缺失。 |
| 404 | 请求的资源未找到。 |
| 500 | 服务器内部错误。 |

**Section sources**
- [api_server.py](file://python/minisgl/server/api_server.py#L244-L273)

## Python API

### LLM 类

`LLM` 类提供了与 Mini-SGLang 系统交互的 Python 接口，支持生成和流式输出文本。

#### 构造函数

```python
def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs)
```

- **参数**:
  - `model_path` (字符串): 模型路径。
  - `dtype` (torch.dtype): 模型权重的数据类型，默认为 `torch.bfloat16`。
  - `**kwargs`: 其他可选参数。

#### generate 方法

```python
def generate(
    self,
    prompts: List[str] | List[List[int]],
    sampling_params: List[SamplingParams] | SamplingParams,
) -> List[str]
```

- **参数**:
  - `prompts` (字符串列表或整数列表): 输入提示。
  - `sampling_params` (SamplingParams 或 SamplingParams 列表): 采样参数。
- **返回值**: 生成的文本列表。

#### stream 方法

```python
async def stream_generate(self, uid: int)
```

- **参数**:
  - `uid` (整数): 用户标识符。
- **返回值**: 异步生成器，用于流式输出文本。

**Section sources**
- [llm.py](file://python/minisgl/llm/llm.py#L29-L99)

## 使用示例

### 同步调用

```python
from minisgl.llm import LLM

llm = LLM(model_path="Qwen/Qwen3-0.6B")
prompts = ["Hello, how are you?"]
results = llm.generate(prompts, max_tokens=50)
print(results)
```

### 异步流式输出

```python
import asyncio
from minisgl.server.api_server import FrontendManager

async def main():
    state = FrontendManager(config)
    uid = state.new_user()
    await state.send_one(TokenizeMsg(uid=uid, text="Hello, how are you?", sampling_params=SamplingParams(max_tokens=50)))
    
    async for chunk in state.stream_generate(uid):
        print(chunk.decode(), end="", flush=True)

asyncio.run(main())
```

**Section sources**
- [llm.py](file://python/minisgl/llm/llm.py#L78-L99)
- [api_server.py](file://python/minisgl/server/api_server.py#L150-L156)