# 阶段二：核心组件深入 - 学习笔记

## 目录

| 编号 | 主题 | 文件 | 对应任务 |
|------|------|------|----------|
| 2.1 | [核心数据结构分析](./2.1-core-data-structures.md) | `core.py` | 任务2.1 |
| 2.2 | [调度器Prefill阶段分析](./2.2-prefill-scheduling.md) | `scheduler/prefill.py` | 任务2.2 |
| 2.3 | [调度器Decode阶段分析](./2.3-decode-scheduling.md) | `scheduler/decode.py`, `scheduler/scheduler.py` | 任务2.3 |
| 2.4 | [引擎初始化分析](./2.4-engine-init.md) | `engine/engine.py` | 任务2.4 |
| 2.5 | [推理流程跟踪](./2.5-forward-pipeline.md) | `engine/engine.py`, `engine/graph.py`, `engine/sample.py` | 任务2.5 |
| 2.6 | [模型实现分析](./2.6-model-impl.md) | `models/` | 任务2.6 |

## 核心学习目标

1. 理解 `Req` / `Batch` / `Context` / `SamplingParams` 数据结构的设计与生命周期
2. 掌握 Prefill 和 Decode 两阶段调度机制
3. 理解 Engine 初始化流程（模型加载、KV缓存分配、CUDA图捕获）
4. 跟踪完整的推理流程（前向传播 → 采样 → 结果处理）
5. 分析模型实现的架构模式（注册机制、权重加载、分层设计）

## 关键源文件索引

```
python/minisgl/
├── core.py                    # SamplingParams, Req, Batch, Context, 全局Context
├── scheduler/
│   ├── scheduler.py           # Scheduler 主循环、overlap scheduling
│   ├── prefill.py             # PrefillManager, PrefillAdder, ChunkedReq
│   ├── decode.py              # DecodeManager
│   ├── cache.py               # CacheManager (调度器侧缓存管理)
│   ├── table.py               # TableManager (page table 槽位管理)
│   └── utils.py               # PendingReq
├── engine/
│   ├── engine.py              # Engine (模型加载、前向推理、采样)
│   ├── graph.py               # GraphRunner (CUDA图捕获与回放)
│   └── sample.py              # Sampler, BatchSamplingArgs
└── models/
    ├── base.py                # BaseLLMModel (抽象基类)
    ├── config.py              # ModelConfig, RotaryConfig
    ├── register.py            # 模型注册表 _MODEL_REGISTRY
    ├── llama.py               # LlamaForCausalLM
    ├── qwen2.py               # Qwen2ForCausalLM
    ├── qwen3.py               # Qwen3ForCausalLM
    └── qwen3_moe.py           # Qwen3MoeForCausalLM
```
