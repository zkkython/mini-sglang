# Mini-SGLang 代码库系统性学习计划

## 上下文

用户是mini-sGLang推理框架的新手，希望系统性地学习这个代码库。mini-sGLang是一个高性能的分布式LLM推理框架，采用多进程架构，支持Tensor Parallelism、先进的KV缓存、注意力后端优化等特性。本计划旨在为新手提供一个循序渐进的学习路径，帮助他们从零开始掌握框架的核心概念、架构设计和实现细节。

## 学习目标

1. 理解mini-SGLang的整体架构和多进程设计
2. 掌握核心数据结构和调度机制
3. 熟悉高级优化技术（注意力后端、KV缓存、CUDA图等）
4. 能够运行、调试和扩展框架
5. 为参与代码贡献做好准备

## 学习阶段与路径

### 阶段一：入门与基础（1-2周）
**目标**：建立整体认知，能够运行框架

**关键文件**：
- `/home/kason/python_workspace/mini-sglang/python/minisgl/__main__.py` - 主入口
- `/home/kason/python_workspace/mini-sglang/python/minisgl/server/launch.py` - 进程启动逻辑
- `/home/kason/python_workspace/mini-sglang/python/minisgl/server/api_server.py` - OpenAI兼容API
- `/home/kason/python_workspace/mini-sglang/README.md` - 项目文档

**学习内容**：
1. 安装和环境配置（Docker或原生安装）
2. 理解多进程架构：API Server、Tokenizer、Detokenizer、Scheduler Workers
3. 运行示例程序，跟踪请求生命周期
4. 学习ZMQ进程间通信机制

**实践练习**：
- 单GPU/多GPU环境部署
- 使用`--log-level DEBUG`观察日志
- 修改配置参数（batch size、端口等）

### 阶段二：核心组件深入（2-3周）
**目标**：掌握核心数据结构和调度机制

**关键文件**：
- `/home/kason/python_workspace/mini-sglang/python/minisgl/core.py` - 核心数据结构（Req, Batch, Context, SamplingParams）
- `/home/kason/python_workspace/mini-sglang/python/minisgl/scheduler/scheduler.py` - 调度器实现
- `/home/kason/python_workspace/mini-sglang/python/minisgl/engine/engine.py` - 推理引擎
- `/home/kason/python_workspace/mini-sglang/python/minisgl/models/` - 模型实现（Llama, Qwen系列）

**学习内容**：
1. Req/Batch/Context数据结构的生命周期
2. 调度器的prefill和decode阶段管理
3. 引擎的推理流程（前向传播、采样）
4. 模型加载和权重管理

**实践练习**：
- 添加Req/Batch生命周期日志
- 实现简单的优先级调度策略
- 加载自定义模型权重

### 阶段三：高级特性与优化（2-3周）
**目标**：理解性能优化技术和分布式通信

**关键文件**：
- `/home/kason/python_workspace/mini-sglang/python/minisgl/attention/` - 注意力后端（FlashAttention, FlashInfer）
- `/home/kason/python_workspace/mini-sglang/python/minisgl/kvcache/` - KV缓存管理（NaiveCache, RadixCache）
- `/home/kason/python_workspace/mini-sglang/python/minisgl/distributed/` - 分布式通信（NCCL）
- `/home/kason/python_workspace/mini-sglang/python/minisgl/kernel/` - CUDA内核（通过TVM-FFI）

**学习内容**：
1. 注意力后端的差异和适用场景
2. KV缓存的管理策略和共享前缀优化
3. Tensor Parallelism的通信模式
4. CUDA图优化原理

**实践练习**：
- 切换注意力后端，比较性能差异
- 实现自定义缓存策略
- 分析NCCL通信瓶颈
- 使用nsys分析CUDA内核性能

### 阶段四：扩展与贡献（1-2周）
**目标**：能够修改框架、添加新功能

**关键文件**：
- `/home/kason/python_workspace/mini-sglang/python/minisgl/utils/` - 工具类
- `/home/kason/python_workspace/mini-sglang/tests/` - 测试框架
- `/home/kason/python_workspace/mini-sglang/benchmark/` - 基准测试
- `/home/kason/python_workspace/mini-sglang/python/minisgl/llm/llm.py` - Python高级接口

**学习内容**：
1. 工具类和辅助函数
2. 测试框架的使用和编写
3. 性能基准测试方法
4. Python API的设计

**实践练习**：
- 添加新模型支持
- 编写单元测试和集成测试
- 优化特定操作（如采样、日志记录）
- 贡献代码到开源项目

## 关键概念学习顺序

1. **数据流**：用户请求 → API Server → Tokenizer → Scheduler → Engine → Detokenizer → 用户
2. **核心数据结构**：Req → Batch → Context
3. **调度阶段**：Prefill（处理新提示） → Decode（生成令牌）
4. **优化技术**：Chunked Prefill → 注意力后端 → KV缓存 → CUDA图 → Overlap Scheduling
5. **分布式架构**：ZMQ控制通信 → NCCL张量通信 → Tensor Parallelism

## 实践建议

### 调试技巧
- 使用`--log-level DEBUG`获取详细日志
- Python调试器：`import pdb; pdb.set_trace()`
- GPU内存监控：`torch.cuda.memory_stats()`
- 性能分析：PyTorch profiler, nsys, nvprof

### 测试运行
```bash
# 单元测试
pytest /home/kason/python_workspace/mini-sglang/tests/ -v

# 基准测试
python /home/kason/python_workspace/mini-sglang/benchmark/offline/bench.py
python /home/kason/python_workspace/mini-sglang/benchmark/online/bench_qwen.py
```

### 学习资源
- 官方文档：README.md, docs/structures.md, docs/features.md
- 代码注释：详细的类型注解和docstring
- 相关论文：SGLang, FlashAttention, FlashInfer
- GitHub issues和discussions

## 评估标准

### 入门级（1-2个月）
- 能够安装和运行框架
- 理解基本架构和组件关系
- 能够修改配置参数

### 中级（3-4个月）
- 理解核心数据结构和算法
- 能够调试常见问题
- 能够进行性能调优

### 高级（5-6个月）
- 能够扩展框架功能
- 能够优化内核性能
- 能够参与代码贡献

## 后续步骤

完成此学习计划后，学习者可以：
1. 深入阅读相关论文和技术文档
2. 参与开源社区讨论和贡献
3. 基于mini-sGLang构建自定义推理服务
4. 探索其他推理框架进行对比学习

## 详细任务拆分

根据用户选择的"具体操作任务"粒度，以下是为每个阶段设计的详细任务：

### 阶段一：入门与基础（预计：20-30小时，2周）

**任务1.1：环境准备** (2-3小时)
- 创建并激活Python虚拟环境（使用uv）
- 安装项目依赖：`uv pip install -e .`
- 验证CUDA环境：`nvidia-smi` 和 `python -c "import torch; print(torch.cuda.is_available())"`

**任务1.2：初次运行** (3-4小时)
- 使用Docker运行：`docker build -t minisgl .` 和 `docker run --gpus all -p 1919:1919 minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0`
- 原生运行：`python -m minisgl --model "Qwen/Qwen3-0.6B"`
- 测试API端点：`curl -X POST http://localhost:1919/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}]}'`

**任务1.3：理解进程架构** (4-5小时)
- 阅读并分析`python/minisgl/server/launch.py`文件，理解多进程启动逻辑
- 使用`ps aux | grep minisgl`观察运行时的进程结构
- 修改`--log-level DEBUG`参数，观察不同组件的日志输出

**任务1.4：跟踪请求生命周期** (5-6小时)
- 在`python/minisgl/server/api_server.py`的`generate()`函数中添加日志
- 在`python/minisgl/scheduler/scheduler.py`的`_process_one_msg()`函数中添加日志
- 发起一个简单请求，在日志中跟踪整个流程

**任务1.5：ZMQ通信分析** (4-6小时)
- 阅读`python/minisgl/message/`目录下的消息定义
- 使用`tcpdump`或Wireshark观察ZMQ通信（如果使用TCP）
- 修改一个消息类型，观察系统行为变化

**任务1.6：配置参数实验** (2-3小时)
- 实验不同的`--port`、`--host`配置
- 尝试`--shell`交互模式
- 测试`--tp 2`多GPU配置（如果有条件）

### 阶段二：核心组件深入（预计：30-40小时，2-3周）

**任务2.1：核心数据结构分析** (6-8小时)
- 详细阅读`python/minisgl/core.py`，理解Req、Batch、Context、SamplingParams
- 创建测试脚本，实例化这些数据结构并调用其方法
- 绘制Req对象的生命周期状态图

**任务2.2：调度器prefill阶段分析** (5-6小时)
- 阅读`python/minisgl/scheduler/prefill.py`和`python/minisgl/scheduler/decode.py`
- 在`scheduler.py`的`_schedule_next_batch()`函数中添加日志，观察prefill调度
- 理解ChunkedReq类和Chunked Prefill机制

**任务2.3：调度器decode阶段分析** (5-6小时)
- 跟踪decode阶段的调度逻辑
- 分析`decode_manager.schedule_next_batch()`的实现
- 观察prefill和decode的切换条件

**任务2.4：引擎初始化分析** (4-5小时)
- 阅读`python/minisgl/engine/engine.py`的`__init__()`方法
- 理解模型加载、KV缓存分配、注意力后端初始化的顺序
- 分析`_determine_num_pages()`方法的内存计算逻辑

**任务2.5：推理流程跟踪** (5-6小时)
- 分析`engine.py`的`forward_batch()`方法
- 理解`graph_runner.can_use_cuda_graph()`的条件
- 跟踪`sampler.sample()`的采样过程

**任务2.6：模型实现分析** (5-6小时)
- 阅读`python/minisgl/models/`目录下的模型实现
- 分析Llama和Qwen模型的异同
- 理解权重加载和分片逻辑

### 阶段三：高级特性与优化（预计：30-40小时，2-3周）

**任务3.1：注意力后端对比** (6-8小时)
- 阅读`python/minisgl/attention/`目录下的所有文件
- 分别使用`--attn fa`和`--attn fi`运行，比较性能差异
- 分析`create_attention_backend()`工厂函数的选择逻辑

**任务3.2：FlashAttention实现分析** (5-6小时)
- 详细阅读`python/minisgl/attention/fa.py`
- 理解FlashAttention的forward方法实现
- 分析`prepare_metadata()`的作用

**任务3.3：FlashInfer实现分析** (5-6小时)
- 详细阅读`python/minisgl/attention/fi.py`
- 比较FlashInfer与FlashAttention的接口差异
- 分析张量核心的启用条件

**任务3.4：KV缓存管理分析** (6-8小时)
- 阅读`python/minisgl/kvcache/`目录下的所有文件
- 分别使用`--cache naive`和`--cache radix`运行，观察内存使用差异
- 分析RadixCache的`match_prefix()`和`insert_prefix()`算法

**任务3.5：分布式通信分析** (4-6小时)
- 阅读`python/minisgl/distributed/`目录
- 分析Tensor Parallelism的通信模式
- 理解`enable_pynccl_distributed()`的作用

**任务3.6：CUDA内核分析** (4-6小时)
- 浏览`python/minisgl/kernel/`目录的结构
- 分析一个简单的内核实现，如`index.py`
- 理解TVM-FFI的JIT编译机制

### 阶段四：扩展与贡献（预计：20-30小时，1-2周）

**任务4.1：工具类分析** (4-5小时)
- 阅读`python/minisgl/utils/`目录
- 分析`init_logger()`、`ZmqAsyncPushQueue`、`ZmqAsyncPullQueue`的实现
- 理解环境变量配置系统（`minisgl/env.py`）

**任务4.2：测试框架学习** (5-6小时)
- 运行所有测试：`pytest /home/kason/python_workspace/mini-sglang/tests/ -v`
- 分析一个测试文件的结构，如`tests/core/test_scheduler.py`
- 编写一个简单的测试用例

**任务4.3：基准测试分析** (5-6小时)
- 运行离线基准测试：`python /home/kason/python_workspace/mini-sglang/benchmark/offline/bench.py`
- 运行在线基准测试：`python /home/kason/python_workspace/mini-sglang/benchmark/online/bench_qwen.py`
- 分析基准测试客户端实现（`python/minisgl/benchmark/client.py`）

**任务4.4：Python API分析** (3-4小时)
- 阅读`python/minisgl/llm/llm.py`
- 理解LLM类的高级接口设计
- 对比API server和Python API的使用差异

**任务4.5：功能扩展实践** (6-8小时)
- 添加简单的日志功能
- 修改采样参数处理逻辑
- 尝试添加新的配置选项