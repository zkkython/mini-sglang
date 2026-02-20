# 阶段三：高级特性与优化 - 学习笔记

## 目录

| 编号 | 主题 | 文件 | 对应任务 |
|------|------|------|----------|
| 3.1 | [注意力后端对比](./3.1-attention-backends.md) | `attention/` | 任务3.1 |
| 3.2 | [FlashAttention 实现分析](./3.2-flash-attention.md) | `attention/fa.py` | 任务3.2 |
| 3.3 | [FlashInfer 实现分析](./3.3-flash-infer.md) | `attention/fi.py` | 任务3.3 |
| 3.4 | [KV缓存管理分析](./3.4-kv-cache.md) | `kvcache/` | 任务3.4 |
| 3.5 | [分布式通信分析](./3.5-distributed.md) | `distributed/`, `layers/` | 任务3.5 |
| 3.6 | [CUDA 内核分析](./3.6-cuda-kernels.md) | `kernel/` | 任务3.6 |

## 核心学习目标

1. 理解 FlashAttention 与 FlashInfer 两种注意力后端的差异及 Hybrid 模式
2. 掌握 NaiveCache 和 RadixCache 的设计差异与前缀共享优化
3. 理解 Tensor Parallelism 的通信模式 (all_reduce / all_gather)
4. 分析 TVM-FFI JIT 编译机制和 Triton MoE 内核
5. 理解 CUDA 图与注意力后端的协同工作

## 关键源文件索引

```
python/minisgl/
├── attention/
│   ├── __init__.py          # 后端注册表、auto选择、HybridBackend
│   ├── base.py              # BaseAttnBackend/BaseAttnMetadata 抽象接口
│   ├── fa.py                # FlashAttention 后端 (sgl_kernel)
│   ├── fi.py                # FlashInfer 后端
│   └── utils.py             # BaseCaptureData (CUDA图捕获缓冲区)
├── kvcache/
│   ├── __init__.py          # KVCache/CacheManager 工厂函数
│   ├── base.py              # BaseKVCache/BaseCacheManager/BaseCacheHandle
│   ├── mha_pool.py          # MHAKVCache (实际显存分配)
│   ├── naive_manager.py     # NaiveCacheManager (无缓存复用)
│   └── radix_manager.py     # RadixCacheManager (Radix Tree 前缀共享)
├── distributed/
│   ├── __init__.py          # 导出
│   ├── impl.py              # DistributedCommunicator (插件式 all_reduce/all_gather)
│   └── info.py              # DistributedInfo (TP rank/size 全局单例)
├── kernel/
│   ├── __init__.py          # 内核导出
│   ├── utils.py             # TVM-FFI JIT/AOT 加载工具
│   ├── index.py             # indexing 内核 (Embedding 查找)
│   ├── store.py             # store_cache 内核 (KV写入)
│   ├── pynccl.py            # PyNCCL 通信初始化
│   ├── radix.py             # fast_compare_key (Radix Tree 键比较)
│   ├── tensor.py            # 测试用内核
│   ├── moe_impl.py          # MoE Triton 内核入口
│   └── triton/fused_moe.py  # Triton fused_moe_kernel
└── moe/
    ├── __init__.py           # MoE 后端注册表
    ├── base.py               # BaseMoeBackend 抽象接口
    └── fused.py              # FusedMoe 实现 (topk + align + triton)
```
