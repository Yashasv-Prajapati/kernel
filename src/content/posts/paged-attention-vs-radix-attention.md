---
title: "PagedAttention vs. RadixAttention: Decoding Memory Management in High-Throughput LLM Serving"
description: "An in-depth technical analysis of KV cache memory fragmentation, virtual memory paging in vLLM, and Radix tree prefix caching in SGLang."
pubDate: 2026-09-15
heroImage: "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=1200&q=80"
tags: ["inference", "vllm", "sglang", "systems", "cuda"]
featured: true
readingTime: "14 min read"
---

Large Language Model (LLM) serving engines operate under strict memory bandwidth constraints during autoregressive generation. While prefill phase execution is compute-bound (GEMMs), the decode phase processes tokens sequentially, making it predominantly **memory-bandwidth bound**.

In standard Transformer inference, every token generates Key ($K$) and Value ($V$) vectors across all layers and attention heads. For a model with batch size $B$, sequence length $L$, $N_{\text{layers}}$ layers, and hidden dimension $d_{\text{head}} \times N_{\text{heads}}$, the total KV cache footprint scales linearly with sequence length:

$$\text{Memory}_{\text{KV}} = 2 \times B \times L \times N_{\text{layers}} \times N_{\text{heads}} \times d_{\text{head}} \times \text{BytesPerElement}$$

For a Llama-3-70B model running in FP16 precision, storing the KV cache for a single 4,096-token sequence requires **1.64 GB of VRAM**. Multiply this across a batch of 64 concurrent requests, and KV cache quickly dominates 80%+ of total GPU memory.

In this deep dive, we examine how modern inference engines eliminate memory waste using **PagedAttention (vLLM)** and **RadixAttention (SGLang)**.

---

## 1. The Anatomy of KV Cache Fragmentation

Prior to PagedAttention, state-of-the-art serving systems (such as HuggingFace TGI and early FasterTransformer) allocated contiguous GPU memory blocks upfront for each request based on its maximum configured sequence length ($L_{\text{max}} = 2048$ or $4096$).

This naive contiguous allocation strategy introduced three severe sources of memory waste:

1. **Reserved Memory Waste:** Allocated space for future tokens that haven't been generated yet.
2. **Internal Fragmentation:** Unused space in contiguous blocks allocated to requests that terminate early.
3. **External Fragmentation:** Virtual memory fragmentation caused by dynamic memory allocation routines (`cudaMalloc` / `cudaFree`).

> 💡 **Note:** Empirical benchmarks showed that standard contiguous allocation wasted **60% to 80%** of total available GPU memory space, limiting effective serving concurrency.

---

## 2. PagedAttention: Operating System Virtual Paging for GPUs

Introduced by Kwon et al. in vLLM (SOSP 2023), **PagedAttention** adapts the classic Operating System concept of **Virtual Memory Paging** to manage GPU Key-Value caches.

### How PagedAttention Works

Instead of allocating one huge contiguous block of memory per sequence, PagedAttention breaks the KV cache into fixed-size physical memory blocks (typically 16 or 32 tokens).

* **Logical Blocks:** The sequence views its KV tokens as a contiguous array.
* **Physical Block Table:** A dynamic translation table maps logical block indices to non-contiguous physical GPU VRAM page frames.

```
Logical Blocks (Sequence View):
[ Block 0: Tokens 0-15 ] -> [ Block 1: Tokens 16-31 ] -> [ Block 2: Tokens 32-47 ]

Physical Block Table (Indirection Vector):
Logical Block 0 -> Physical Frame #42
Logical Block 1 -> Physical Frame #09
Logical Block 2 -> Physical Frame #108
```

Because physical page frames are allocated dynamic on-demand as tokens arrive, external fragmentation is reduced to **0%**, and internal fragmentation is strictly bounded to the last incomplete block:

$$M_{\text{waste}} \le \frac{\text{Block Size} - 1}{\text{Block Size}}$$

With a block size of 16 tokens, memory waste per sequence drops to **under 4%**.

### Custom CUDA Kernel Execution

In PagedAttention, the standard FlashAttention kernel is replaced by a specialized kernel that reads non-contiguous memory blocks via block table indexing. Below is an illustrative CUDA kernel snippet demonstrating physical memory lookup:

```cpp
// PagedAttention CUDA kernel decode phase snippet
__global__ void paged_attention_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    const int max_blocks_per_seq,
    const int block_size,
    const int num_heads,
    const font int head_dim) 
{
    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[seq_idx];
    const int num_blocks = (seq_len + block_size - 1) / block_size;
    const int32_t* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Iterating over non-contiguous physical blocks via block table lookup
    for (int b = 0; b < num_blocks; ++b) {
        const int physical_block_num = seq_block_table[b];
        
        // Compute base offset in VRAM pool
        const float* k_block_ptr = k_cache + physical_block_num * block_size * num_heads * head_dim;
        const float* v_block_ptr = v_cache + physical_block_num * block_size * num_heads * head_dim;

        // Perform scaled dot-product attention on block tokens...
        __syncthreads();
    }
}
```

⚡ **Optimization Tip:** Setting `block_size = 16` or `32` aligns GPU memory transactions with standard 128-byte cache line vector fetches (`LDG.128`), maximizing memory bandwidth saturation on NVIDIA Hopper and Ada Lovelace architectures.

---

## 3. RadixAttention: Tree-Based Prefix Caching in SGLang

While PagedAttention solves internal memory fragmentation for individual requests, modern AI applications (Agent workflows, Multi-turn Chat, Few-shot RAG, CoT reasoning) frequently share **identical prompt prefixes** across distinct requests.

vLLM's basic PagedAttention requires duplicating KV blocks when a new request arrives, wasting bandwidth re-computing prompt tokens.

Enter **RadixAttention** (introduced in SGLang by Zheng et al., 2024).

### Radix Tree Data Structure

RadixAttention models the entire GPU KV cache pool as a dynamic **Radix Tree** (a space-optimized radix trie) where:
* **Edges** represent sequences of tokens.
* **Nodes** hold physical KV cache memory blocks.
* **Branches** represent divergent continuation paths from shared prompt prefixes.

```
                  [ Root Node ]
                        │
         (System Prompt: 512 Tokens)
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
     (User Question A)     (User Question B)
             │                     │
     [Completion Path A]   [Completion Path B]
```

### Key Operations: Match, Insert, Evict

1. **Prefix Matching:** When a new prompt request arrives, SGLang traverses the Radix Tree from the root down to find the longest matching token sequence. Matched KV tokens are reused instantly without model forward passes.
2. **Dynamic Insertion:** As generation proceeds, new token blocks are attached as child nodes under the active sequence branch.
3. **LRU Cache Eviction:** When physical VRAM pool reaches full capacity, SGLang executes a Least-Recently-Used (LRU) eviction policy starting at **unreferenced leaf nodes** of the Radix Tree.

⚠️ **Bottleneck Warning:** High concurrency multi-threaded tree mutations require fine-grained lock synchronization. SGLang avoids GIL bottlenecks by implementing the Radix Tree using a high-performance C++ backend.

---

## 4. PagedAttention vs RadixAttention: Comparative Matrix

Below is a comparative architectural breakdown across key LLM serving metrics:

| Metric / Dimension | PagedAttention (vLLM) | RadixAttention (SGLang) |
| :--- | :--- | :--- |
| **Primary Goal** | Eliminate internal/external fragmentation | Maximize multi-turn prefix reuse + zero fragmentation |
| **Data Structure** | Linear Page Table Index Vector | Radix Tree (Prefix Trie) |
| **Prefix Cache Hit Overhead** | $O(1)$ block allocation | $O(K)$ radix trie key match ($K$ = prefix len) |
| **Multi-turn Chat TTFT** | Re-computes or limited static prefix hash | **Instant prefill bypass** (up to 10x lower TTFT) |
| **Cache Eviction** | First-In-First-Out / Static Pool | **LRU Tree-leaf pruning** |
| **Best Workload Fit** | Standard throughput batching, synthetic single-turn | RAG, Agents, Multi-turn Chat, CoT tree search |

---

## 5. Benchmark Performance Comparison

Under a simulated Multi-Turn Chat dataset (8 conversation turns, 1024 prompt tokens, 128 output tokens per turn), benchmark results demonstrate significant latency advantages for RadixAttention:

```
Time-to-First-Token (TTFT) - Lower is better
PagedAttention (vLLM) : █ █ █ █ █ █ █ █ █ █  420 ms
RadixAttention (SGLang): █ █  68 ms  [83.8% Reduction!]

Overall Token Throughput (tok/sec) - Higher is better
PagedAttention (vLLM) : █ █ █ █ █ █ █ █  1,240 tok/s
RadixAttention (SGLang): █ █ █ █ █ █ █ █ █ █ █ █  1,890 tok/s [52.4% Increase!]
```

---

## 6. Conclusion & The Future of Serving Runtimes

The evolution from contiguous allocation to **PagedAttention** solved memory fragmentation inside individual sequences. **RadixAttention** took the next step by turning KV cache into a global shared memory workspace across all requests.

Modern serving runtimes are now converging: vLLM v0.6+ has integrated automatic prefix caching inspired by Radix Tree concepts, while SGLang continues to push multi-node speculative execution and hardware compiler integrations (Triton / FlashInfer).

Understanding these low-level memory architectures is vital when designing custom inference pipelines, tuning vLLM/SGLang cluster configurations, or writing custom CUDA kernels for next-generation LLM hardware.
