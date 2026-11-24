---
title: "FlashAttention from First Principles: How Online Softmax and SRAM Tiling Broke the GPU Memory Wall"
description: "A hardware-first, visual breakdown of FlashAttention for software engineers. Learn why standard attention chokes on memory transfers, how Online Softmax works on simple vectors, and how 2D block tiling achieves 3x speedups without approximations."
pubDate: 2026-10-27
tags: ["inference", "flashattention", "cuda", "gpu"]
featured: true
readingTime: "16 min read"
---

If you have ever tried running a long prompt through a Large Language Model and watched it suddenly crash with `CUDA out of memory`, or noticed that generation gets sluggish as conversation history grows, you have run directly into the **Attention Memory Wall**.

When you read about **FlashAttention**, it can sound intimidating—full of references to CUDA warps, shared memory bank conflicts, and fused kernels.

Here is the secret: **FlashAttention is not a new deep learning model, and it changes zero math.**

It is an exact, 100% mathematical drop-in replacement for the standard attention layer. If you pass in the exact same input numbers, you get the exact same output numbers down to the last decimal place. No approximations, no pruning, no quality trade-offs.

What FlashAttention does is solve a physical **computer systems bottleneck**: it reorganizes how the GPU moves numbers between its memory chips so attention runs **2x to 4x faster** while using up to **10x less memory**.

> **Scope Note:** In this deep dive, we focus specifically on the **forward pass** of attention (inference & generation). The exact same tiling and online rescaling principles extend to the backward pass during training.

> ### TL;DR for Skimmers (The Mental Model in 4 Sentences)
>
> 1. **The Problem:** Standard attention creates massive $N \times N$ intermediate matrices ($S$ and $P$) and constantly ships them back and forth across the slow GPU memory bus, burning memory bandwidth and crashing on long prompts.
> 2. **The Blocker:** We cannot simply process attention in small blocks because the softmax denominator requires summing across the entire sequence before any single number can be normalized.
> 3. **The Breakthrough:** *Online Softmax* algebraically rescales previously accumulated sums on the fly whenever a larger number arrives, allowing exact softmax computation in small streaming chunks.
> 4. **The Solution:** FlashAttention keeps small blocks of tokens pinned in fast on-chip SRAM, computes attention incrementally, and immediately discards intermediate matrices—reducing memory transfers from $O(N^2)$ down to $O(N)$.

---

## Act 1: The Crime (Why Context Windows Melt GPUs)

Every software engineer knows the classic time-space tradeoff: you spend a bit of memory to save time, or do extra computation to save RAM.

In Large Language Models, the dominant operation is **Self-Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

At first glance, this looks like two harmless matrix multiplications wrapped around a softmax function.

But look closer at the dimensions. If your prompt contains $N$ tokens:
- $Q$ has shape $(N \times d)$
- $K^T$ has shape $(d \times N)$
- Their product, the **Attention Score matrix $S = Q K^T$**, has shape **$(N \times N)$**

For a short 512-token prompt, $512 \times 512$ is about $262,000$ numbers. Trivial.

Now extend that to modern context windows:
- At $N = 8,192$ tokens: $8,192 \times 8,192 \approx \mathbf{67\text{ million}}$ numbers per head.
- At $N = 32,768$ tokens: $32,768 \times 32,768 \approx \mathbf{1.07\text{ billion}}$ numbers per head.
- At $N = 64,000$ tokens: $64,000 \times 64,000 \approx \mathbf{4.1\text{ billion}}$ numbers per head.

Notice the crime: the memory required to hold this intermediate matrix scales **quadratically ($O(N^2)$)** with sequence length. Double your prompt length, and you quadruple the memory consumed. This is exactly what we mean by the **Attention Memory Wall**—at long context lengths, moving and storing the $N \times N$ matrix becomes far more expensive than computing it.

To understand why this causes GPUs to grind to a halt, we need to look at how computer hardware actually stores and moves data.

---

## Act 2: The Crime Scene (Hardware Reality & Standard Attention)

All computers share the same universal rule of memory:

> **Fast memory is tiny and expensive. Big memory is slow and cheap.**

![Universal Memory Hierarchy Pyramid](/images/memory-pyramid.svg)

In a standard CPU server, this hierarchy looks like:
1. **Registers**: In the CPU core, ultra-fast (sub-nanosecond), holding just a few bytes.
2. **L1 / L2 / L3 Caches**: Right on the silicon chip, holding a few megabytes.
3. **Main Memory (RAM)**: Located inches away across motherboard pins, holding hundreds of gigabytes, but hundreds of times slower to access.

### GPUs Take This Hierarchy to an Extreme

GPUs stretch this exact idea to its limits. On a modern GPU like the **NVIDIA H100 (SXM5)**, we see two key tiers that matter for attention:

1. **High Bandwidth Memory (HBM / VRAM)**:
   - This is the main 80 GB pool of the GPU. It stores model weights, key-value caches, and user inputs.
   - It is physically located off the main GPU computing die.
   - Data must travel across a physical memory bus to reach compute units. Despite its impressive bandwidth, moving bytes here is the expensive part.
2. **SRAM (Static RAM / Shared Memory)**:
   - This is on-chip memory etched directly next to the compute units.
   - It is orders of magnitude faster than HBM and delivers far lower latency.
   - But it is tiny: only about **228 KB** per compute core.

### What is an SM (Streaming Multiprocessor)?

Think of an **SM** as an **independent factory floor**.

An NVIDIA H100 GPU contains **132 independent SMs**. Each factory floor houses:
- Hundreds of arithmetic units and specialized **Tensor Cores** capable of multiplying matrices at blazing speeds.
- Its own private **228 KB SRAM workbench**.

The factory workers (GPU threads) can grab parts from their local 228 KB workbench almost instantaneously. But if they have to wait for a delivery truck to haul parts from the off-chip HBM warehouse, the entire factory floor sits completely idle.

> **The Universal Rule of Modern GPU Systems:**
> Compute is essentially free. Moving data across the memory bus is what burns time. If a GPU's compute cores are starved waiting for bytes to arrive from HBM, the hardware goes to waste.

Now that we know data is expensive to move, let's see what attention actually does with it.

---

### The Inputs: What are $N$ and $d$?

Before looking at GPU execution, let's clarify our input notation:

- **$N$ (Sequence Length)**: The number of tokens (words/sub-words) in your prompt.
- **$d$ (Head Dimension)**: The number of feature dimensions per token inside a single attention head (typically $d = 64$ or $d = 128$).

In an LLM with 32 attention heads, each head operates **completely independently**. We can study the algorithm by looking at just one head with its three input matrices:
- **Query ($Q$)**: Shape $(N \times d)$
- **Key ($K$)**: Shape $(N \times d)$
- **Value ($V$)**: Shape $(N \times d)$

---

### How Standard Attention Executes: The 3 CUDA Kernels

In GPU programming, a **CUDA kernel** is simply *a function that runs on the GPU across thousands of parallel threads, launched by the CPU*.

When deep learning frameworks (like native PyTorch prior to FlashAttention) compute attention:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

they break it down into **three separate CUDA kernels** (functions launched by the CPU to run on the GPU) executed sequentially:

1. **Kernel 1 (Matrix Multiplication)**: Computes $S = \frac{Q K^T}{\sqrt{d}}$.
2. **Kernel 2 (Softmax)**: Computes $P = \text{softmax}(S)$.
3. **Kernel 3 (Matrix Multiplication)**: Computes $O = P V$.

---

### How Data is Read from HBM: Tiles vs. Numbers

When the GPU executes these operations, does it read floating-point numbers one by one, or entire matrices at once?

Because the fast SRAM on each SM is only **228 KB**, it cannot fit an entire $N \times N$ matrix. Modern GPUs read and write data in **small 2D tiles** (e.g., $16 \times 16$ or $32 \times 32$ floats):

1. In **Kernel 1**, GPU threads load a tile of rows from $Q$ and a tile of columns from $K$ from HBM into SRAM, use Tensor Cores to compute their dot products, and immediately write that sub-tile of $S$ out to HBM.
2. In **Kernel 2**, threads read rows of $S$ in tiles from HBM into SRAM, compute row maximums and sums of exponentials to produce tiles of probability matrix $P$, and write those tiles of $P$ back out to HBM.
3. In **Kernel 3**, threads read tiles of $P$ and tiles of $V$ from HBM into SRAM, compute output tiles, and write tiles of $O$ to HBM.

Here is the fatal bottleneck: **GPU hardware guarantees zero persistence of SRAM between separate kernel launches.**

Even though data is processed in small tiles inside each kernel, because Kernel 1, Kernel 2, and Kernel 3 are separate programs, **every single element of the gigantic $N \times N$ score matrix $S$ and probability matrix $P$ must be flushed out to slow off-chip HBM** and re-read back across the memory bus!

![Standard Attention 3-Kernel HBM Round Trips](/images/standard-attention-flow.svg)

Notice what happened: the GPU hauled the gigantic $N \times N$ matrices across the slow memory bus **three separate times**!

---

### The $N^2$ Memory Explosion

Let's put real numbers to this disaster. Suppose we run a model with 32 attention heads in 16-bit precision (2 bytes per number). Storing both $S$ and $P$ requires:

$$\text{Memory} = 2 \times 32 \times (N^2 \times 2) \text{ bytes} = 128 \times N^2 \text{ bytes}$$

| Sequence Length ($N$) | Elements per Matrix | Total Memory for $S + P$ | Status on 80 GB GPU |
| :--- | :--- | :--- | :--- |
| **$2,048$** | 4.19 Million | **0.54 GB** | Runs comfortably |
| **$8,192$** | 67.1 Million | **8.59 GB** | High memory pressure |
| **$32,768$** | 1.07 Billion | **137.4 GB** | <strong class="text-rose-600 dark:text-rose-400">CRASH: Out of Memory!</strong> |
| **$64,000$** | 4.10 Billion | **524.3 GB** | <strong class="text-rose-600 dark:text-rose-400">CRASH: Exceeds VRAM by 6.5x</strong> |

![N^2 Memory Explosion Bar Chart](/images/n2-explosion.svg)

At 32k tokens, standard attention demands **137 GB of memory just for scratch matrices**—on an 80 GB card!

---

### The Natural Engineering Question: Why Not Just Tile Attention?

Looking at this memory explosion, every software engineer arrives at the same immediate thought:

> *"Wait. Why don't we just break $Q$, $K$, and $V$ into small blocks that fit entirely inside the 228 KB SRAM, do the math locally, and avoid creating the gigantic $N \times N$ matrices in HBM altogether?"*

This technique is called **tiling**, and it is the standard way GPUs accelerate matrix multiplications. On paper, it sounds like an obvious fix.

So why can't we just tile attention like a normal matrix multiplication?

---

## Act 3: The Clue (The Softmax Blocker)

> **Matrix multiplies are easy to tile. Softmax is the problem. Here's why.**

When you multiply two matrices $C = A \cdot B$, every element $C_{ij}$ is simply the dot product of row $i$ from $A$ and column $j$ from $B$:

$$C_{ij} = \sum_k A_{ik} B_{kj}$$

Because addition is associative and commutative, you can divide $A$ and $B$ into arbitrary small tiles. 

![Why Matrix Multiplication Tiles Easily](/images/matrix-mult-tiling.svg)

As shown above:
- To compute sub-block $C_{11}$, you only need row-tile $A_1$ and col-tile $B_1$.
- You never need to know what numbers exist in $A_2$ or $B_2$.
- Each tile can be computed completely inside SRAM and accumulated independently without global coordination!

So why doesn't this work for attention?

Look at the mathematical definition of **Softmax** on a row vector $x = [x_1, x_2, \dots, x_N]$:

$$\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^N e^{x_j - m}} \quad \text{where } m = \max(x_1, \dots, x_N)$$

Look at the denominator:

$$l = \sum_{j=1}^N e^{x_j - m}$$

To normalize even the **very first number** $\text{softmax}(x_1)$, you must divide by the sum of exponentials of **every single element in the entire row**!

And to compute those exponentials stably without numeric overflow, you must subtract the maximum value $m$ of the **entire row**.

Here is the roadblock:
- If you split a row of 8,192 tokens into chunks, you cannot normalize Chunk 1 because you don't know if a later chunk contains a number larger than anything in Chunk 1.
- You cannot compute the denominator sum of Chunk 1 because you haven't seen the numbers in the remaining chunks yet.

It looks like an impossible barrier: **softmax demands global row visibility.**

If you must wait for the entire row before you can compute softmax, you are forced to materialize the full $N \times N$ matrix in memory before moving to the next step.

Or are you?

---

## Act 4: The Breakthrough (Online Softmax: Streaming 1 Element at a Time)

We have the goal: process data in small SRAM blocks. But softmax blocks us. 

To break through this wall, we need an algebraic trick that lets us compute softmax incrementally across streaming inputs without ever buffering the entire array.

Forget about attention, queries, keys, and 2D matrices for a moment. Grab a piece of paper and a pencil. We are going to solve this on a pure 1D array of 4 numbers, **streaming them one element at a time**:

$$x = [2, 4, 1, 5]$$

### The Amnesia Analogy: The Sticky Note

Imagine you have amnesia. You are only allowed a **tiny sticky note** with two slots on it:
1. $m$: The running maximum seen so far.
2. $l$: The running sum of exponentials seen so far.

Once you process an element, you must **throw that element away**. You can only keep your sticky note!

![Online Softmax Streaming 1 Element at a Time](/images/online-softmax-merge.svg)

---

### Step 1: Element $x_1 = 2$ Arrives

This is the very first number you see.
1. **Running maximum**: $m = 2$
2. **Running sum of exponentials**:
   $$l = e^{2 - 2} = e^0 = \mathbf{1.0}$$

You write on your sticky note:
$$\mathbf{Sticky\ Note}: \quad m = 2, \quad l = 1.0$$

You throw away the number $2$. It is gone from memory.

---

### Step 2: Element $x_2 = 4$ Arrives (New Maximum!)

Now the second number arrives: $4$.
1. **Compare with sticky note**: $4 > 2$. We found a new maximum!
   $$m_{\text{new}} = \max(2, 4) = \mathbf{4}$$

Now comes the crisis:
The sum on your sticky note ($l = 1.0$) was computed using $m = 2$:
$$l_{\text{old}} = e^{2 - 2}$$

To merge with the new number, that old term *should* have been computed using the new maximum $m_{\text{new}} = 4$:
$$l_{\text{correct}} = e^{2 - 4} = e^{-2}$$

Must you reload the number $2$ from memory? **No! Look at the algebra:**

$$e^{x_i - m_{\text{new}}} = e^{x_i - m_{\text{old}} + m_{\text{old}} - m_{\text{new}}} = e^{x_i - m_{\text{old}}} \cdot e^{m_{\text{old}} - m_{\text{new}}}$$

Because multiplication distributes over addition:

$$\mathbf{Corrected\ Old\ Sum} = l_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}}$$

To update your old sum to the new maximum, you simply multiply your previous sum by the single scalar correction factor:

$$\alpha = e^{m_{\text{old}} - m_{\text{new}}} = e^{2 - 4} = e^{-2} \approx 0.1353$$

Let's execute the update:
1. **Rescale the old sum**:
   $$l_{\text{old}}^{\text{rescaled}} = 1.0 \times e^{-2} \approx \mathbf{0.1353}$$
2. **Add the new element's contribution** ($e^{4 - 4} = e^0 = 1.0$):
   $$l_{\text{new}} = 0.1353 + 1.0 = \mathbf{1.1353}$$
3. **Update your sticky note**:
   $$\mathbf{Sticky\ Note}: \quad m = 4, \quad l = 1.1353$$

You throw away the number $4$.

---

### Step 3: Element $x_3 = 1$ Arrives (Smaller Number)

Now the third number arrives: $1$.
1. **Compare with sticky note**: $1 < 4$. The maximum did **not** change!
   $$m_{\text{new}} = \max(4, 1) = \mathbf{4}$$
2. **Correction factor**:
   $$\alpha = e^{4 - 4} = e^0 = 1.0$$
   Because the max did not change, **no rescaling of the old sum is needed!**
3. **Add the new element's contribution** ($e^{1 - 4} = e^{-3} \approx 0.0498$):
   $$l_{\text{new}} = 1.1353 + e^{-3} \approx 1.1353 + 0.0498 = \mathbf{1.1851}$$
4. **Update your sticky note**:
   $$\mathbf{Sticky\ Note}: \quad m = 4, \quad l = 1.1851$$

You throw away the number $1$.

---

### Step 4: Element $x_4 = 5$ Arrives (Final Maximum)

Now the fourth number arrives: $5$.
1. **Compare with sticky note**: $5 > 4$. A new maximum!
   $$m_{\text{new}} = \max(4, 5) = \mathbf{5}$$
2. **Rescale the old sum** by $\alpha = e^{4 - 5} = e^{-1} \approx 0.3679$:
   $$l_{\text{old}}^{\text{rescaled}} = 1.1851 \times e^{-1} \approx 1.1851 \times 0.3679 = \mathbf{0.4360}$$
3. **Add the new element's contribution** ($e^{5 - 5} = e^0 = 1.0$):
   $$l_{\text{new}} = 0.4360 + 1.0 = \mathbf{1.4360}$$
4. **Update your sticky note**:
   $$\mathbf{Sticky\ Note}: \quad m = 5, \quad l = 1.4360$$

---

### Step 5: Verify Against Standard Softmax

Let's check if our streaming result matches standard 3-pass softmax computed over the whole array $[2, 4, 1, 5]$ all at once with global max $m = 5$:

$$l_{\text{standard}} = e^{2-5} + e^{4-5} + e^{1-5} + e^{5-5} = e^{-3} + e^{-1} + e^{-4} + e^0$$
$$l_{\text{standard}} \approx 0.0498 + 0.3679 + 0.0183 + 1.0 = \mathbf{1.4360}$$

**It is an exact mathematical match down to the last decimal point!**

We computed the exact global denominator sum by streaming numbers one by one without ever holding more than 1 element in memory at any time.

---

### The Code: Standard vs. Online Softmax in Python

Here is how the two approaches compare in clean, runnable Python:

```python
import numpy as np

# Standard 3-Pass Softmax (Requires buffering full array)
def standard_softmax(x):
    m = np.max(x)                # Pass 1: Global row maximum
    unnormalized = np.exp(x - m) # Pass 2: Exponentials
    l = np.sum(unnormalized)     # Pass 2: Global sum
    return unnormalized / l      # Pass 3: Normalization

# Online 1-Pass Softmax (Streams 1 element at a time)
def online_softmax_demo(stream):
    m = -float('inf')
    l = 0.0
    
    for x_i in stream:
        m_new = max(m, x_i)
        
        # Rescale previous sum if max increased
        rescale = np.exp(m - m_new) if m != -float('inf') else 0.0
        
        l = (l * rescale) + np.exp(x_i - m_new)
        m = m_new
        
    return m, l

# Test on [2, 4, 1, 5] streamed one element at a time
stream = [2.0, 4.0, 1.0, 5.0]
m_final, l_final = online_softmax_demo(stream)
print(f"Online result: max={m_final}, sum={l_final:.4f}")
# Output: Online result: max=5.0, sum=1.4360
```

---

## Act 5: The Application (Tiling 2D Attention Without the Jump)

Now that we know how to merge softmax statistics across chunks, let's apply this to attention's 2D matrices.

> **Recap:** We can merge running maximums and sums across streaming inputs using a scalar rescale factor ($e^{m_{\text{old}} - m_{\text{new}}}$). But attention doesn't just produce probabilities—it produces weighted sums of $V$. Notice that $O_i = \sum_{j=1}^N P_{ij} V_j = \frac{1}{l}\sum_{j=1}^N e^{S_{ij} - m} V_j$. So if $m$ and $l$ change when we see a new block with a larger max, we need to rescale both our running sum $l$ and our running weighted sum $O$ by the same factor. Let's see how this works.

Look closely at the formula for row $i$ of the attention output:

$$O_i = \sum_{j=1}^N P_{ij} V_j = \sum_{j=1}^N \left(\frac{e^{S_{ij} - m}}{l}\right) V_j = \frac{1}{l} \sum_{j=1}^N e^{S_{ij} - m} V_j$$

Notice the structure:
- $O_i$ is simply a **weighted sum of rows from $V$**.
- The weights are the exact same exponentials $e^{S_{ij} - m}$ that make up the denominator $l$!

This means: **The exact same factor that rescales the sum $l$ also rescales the accumulated output vector $O$!**

---

### Fixing the Jump: Row-by-Row Elementwise Rescaling

In formal papers, you often see this written using diagonal matrices:

$$O \leftarrow \text{diag}(\alpha) \cdot O$$

To a software engineer, multiplying by a diagonal matrix looks like an expensive, mysterious operation. 

In reality, it is dirt simple: **it is just elementwise multiplication row by row.**

![Row by Row Elementwise Rescaling of Accumulator O](/images/accumulator-rescale.svg)

Here is what happens inside the GPU for row $i$:
1. When a new tile arrives with a higher row maximum $m_{i,\text{new}}$, you compute the scalar:
   $$\alpha_i = e^{m_{i,\text{old}} - m_{i,\text{new}}}$$
2. You multiply **every number in row $i$ of your running accumulator $O$ by $\alpha_i$** (elementwise), then add the new tile's weighted contribution:
   $$O_{i} \leftarrow \alpha_i \cdot O_{i} + \sum_{j \in \text{new tile}} e^{S_{ij} - m_{i,\text{new}}} V_j$$

That is all it is! No matrix inversion, no diagonal multiplications—just multiplying the numbers in row $i$ by a scalar before adding the new tile's weighted values. (If you're curious, this is mathematically equivalent to left-multiplying by $\text{diag}(\alpha)$.)

---

### A Concrete Walkthrough: $N = 8, d = 4$

Let's trace a concrete example with real numbers:
- **Sequence Length**: $N = 8$ tokens
- **Head Dimension**: $d = 4$ features
- **Block Size**: We choose block size $4 \times 4$.

> **Why Block Size 4×4?** In real models, block sizes are typically $64 \times 64$ or $128 \times 128$. The rule of thumb: you pick a block size that is *just large enough to saturate the Tensor Cores and fit comfortably inside the 228 KB SRAM, but small enough that it never spills over into slow HBM.* For our toy $N=8, d=4$ example, $4 \times 4$ divides the sequence into two clean halves.

This cuts our $(8 \times 4)$ matrices into **two $4 \times 4$ blocks**:
- $Q$ is split into $Q_1$ (Tokens 0–3) and $Q_2$ (Tokens 4–7).
- $K$ is split into $K_1$ and $K_2$.
- $V$ is split into $V_1$ and $V_2$.

![Matrix Slicing into 4x4 Blocks](/images/flash-attention-tiling-n8-d4.svg)

Let's watch how one SM computes the output for **Tokens 0–3 ($Q_1$)**:

![Tiling SRAM vs HBM Lifecycle](/images/tiling-sram-hbm.svg)

---

### Step 1: Pin $Q_1$ in SRAM and Initialize Trackers
1. Load $Q_1$ ($4 \times 4$) from HBM into fast on-chip SRAM. **$Q_1$ will stay pinned in SRAM for the entire loop.**
2. Initialize three small tracking structures in SRAM:
   - Running row maximum vector: $m = [-\infty, -\infty, -\infty, -\infty]^T$
   - Running row sum vector: $l = [0, 0, 0, 0]^T$
   - Output accumulator matrix: $O_1 = \mathbf{0}_{4 \times 4}$ (all zeros)

---

### Step 2: Stream in Block 1 ($K_1, V_1$)
1. Load $K_1$ ($4 \times 4$) and $V_1$ ($4 \times 4$) from HBM into SRAM.
2. Compute the first mini score tile inside SRAM:
   $$S_{11} = \frac{Q_1 K_1^T}{\sqrt{4}} \quad (4 \times 4)$$
3. Compute row maximums $m_1$ and unnormalized exponentials $\tilde{P}_{11} = \exp(S_{11} - m_1)$, and sum each row to get $l_1$.
4. Multiply by $V_1$ to accumulate into $O_1$:
   $$O_1 = \tilde{P}_{11} \cdot V_1 \quad (4 \times 4)$$

> ### THE PIVOTAL MOMENT: DISCARDING FROM SRAM
>
> We are completely done with $K_1$, $V_1$, $S_{11}$, and $\tilde{P}_{11}$.
>
> **We discard them from SRAM immediately. Nothing was written to HBM!**

---

### Step 3: Stream in Block 2 ($K_2, V_2$) & Rescale on the Fly
1. Keep $Q_1, m_1, l_1, O_1$ sitting in SRAM.
2. Load $K_2$ ($4 \times 4$) and $V_2$ ($4 \times 4$) from HBM into SRAM.
3. Compute the second mini score tile:
   $$S_{12} = \frac{Q_1 K_2^T}{\sqrt{4}} \quad (4 \times 4)$$
4. Find the row maximums of this new block: $m_2 = \text{row\_max}(S_{12})$.
5. **Apply the Online Rescale**:
   - For each row $i$, find the new max: $m_{i,\text{new}} = \max(m_{1,i}, m_{2,i})$.
   - Compute row rescale scalar: $\alpha_i = \exp(m_{1,i} - m_{i,\text{new}})$.
   - Update running sum: $l_{i,\text{new}} = (l_{1,i} \cdot \alpha_i) + \sum_{j} \exp(S_{12, ij} - m_{i,\text{new}})$.
   - **Rescale accumulator $O_1$**: Multiply each row $i$ of $O_1$ by $\alpha_i$, then add the new tile's weighted contribution:
     $$O_{1, i} \leftarrow (\alpha_i \cdot O_{1, i}) + \sum_{j} \exp(S_{12, ij} - m_{i,\text{new}}) V_{2, j}$$

> ### DISCARD BLOCK 2 FROM SRAM
>
> We are done with $K_2$, $V_2$, $S_{12}$, and $\tilde{P}_{12}$.
>
> **We discard them from SRAM. Zero bytes written to HBM!**

---

### Step 4: Write Final Output $O_1$ to HBM (Only Once!)
1. Divide each row $i$ of $O_1$ by its final normalization sum $l_{i,\text{new}}$:
   $$O_{1, i} = \frac{O_{1, i}}{l_{i,\text{new}}}$$
2. **Write the finished $4 \times 4$ output tile $O_1$ out to HBM.**

Now repeat the exact same process for $Q_2$ (Tokens 4–7) to compute $O_2$.

> ### The Grand Result: Zero Bytes Written for Intermediate Matrices
>
> • Standard Attention HBM writes for $S$ and $P$: **$2 \times (N \times N)$ elements**
>
> • FlashAttention HBM writes for $S$ and $P$: **0 BYTES**
>
> • Total HBM memory traffic drops from **$O(N^2)$** down to **$O(N)$**.

---

## Act 6: The Verdict (Why Doing More Math is 3x Faster)

> **Recap:** We went from writing $O(N^2)$ values across the memory bus to writing only $O(N)$ values. But we also added extra multiplications to rescale our accumulators. So why is this actually faster if we do *more* operations?

Notice what we had to do in Step 3: every time a new tile arrived with a larger maximum, we had to multiply our running accumulator by the scalar $\alpha_i$.

If you strictly count floating-point operations (FLOPs), FlashAttention actually performs **more math** than standard attention due to these online rescaling multiplications!

So why does it run **2x to 4x faster** in real-world benchmarks?

![Memory Highway Comparison](/images/traffic-comparison.svg)

### The Hardware Reality: Math is Free, Memory is the Bottleneck

Your brain naturally wants to say:
> *"Do less work to run faster."*

The GPU architecture responds:
> *"Do more local work in SRAM to avoid moving data across the bus."*

In computer systems, this is captured by the **Roofline Model**:

$$\text{Arithmetic Intensity} = \frac{\text{Math Operations (FLOPs)}}{\text{Memory Transferred (Bytes)}}$$

The key takeaway is simple: **if you do more math per byte moved from HBM, your algorithm becomes faster on GPUs.** Standard attention does very little math relative to how many bytes it shuttles for $S$ and $P$. FlashAttention does a bit more local math (the rescaling), but dramatically reduces bytes moved by keeping everything in SRAM.<sup id="fnref:1"><a href="#fn:1">1</a></sup>

- **Standard Attention** was severely **memory-bandwidth bound**. Its compute cores spent most of their time stalled waiting for the gigantic $S$ and $P$ matrices to crawl across the memory bus.
- **FlashAttention** keeps the computation entirely inside high-speed SRAM. Even though it performs extra multiplications to rescale accumulators, SRAM bandwidth is nearly **10x faster** than HBM, and the memory bus stays far less congested.

By eliminating the intermediate memory traffic jam, FlashAttention transforms an algorithm that once stalled GPU memory buses into a stream of uninterrupted, lightning-fast Tensor Core operations.

---

## The Takeaways: What Every Engineer Should Remember

1. **FlashAttention changes zero math**: It produces the exact same floating-point output as standard attention.
2. **The bottleneck was memory traffic, not compute**: Standard attention choked because it wrote and re-read gigantic $(N \times N)$ matrices to slow HBM across 3 separate CUDA kernels.
3. **Softmax seemed to block block-tiling**: Because softmax requires the maximum and denominator sum of the entire sequence, it appeared impossible to process in independent chunks.
4. **Online Softmax solved the blocker**: By tracking a running maximum and running sum, and rescaling previous accumulators by $e^{m_{\text{old}} - m_{\text{new}}}$ when a larger number appears, softmax can be computed in streaming tiles.
5. **Intermediate matrices are discarded**: FlashAttention computes score tiles directly in fast on-chip SRAM, updates accumulators, and discards the tiles immediately—reducing memory traffic from $O(N^2)$ to $O(N)$.

---

<div class="not-prose mt-12 pt-6 border-t border-neutral-200 dark:border-neutral-800 text-xs text-neutral-500 dark:text-neutral-400 space-y-2">
<p id="fn:1"><sup>1</sup> An NVIDIA H100 SXM5 boasts ~989 TFLOPS of FP16 Tensor Core compute against 3.35 TB/s of HBM3 bandwidth, yielding a roofline knee of roughly ~295 FLOPs per byte. Any algorithm with lower arithmetic intensity is strictly memory-bandwidth bound.</p>
</div>
