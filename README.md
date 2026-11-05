# llama.cpp-moe-streaming

My senior thesis at Princeton, advised by David Wentzlaff: running trillion-parameter mixture-of-experts (MoE) models on a few ordinary PCs. The whole model stays on NVMe flash, and each token pulls in only the experts it actually uses. Built on [llama.cpp](https://github.com/ggml-org/llama.cpp).

**Status: setting up.** Nothing in this fork differs from upstream yet. This page describes the plan, and results will be added as they come in.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/thesis/overview-dark.svg">
  <img alt="NVMe NAND to DRAM to GPU over PCIe on each node, with layers split across 3 to 4 nodes, an expert cache, prefetch and compute/IO overlap" src="docs/thesis/overview-light.svg">
</picture>

## The problem

A trillion parameters at 4 bits is about 500 GB of weights. Each of these nodes has 12 GB of VRAM and 32 GB of RAM, so nearly all of the model has to live on flash. An MoE routes each token through only a few experts, so decode doesn't need every weight, just the active experts that aren't already in RAM or VRAM. That makes decode a flash-bandwidth problem:

```math
\text{tokens/s} \;\lesssim\; \frac{B_{\text{flash}}}{(1 - h)\, S_{\text{active}}}
```

- $S_{\text{active}}$: bytes of expert weights one token touches
- $h$: the fraction of those already cached in RAM or VRAM
- $B_{\text{flash}}$: how fast a node can read from its NVMe drives

My bandwidth model puts sequential decode at around **1.5 tok/s** on this hardware. The goal is to get past **10**.

## What this fork will change

Each change targets one term of that bound.

| Change | What it moves |
| --- | --- |
| An expert cache across RAM and VRAM that keeps hot experts resident | Raises $h$ |
| Prefetching experts for upcoming layers | Hides flash latency |
| Overlapping compute with IO | The GPU stops waiting on reads |
| Reading three Gen5 NVMe drives per node in parallel | Raises $B_{\text{flash}}$ |
| Splitting layers across nodes (pipeline parallelism) | Each node stores and streams only its slice |

## Hardware (planned, 3–4 nodes)

| Per node | |
| --- | --- |
| CPU | Ryzen 5 7600X (Zen 4, AVX-512) |
| GPU | RTX 5070 12 GB, PCIe x16 |
| RAM | 32 GB DDR5-6000 |
| Flash | 3 × 1 TB PCIe Gen5 NVMe |

The small amount of RAM is deliberate. The model can't hide in DRAM, so the results measure the flash tier, not memory capacity. The 5070 won out over the 16 GB 5060 Ti because it gets a full x16 link, which matters when every expert crosses PCIe.

## Evaluation

- **Baseline:** stock llama.cpp on the same nodes with the same model, which isolates what these changes contribute.
- **Comparison:** [Strata](https://github.com/Niko1221/Strata), a hybrid VRAM / RAM / SSD MoE engine. Its test machine is almost identical to one of these nodes.
- **Metric:** sequential decode tok/s, with each change enabled one at a time.

## Results

None yet.

## The code

Once there's something to show, [this compare view](https://github.com/ggml-org/llama.cpp/compare/master...aarushgoradia:llama.cpp-moe-streaming:thesis) shows only what this fork changes. Everything else is stock llama.cpp. See the [upstream README](https://github.com/ggml-org/llama.cpp#readme) for building and running it.
