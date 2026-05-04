# My AI Portfolio
*Engineering advanced AI systems—from autonomous multi-agent systems and scaling reasoning-focused LLMs on multi-node GPU clusters to performance profiling and distilling DeepSeek R1.*

* [AI Agent](./ai-agent)
* [LLM Benchmarking and Profiling](./benchmarking)
* [LLM/Diffusion Distillation & Fine-Tuning](./distillation-finetuning)

---

## [AI Agent](./ai-agent)

Building intelligent **AI agents** that dynamically reason, retrieve, and self-correct—from Agentic RAG with colocated vLLM inference to tool-augmented reasoning on the GAIA benchmark.

<p align="center">
  <img src="images/RAG_vs_Agentic.jpeg" width="100%" />
</p>

**Key projects:**
- **Agentic RAG** — Agent-based retrieval with iterative query refinement, achieving superior accuracy over Standard RAG and standalone LLMs. [Details →](./ai-agent#agentic-retrieval-augmented-generation-rag)
- **Colocated vLLM Inference** — Zero-egress, GPU-cluster deployment with a three-phase hybrid pipeline that collapses latency by an order of magnitude. [Details →](./ai-agent#agentic-rag-with-colocated-vllm-inference)
- **GAIA Benchmark** — Tool-augmented code agent achieving **40%** accuracy, outperforming GPT-4's **14.4%**. [Details →](./ai-agent#ai-agent-tool-augmented-reasoning-on-gaia-benchmark)

[Explore all AI Agent projects →](./ai-agent)

---

## [LLM Benchmarking and Profiling](./benchmarking)

Systematic performance analysis of Transformer architectures—benchmarking FP32 vs. BF16 mixed precision and profiling compute- vs. memory-bound operations in self-attention.

<p align="center">
  <img src="images/benchmark_comparison_training.png" width="49%" />
  <img src="images/benchmark_comparison_inference.png" width="49%" />
</p>

**Key projects:**
- **FP32 vs. BF16 Benchmarking** — BF16 mixed precision delivers up to **6× inference throughput** and unlocks training of larger architectures that fail under FP32. [Details →](./benchmarking#performance-benchmarking-fp32-vs-bf16-mixed-precision)
- **Arithmetic Intensity Profiling** — Reveals why MatMul completes in half the time of Softmax despite **25.6× more FLOPs**, demonstrating the compute-bound vs. memory-bound paradigm. [Details →](./benchmarking#profiling-arithmetic-intensity-matmul-vs-softmax-in-self-attention)

[Explore all Benchmarking projects →](./benchmarking)

---

## [LLM/Diffusion Distillation & Fine-Tuning](./distillation-finetuning)

Advanced post-training and fine-tuning across LLMs and diffusion models—from distilling DeepSeek R1 on multi-node HPC to LoRA-adapted Stable Diffusion.

<p align="center">
  <img src="https://github.com/Wen-ChuangChou/sentiment_analysis/blob/main/pic/radarplot.png?raw=true" alt="Radar plot" width="400"/>
</p>

**Key projects:**
- **DeepSeek R1 Distillation** — Boosted Qwen2.5-Math-7B accuracy from 13.3% to **56.7%** on AIME 2024 via SFT + GRPO across 8 H100 GPUs. [Details →](./distillation-finetuning#distilling-deepseek-r1-for-enhanced-llm-performance)
- **Llama 3 Sentiment Analysis** — Fine-tuned Llama 3.1–8B achieving **81.49%** accuracy on MTEB tweet sentiment. [Details →](./distillation-finetuning#fine-tuning-llama-3-for-sentiment-analysis)
- **Stable Diffusion LoRA** — Fine-tuned SD v2 with LoRA for Naruto-style generation, with **77% training time reduction** via multi-GPU. [Details →](./distillation-finetuning#fine-tuning-stable-diffusion-with-lora)
- **Bike Traffic Prediction** — Graph Attention Networks for urban traffic forecasting; **2nd place** at BTW 2023. [Details →](./distillation-finetuning#predicting-bike-traffic)
- **Speaker Identification** — Transformer/Conformer encoders achieving **91.8%** accuracy. [Details →](./distillation-finetuning#speaker-identification)
- **Anime Face Generator** — Diffusion probabilistic model trained on 71k anime faces. [Details →](./distillation-finetuning#anime-face-generator)

[Explore all Distillation & Fine-Tuning projects →](./distillation-finetuning)
