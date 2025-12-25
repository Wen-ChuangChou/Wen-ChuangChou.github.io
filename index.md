# My AI Portfolio
*Engineering advanced LLMs—from benchmarking and profiling performance to distilling DeepSeek R1.*

* [LLM Benchmarking and Profiling](#performance-benchmarking-fp32-vs-bf16-mixed-precision)
* [AI Agent](#ai-agent--agentic-retrieval-augmented-generation-rag)
* [LLM/Diffusion Distillation & Fine-Tuning](#distilling-deepseek-r1-for-enhanced-llm-performance)

---

## Performance Benchmarking: FP32 vs. BF16 Mixed Precision

### Brief Technique & Impact 
The benchmarks evaluate a modern decoder-only Transformer, spanning five configurations from a "Small" base (12 layers, 768 hidden dimension) up to a 2.7B parameter model. Adopting BF16 mixed precision boosts inference throughput up to 6x and enables training larger, memory-intensive architectures that otherwise fail with full precision due to Out-of-Memory (OOM) constraints.

<p align="center">
  <img src="https://github.com/Wen-ChuangChou/Wen-ChuangChou.github.io/blob/master/images/benchmark_comparison_training.png?raw=true" width="49%" />
  <img src="https://github.com/Wen-ChuangChou/Wen-ChuangChou.github.io/blob/master/images/benchmark_comparison_inference.png?raw=true" width="49%" />
</p>

### Performance Highlights
The benchmarks demonstrate BF16's superior scalability. In inference, the largest 2.7B model achieves a nearly 600% speedup, jumping from 6.6k to 38.6k tokens/second. The small model (128M parameters) sees throughput nearly triple, reaching over 400k tokens/second. Training benefits are equally critical; while the 128M model trains 2.8x faster with BF16, the technique’s true value is unlocking larger architectures. FP32 fails to train the 'large' configuration due to memory limits, whereas BF16 handles it successfully at 24.8k tokens/second, proving essential for resource-constrained high-performance tasks.

---

## AI agent : Agentic Retrieval-Augmented-Generation (RAG)

My work on **Agentic RAG** significantly enhances Large Language Model (LLM) performance for complex information-seeking. This project integrates intelligent **AI agents** into the RAG pipeline, yielding remarkably more accurate, robust, and contextually rich responses than traditional RAG.  

### Brief Technique & Impact
I developed an **AI agent** (using the `smolagent` package) capable of dynamic decision-making, iterative query reformulation, and intelligent document evaluation. A key contribution is an **optimized parallel processing pipeline** for efficient FAISS-based vector database creation from technical documentation. This framework fundamentally improves LLM output grounding through advanced reasoning and self-correction.

### Performance Highlights 
Evaluated on a technical Q&A dataset, Agentic RAG consistently demonstrated superior accuracy across various LLMs compared to both Standard RAG and standalone LLM performance:
  
<div align="center">
  <table>
    <thead>
      <tr>
        <th align="center">Model</th>
        <th align="center">Agentic RAG</th>
        <th align="center">Standard RAG</th>
        <th align="center">LLM Only</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td align="center">Gemini-1.5-flash</td>
        <td align="center">91.5%</td>
        <td align="center">85.4%</td>
        <td align="center">35.4%</td>
      </tr>
      <tr>
        <td align="center">Gemini-2.0-flash</td>
        <td align="center">90.8%</td>
        <td align="center">85.4%</td>
        <td align="center">64.1%</td>
      </tr>
      <tr>
        <td align="center">Gemini-2.5-flash</td>
        <td align="center">90.8%</td>
        <td align="center">86.2%</td>
        <td align="center">63.8%</td>
      </tr>
    </tbody>
  </table>
</div>

<p align="center"><b>All values above are accuracy scores (in %)</b></p>


More details can be found in the project repository on [GitHub](https://wen-chuangchou.github.io/Agentic_RAG/).

---

## AI Agent: Tool-Augmented Reasoning on GAIA Benchmark

I developed a tool-augmented AI code agent using the `smolagents` framework to tackle complex, agent-evaluating questions from the GAIA benchmark.  
This system achieved a **40%** correct answer rate—substantially outperforming GPT-4, which reached **14.4%** under the same conditions.

> **Note:** This project is currently under active development to further improve accuracy and generalization.

---

## Distilling DeepSeek R1 for Enhanced LLM Performance

This project showcases a successful methodology for significantly enhancing large language model performance through advanced **fine-tuning** in a distributed HPC environment.

### Brief Technique & Impact

This work focused on post-training weaker LLMs by fine-tuning the Qwen2.5 model using high-quality data distilled from DeepSeek R1. Employing Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) techniques across **8 H100 GPUs** distributed over **2 HPC nodes**, a key aspect of this project involved learning, optimizing, and simplifying the deployment of the training workflow for common HPC setups.

### Performance Highlights

The rigorous fine-tuning process yielded substantial gains, boosting the Qwen2.5-Math-7B-Instruct model's `pass@1` accuracy on the AIME 2024 benchmark from 13.3% to a remarkable **56.7%**, and on GPQA Diamond from 28.3% to **54.5%**. This demonstrates the effectiveness of the distilled data approach in bringing weaker LLMs closer to DeepSeek R1's performance.

<div align="center">

<table>
  <thead>
    <tr>
      <th style="text-align:center;">Model</th>
      <th style="text-align:center;">AIME 2024<br>pass@1</th>
      <th style="text-align:center;">MATH-500<br>pass@1</th>
      <th style="text-align:center;">GPQA Diamond<br>pass@1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">Qwen2.5-Math-7B-Instruct<br> (Original)</td>
      <td style="text-align:center;">13.3</td>
      <td style="text-align:center;">80.2</td>
      <td style="text-align:center;">28.3</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-Math-7B-Instruct<br>(Fine-tuned on DeepSeek R1 distilled data)</td>
      <td style="text-align:center;">56.7</td>
      <td style="text-align:center;">89.8</td>
      <td style="text-align:center;">54.5</td>
    </tr>
    <tr>
      <td style="text-align:center;">DeepSeek-R1-Distill-Qwen-7B (Teacher)</td>
      <td style="text-align:center;">53.3</td>
      <td style="text-align:center;">93.2</td>
      <td style="text-align:center;">53.0</td>
    </tr>
  </tbody>
</table>

</div>

More details can be found in the project repository on [GitHub](https://wen-chuangchou.github.io/Open-R1/).<br><br> 

---

## Fine-Tuning Llama 3 for Sentiment Analysis

This project fine-tunes **Llama 3.1–8B Instruct** to perform sentiment classification on short-form text, such as tweets. The model learns to identify sentiments—**positive**, **neutral**, or **negative**—using the `tweet_sentiment_extraction` subset from the **MTEB benchmark**.

### Technique & Impact

Using instruction-style prompts and a streamlined training pipeline, the model was fine-tuned to produce accurate, single-word sentiment predictions. This significantly improved performance and makes the model well-suited for real-world applications like social media monitoring and customer feedback analysis.

### Performance Highlights

On the MTEB tweet sentiment test set, the fine-tuned model achieved a notable accuracy gain:

<p align="center"><b>Accuracy on MTEB Tweet Sentiment Classification</b></p>

| **Model**                 | **Accuracy (%)** |
|:-------------------------:|:----------------:|
| Llama 3.1–8B (zero-shot)  |      63.41        |
| Llama 3.1–8B (fine-tuned) |    **81.49**      |

<p align="center">
  <img src="https://github.com/Wen-ChuangChou/sentiment_analysis/blob/main/pic/radarplot.png?raw=true" alt="Radar plot showing model performance" width="400"/>
</p>

More details can be found in the project repository on [GitHub](https://wen-chuangchou.github.io/Sentiment-Analysis/).

---
## Fine-Tuning Stable Diffusion with LoRA

This project fine-tunes **Stable Diffusion v2** (by Stability AI) using the **Hugging Face Diffusers** library to generate images in a customized visual style—in this case, the Naruto anime aesthetic.

### Technique & Impact

The fine-tuning was performed using **LoRA** (Low-Rank Adaptation), which adapts pretrained diffusion models by inserting trainable low-rank matrices into existing weights, significantly reducing memory and compute requirements. The process was accelerated using **8× H100 GPUs** across **2 HPC nodes**, achieving a **77% reduction in training time** compared to single-GPU training.

We used the [`lambdalabs/naruto-blip-captions`](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) dataset to teach the model the distinct visual characteristics of Naruto anime.

#### Dataset Examples


<p float="left" align="center">
  <img src="https://github.com/Wen-ChuangChou/stable-diffusion-fine-tuning/raw/main/images/naruto_image1.jpg" alt="Naruto Example 1" width="220"/>
  &nbsp;
  <img src="https://github.com/Wen-ChuangChou/stable-diffusion-fine-tuning/raw/main/images/naruto_image2.jpg" alt="Naruto Example 2" width="220"/>
</p>

### Performance Highlights

**Prompt:**  
*A detailed portrait of Hello Kitty, rendered in the style of Naruto anime, with a blue background.*

| Model                     | Output Example |
|:--------------------------:|:----------------:|
| **Base Stable Diffusion** | <img src="https://github.com/Wen-ChuangChou/stable-diffusion-fine-tuning/raw/main/images/Hello_Kitty_naruto_base.png" alt="Base SD" width="220"/> |
| **LoRA Fine-Tuned Model** | <img src="https://github.com/Wen-ChuangChou/stable-diffusion-fine-tuning/raw/main/images/Hello_Kitty_lora.png" alt="LoRA Output 1" width="220"/> <img src="https://github.com/Wen-ChuangChou/stable-diffusion-fine-tuning/raw/main/images/Hello_Kitty_lora2.png" alt="LoRA Output 2" width="220"/> |

The base model fails to capture Naruto's stylistic elements, while the fine-tuned model successfully generates images in the correct anime style.

> **Note:** This project is currently under active development to further improve accuracy and generalization.

---

## Predicting Bike Traffic  
I implemented **Graph Attention Networks** to predict bike traffic volume using social and environmental data. The models were trained separately on datasets from Dresden, Leipzig, and Hamburg. The following plot illustrates the results across the three cities:

<p align="center">
<img src="https://github.com/Wen-ChuangChou/Predict-Bike-Traffic/blob/main/doc/fig/prediction.png?raw=true" alt="prediction" width="700"/>
</p>

This project won **second place** in the data science challenge at BTW 2023. More details are available on [GitHub](https://wen-chuangchou.github.io/Predict-Bike-Traffic/).<br><br>  

---

## Speaker Identification  
I developed a speaker identification system using **Transformer and Conformer encoders**, improving accuracy from **53.94% to 91.8%** on a validation dataset of 56,666 voice recordings. More details are available on [GitHub](https://wen-chuangchou.github.io/Speaker-identification/).<br><br>  

---

## Anime Face Generator  
Using a dataset of approximately 71,000 anime face images, I trained a **diffusion probabilistic model** to generate anime-style portraits. The generative network improved significantly over training iterations, as shown in the images below:  

After 1,000 iterations (left) vs. 20,000 iterations (right):
<p align="center">
<img src="https://github.com/Wen-ChuangChou/Anime-face-generator/blob/main/doc/fig/1000iterations.png?raw=true" alt="1000" width="220"/>
 <img src="https://github.com/Wen-ChuangChou/Anime-face-generator/blob/main/doc/fig/20000iterations.png?raw=true" alt="20000" width="220"/> 
</p>

More details are available on [GitHub](https://wen-chuangchou.github.io/Anime-face-generator/).  
