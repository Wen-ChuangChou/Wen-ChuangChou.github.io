# Selected AI Projects
## AI agent : Agentic Retrieval-Augmented-Generation (RAG) Project

My work on **Agentic RAG** significantly enhances Large Language Model (LLM) performance for complex information-seeking. This project integrates intelligent **AI agents** into the RAG pipeline, yielding remarkably more accurate, robust, and contextually rich responses than traditional RAG.  

### Brief Technique & Impact
I developed an **AI agent** (using the `smolagent` package) capable of dynamic decision-making, iterative query reformulation, and intelligent document evaluation. A key contribution is an **optimized parallel processing pipeline** for efficient FAISS-based vector database creation from technical documentation. This framework fundamentally improves LLM output grounding through advanced reasoning and self-correction.  

### Performance Highlights 
Evaluated on a technical Q&A dataset, Agentic RAG consistently demonstrated superior accuracy across various LLMs compared to both Standard RAG and standalone LLM performance:

<div align="center">
  
|       **Model**        | **Agentic RAG** | **Standard RAG** | **LLM Only** |
|:----------------------:|:---------------:|:----------------:|:------------:|
| Gemini-1.5-flash       |      91.5       |       85.4       |     35.4     |
| Gemini-2.0-flash       |      90.8       |       85.4       |     64.1     |
| Gemini-2.5-flash-preview-05-20 | 90.8       |       86.2       |     63.8     |

</div>

<p align="center"><b>All values above are accuracy scores (in %)</b></p>


More details can be found in the project repository on [GitHub](https://github.com/Wen-ChuangChou/Agentic_RAG/tree/optimize/agent).

## AI Agent: Tool-Augmented Reasoning on GAIA Benchmark

I developed a tool-augmented AI code agent using the `smolagents` framework to tackle complex, agent-evaluating questions from the GAIA benchmark.  
This system achieved a **40%** correct answer rate—substantially outperforming GPT-4, which reached **14.4%** under the same conditions.

> **Note:** This project is currently under active development to further improve accuracy and generalization.

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

## Fine-Tuning Llama 3 for Sentiment Analysis

This project fine-tunes **Llama 3.1–8B Instruct** to perform sentiment classification on short-form text, such as tweets. The model learns to identify sentiments—**positive**, **neutral**, or **negative**—using the `tweet_sentiment_extraction` subset from the **MTEB benchmark**.

### Technique & Impact

Using instruction-style prompts and a streamlined training pipeline, the model was fine-tuned to produce accurate, single-word sentiment predictions. This significantly improved performance and makes the model well-suited for real-world applications like social media monitoring and customer feedback analysis.

### Performance Highlights

On the MTEB tweet sentiment test set, the fine-tuned model achieved a notable accuracy gain:

<p align="center"><b>Accuracy on MTEB Tweet Sentiment Classification</b></p>

<div align="center">

| **Model**                 | **Accuracy (%)** |
|---------------------------|------------------|
| Llama 3.1–8B (zero-shot)  | 63.41            |
| Llama 3.1–8B (fine-tuned) | **81.49**        |

</div>

<p align="center">
  <img src="https://github.com/Wen-ChuangChou/sentiment_analysis/blob/main/pic/radarplot.png?raw=true" alt="Radar plot showing model performance" width="400"/>
</p>


## Predicting Bike Traffic  
I implemented **Graph Attention Networks** to predict bike traffic volume using social and environmental data. The models were trained separately on datasets from Dresden, Leipzig, and Hamburg. The following plot illustrates the results across the three cities:

<p align="center">
<img src="https://github.com/Wen-ChuangChou/Predict-Bike-Traffic/blob/main/doc/fig/prediction.png?raw=true" alt="prediction" width="700"/>
</p>

This project won **second place** in the data science challenge at BTW 2023. More details are available on [GitHub](https://wen-chuangchou.github.io/Predict-Bike-Traffic/).<br><br>  


## Speaker Identification  
I developed a speaker identification system using **Transformer and Conformer encoders**, improving accuracy from **53.94% to 91.8%** on a validation dataset of 56,666 voice recordings. More details are available on [GitHub](https://wen-chuangchou.github.io/Speaker-identification/).<br><br>  


## Anime Face Generator  
Using a dataset of approximately 71,000 anime face images, I trained a **diffusion probabilistic model** to generate anime-style portraits. The generative network improved significantly over training iterations, as shown in the images below:  

After 1,000 iterations (left) vs. 20,000 iterations (right):
<p align="center">
<img src="https://github.com/Wen-ChuangChou/Anime-face-generator/blob/main/doc/fig/1000iterations.png?raw=true" alt="1000" width="220"/>
 <img src="https://github.com/Wen-ChuangChou/Anime-face-generator/blob/main/doc/fig/20000iterations.png?raw=true" alt="20000" width="220"/> 
</p>

More details are available on [GitHub](https://wen-chuangchou.github.io/Anime-face-generator/).  
