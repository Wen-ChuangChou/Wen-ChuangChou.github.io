# Selected AI Projects
## AI Aigent : Agentic Retrieval-Augmented-Generation (RAG) Project

My work on **Agentic RAG** significantly enhances Large Language Model (LLM) performance for complex information-seeking. This project integrates intelligent **AI agents** into the RAG pipeline, yielding remarkably more accurate, robust, and contextually rich responses than traditional RAG.  

**Brief Technique & Impact:**  
I developed an **AI agent** (using the `smolagent` package) capable of dynamic decision-making, iterative query reformulation, and intelligent document evaluation. A key contribution is an **optimized parallel processing pipeline** for efficient FAISS-based vector database creation from technical documentation. This framework fundamentally improves LLM output grounding through advanced reasoning and self-correction.  

**Performance Highlights:**  
Evaluated on a technical Q\&A dataset, Agentic RAG consistently demonstrated superior accuracy across various LLMs compared to both Standard RAG and standalone LLM performance: 

|         Model         | Agentic RAG Accuracy | Standard RAG Accuracy | LLM Only Accuracy |
|:---------------------:|:--------------------:|:----------------------:|:------------------:|
|  Gemini-1.5-flash     |        91.5%         |         85.4%          |       35.4%        |
|  Gemini-2.0-flash     |        90.8%         |         85.4%          |       64.1%        |
| Gemini-2.5-flash-preview-05-20 |    90.8%         |         86.2%          |       63.8%        |


More details can be found in the project repository on [GitHub](https://github.com/Wen-ChuangChou/Agentic_RAG/tree/optimize/agent).


## Reproducing Post-Training Approaches from DeepSeek R1
This project implemented two post-training techniques—**Supervised Fine-Tuning (SFT)** and **Group Relative Policy Optimization (GRPO)**—to fine-tune large language models (LLMs) using **8 H100 GPUs across 2 HPC nodes**. The LLMs were fine-tuned using data distilled from DeepSeek R1, resulting in a substantial performance gain on the AIME 2024 benchmark, with accuracy improving from **10.0% to 66.7%**.

<div align="center">

<table>
  <thead>
    <tr>
      <th style="text-align:center;"><b>Model</b></th>
      <th style="text-align:center;"><b>AIME 2024</b> pass@1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">
        <b>Qwen2.5-Math-7B-Instruct</b><br>
        <sub>(Original model)</sub>
      </td>
      <td style="text-align:center;">10.0</td>
    </tr>
    <tr>
      <td style="text-align:center;">
        <b>Qwen2.5-Math-7B-Instruct (Fine-tuned)</b><br>
        <sub>(Supervised Fine-tuned on data distilled from DeepSeek R1)</sub>
      </td>
      <td style="text-align:center;">66.7</td>
    </tr>
    <tr>
      <td style="text-align:center;">
        <b>DeepSeek-R1-Distill-Qwen-7B</b><br>
        <sub>(Reportedly fine-tuned on its own distilled data)</sub>
      </td>
      <td style="text-align:center;">50.0</td>
    </tr>
  </tbody>
</table>

</div>

More details can be found in the project repository on [GitHub](https://wen-chuangchou.github.io/Open-R1/).<br><br> 

## Sentiment Analysis
My final model, **fine-tuned** for sentiment analysis using Llama 3, achieves an accuracy of 81.49%, marking an improvement of over **18%** compared to the base model. The radar plot below illustrates the enhanced performance across various metrics.

<p align="center">
  <img src="https://github.com/Wen-ChuangChou/sentiment_analysis/blob/main/pic/radarplot.png?raw=true" alt="radar plot" width="400"/>
</p>

More details can be found in the project repository on [GitHub](https://wen-chuangchou.github.io/sentiment_analysis/).<br><br>  


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
