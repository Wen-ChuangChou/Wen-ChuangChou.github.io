# LLM Distillation & Fine-Tuning

[← Back to Portfolio](./)

---

## Distilling DeepSeek R1 for Enhanced LLM Performance

This project showcases a successful methodology for significantly enhancing large language model performance through advanced **post-training** in a distributed HPC environment.

### Brief Technique & Impact

This work focused on post-training weaker LLMs by fine-tuning the Qwen2.5 model using high-quality data distilled from DeepSeek R1. Employing Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) techniques across **8 H100 GPUs** distributed over **2 HPC nodes**, a key aspect of this project involved learning, optimizing, and simplifying the deployment of the training workflow for common HPC setups.

### Performance Highlights

The rigorous post-training process yielded substantial gains, boosting the Qwen2.5-Math-7B-Instruct model's `pass@1` accuracy on the AIME 2024 benchmark from 13.3% to a remarkable **56.7%**, and on GPQA Diamond from 28.3% to **54.5%**. This demonstrates the effectiveness of the distilled data approach in bringing weaker LLMs closer to DeepSeek R1's performance.

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

<p align="center"><b>Accuracy on MTEB Tweet Sentiment Classification</b>

| **Model**                 | **Accuracy (%)** |
|:-------------------------:|:----------------:|
| Llama 3.1–8B (zero-shot)  |      63.41        |
| Llama 3.1–8B (fine-tuned) |    **81.49**      |
</p>

<p align="center">
  <img src="https://github.com/Wen-ChuangChou/sentiment_analysis/blob/main/pic/radarplot.png?raw=true" alt="Radar plot showing model performance" width="400"/>
</p>

More details can be found in the project repository on [GitHub](https://wen-chuangchou.github.io/Sentiment-Analysis/).
