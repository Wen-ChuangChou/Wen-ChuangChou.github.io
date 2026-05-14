# Generative AI & Applied Machine Learning

[← Back to Portfolio](./)

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
