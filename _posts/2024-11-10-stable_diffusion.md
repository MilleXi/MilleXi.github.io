---
layout: post
title: "Stable Diffusion浅析"
date:   2024-11-16
tags: [tech]
comments: true
author: MilleXi
---
浅浅速通一下Stable Diffusion
<!-- more -->

<script>
    window.MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']], }};
</script>
<script src='https://cdn.jsdelivr.net/npm/mathjax@3.0.1/es5/tex-mml-chtml.js'></script>


## **1. Stable Diffusion 介绍**

Stable Diffusion（SD）是基于 **扩散模型（Diffusion Models, DM）** 的一种文本到图像（Text-to-Image）生成模型，采用 **潜在扩散（Latent Diffusion Model, LDM）** 以提高计算效率。它可以：

- 根据**文本提示（Prompt）** 生成图像
- 进行 **图像到图像（img2img）** 变换
- 结合 **ControlNet、LoRA** 进行微调和风格控制

**Stable Diffusion vs. 传统扩散模型**

- 传统扩散模型在 **像素空间（Pixel Space）** 进行去噪，计算量大
- Stable Diffusion 在 **潜在空间（Latent Space）** 进行去噪，计算更高效

---

## **2. Stable Diffusion 的核心架构**

Stable Diffusion 由 **四个核心组件** 组成：

1. **文本编码器（Text Encoder, CLIP / T5）**
2. **潜在扩散模型（Latent Diffusion Model, LDM）**
3. **UNet 去噪网络**
4. **VAE 变分自编码器**

---

### **2.1 文本编码器（CLIP 或 DeepFloyd T5）**

Stable Diffusion 使用 **CLIP 文本编码器**（SD 1.5 / 2.1） 或 **DeepFloyd T5（SDXL）** 来处理文本输入。

**作用**

- 将输入的文本 `prompt` 变成高维向量 **文本嵌入（Text Embeddings）**
- 这些嵌入将指导扩散模型生成符合语义的图像

**CLIP 文本编码器**

- 采用 **Transformer 结构**
- 生成固定长度的 **文本向量（Text Embedding）**
- 提供 **跨模态对比学习能力**

**DeepFloyd T5**

- SDXL 采用 **更强的文本编码器**，能够解析更复杂的句子结构，提高 Prompt 理解能力

---

### **2.2 潜在扩散模型（Latent Diffusion Model, LDM）**

**扩散模型的基本原理**

扩散模型（Diffusion Model, DM）是一种 **概率生成模型**，由两部分组成：

1. **前向扩散（Forward Diffusion）**：将干净图像逐步添加噪声
2. **逆向去噪（Reverse Denoising）**：学习去除噪声，重构原始图像

**数学公式**

- **前向过程**：
    
    $$
    q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
    $$
    
    其中：
    
    - $x_t$ 是时间步  的噪声图像
    - $\beta_t$ 是噪声调节参数
    - 经过多个时间步，最终 $x_T$ 接近标准高斯分布 $\mathcal{N}(0, I)$
- **逆向去噪过程**：
    
    $$
    p(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma^2 I)
    $$
    
    - **UNet 负责学习 $\mu_\theta(x_t, t)$**，即如何从噪声恢复原图

**为什么使用潜在扩散？**

- 在像素空间扩散计算成本高
- Stable Diffusion 先用 **VAE 压缩图像** 到潜在空间，再进行扩散

---

### **2.3 UNet 生成网络**

UNet 负责 **去噪**，从噪声中恢复图像。它的结构包括：

- **编码器（Encoder）** 提取特征（ResNet + Self-Attention）
- **解码器（Decoder）** 生成图像
- **跳跃连接（Skip Connection）** 保持高分辨率信息
- **自注意力机制（Self-Attention）** 处理远程依赖

---

### **2.4 VAE 变分自编码器**

VAE 负责：

- **压缩** 原始图像到潜在空间
- **解码** 潜在表示为最终图像

Stable Diffusion 使用 **预训练 VAE**：

- 编码器 $E(x)$ 将 512×512 图像映射到 64×64 潜在空间
- 解码器 $D(z)$ 负责将潜在表示恢复为清晰的图像

---

## **3. Stable Diffusion 的推理流程**

Stable Diffusion 生成图像的完整流程如下：

1️⃣ **文本编码**

- 使用 CLIP / T5 文本编码器将 `prompt` 转换为 **文本嵌入**

2️⃣ **初始化噪声**

- 在潜在空间初始化一个 **随机高斯噪声**

3️⃣ **去噪过程**

- 采用 UNet 逐步去噪，使用 **扩散采样器**（如 Euler、DPM-Solver++）

4️⃣ **VAE 解码**

- 将去噪后的潜在表示 **解码为最终图像**

---

## **4. 扩展优化**

### **4.1 采样算法**

不同的采样器影响生成速度和质量：

- **Euler A**（速度快，适合创意生成）
- **DPM-Solver++**（高质量，最常用）
- **DDIM**（快但细节可能损失）

---

### **4.2 ControlNet**

ControlNet 允许对 **生成图像施加额外控制**：

- **Canny 边缘**（保持物体形状）
- **OpenPose**（控制人物姿势）
- **Depth Map**（深度感知）

---

### **4.3 LoRA 低秩适配**

LoRA 是一种轻量级微调方法：

- 允许用户**训练个性化模型**
- 计算成本低，可以在 **普通 GPU** 上训练

---

## **5. 代码实现**

以下是一个基本的 Stable Diffusion 推理代码（基于 `diffusers` 库）：

```python
from diffusers import StableDiffusionPipeline
import torch

# 加载预训练 Stable Diffusion 模型
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda")

# 设置 Prompt
prompt = "A futuristic cyberpunk city at night with neon lights"

# 生成图像
image = pipe(prompt).images[0]

# 显示图像
image.show()

```

**加入 ControlNet**

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet
)

# 生成带有 Canny 约束的图像
image = pipe(prompt, image=processed_canny_image).images[0]
image.show()

```

---

## **总结**

- Stable Diffusion 采用 **潜在扩散模型（LDM）**，在 **潜在空间** 进行扩散，提高计算效率
- 主要组件包括 **文本编码器（CLIP / T5）、UNet、VAE**
- 采样器、ControlNet 和 LoRA 可以 **优化和控制生成过程**
- 可用于 **文本生成、风格迁移、局部编辑**