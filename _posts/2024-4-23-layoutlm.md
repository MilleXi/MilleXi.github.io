---
layout: post
title: "论文阅读小记：LayoutLMv3"
date:   2024-05-28
tags: [tech]
comments: true
author: MilleXi
---
笔者印象非常深刻的论文之一，也是笔者在自己的专利（申请ing）中曾参考过的一篇论文，也曾尝试扒过代码，总的来说确实是非常强的智能文档AI模型，不愧是MSR出手，推推推！so，浅浅做个阅读小记...
<!-- more -->

## **题目：**
LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking

### **论文背景：**
近年来，文档AI（Document AI）领域在自监督预训练技术的推动下取得了显著进展。该领域的任务包括文档分类、表单理解、收据理解、文档视觉问答（DocVQA）等。然而，现有多模态预训练模型在文本和图像模态的处理上存在较大差异，使得跨模态对齐（Cross-modal Alignment）成为一个挑战。

传统方法通常在文本模态上使用BERT提出的掩码语言模型（MLM）进行预训练，而在图像模态上采用不同的方法，例如像DocFormer使用CNN解码器重建像素信息，而SelfDoc则采用回归掩码区域特征的方法。这些方法导致文本与图像的特征粒度不匹配，使得跨模态表示学习更加困难。

### **论文的Motivation：**
为了缓解文本和图像模态之间的预训练目标差异，并促进多模态表示学习，作者提出了LayoutLMv3。该方法的核心创新点包括：
1. **统一的文本和图像掩码机制（Unified Text and Image Masking）**：在文本和图像两种模态上均采用掩码预测任务（MLM + MIM）。
2. **词-补丁对齐目标（Word-Patch Alignment, WPA）**：提出了一种新的对齐机制，以学习文本词和对应图像补丁之间的对齐关系。
3. **移除CNN依赖**：不同于LayoutLMv2依赖CNN或Faster R-CNN提取图像特征，LayoutLMv3直接采用ViT风格的线性投影（Linear Projection）来处理图像。

## **方法论：**

### **1. 模型架构**
LayoutLMv3 采用了多层 Transformer 结构，每一层主要由**多头自注意力（Multi-head Self-Attention, MSA）**和**前馈神经网络（Feed-forward Network, FFN）**组成。输入由**文本嵌入（Text Embedding）**和**图像嵌入（Image Embedding）**组成：
- **文本嵌入**：包括**词嵌入（Word Embeddings）**和**位置嵌入（Position Embeddings）**，其中位置嵌入又分为**1D 位置嵌入（文本序列索引）**和**2D 布局位置嵌入（文本的边界框坐标）**。
- **图像嵌入**：采用**线性投影（Linear Projection）**处理图像补丁，而不依赖 CNN 进行特征提取。具体而言，将文档图像切分为 16×16 的小块（patches），然后用线性变换投影到特征空间，并加上 1D 位置嵌入。

### **2. 预训练目标**
LayoutLMv3 采用了三大预训练目标：
1. **掩码语言建模（Masked Language Modeling, MLM）**
   - 从文本序列中随机 Mask 一部分 Token，并要求模型预测其原始 Token。
   - Mask 方式参考 BART，采用跨度 Mask（Span Masking），Mask 长度从泊松分布（Poisson Distribution）抽样。

2. **掩码图像建模（Masked Image Modeling, MIM）**
   - 受 BEiT 启发，将 40% 的图像补丁 Mask 掉，模型需要预测这些 Mask 位置的离散图像 Token。
   - 采用离散 VAE 进行图像 Token 化，目标是预测被 Mask 掉的图像 Token。

3. **词-补丁对齐（Word-Patch Alignment, WPA）**
   - 由于 MLM 和 MIM 目标独立进行，无法显式学习文本与图像之间的对齐关系。
   - 提出 WPA 任务：如果一个文本 Token 对应的图像补丁未被 Mask，则其对齐标签为**Aligned**，否则为**Unaligned**。
   - 通过二分类任务（Binary Classification）来预测是否为 Aligned/Unaligned。

## **实验与结果：**

### **1. 数据集**
LayoutLMv3 在 IIT-CDIP 数据集（约 1100 万文档图像）上进行预训练，并在多个公共基准数据集上进行微调：
- **FUNSD（表单理解）**
- **CORD（收据理解）**
- **RVL-CDIP（文档分类）**
- **DocVQA（文档视觉问答）**
- **PubLayNet（文档布局分析）**

### **2. 主要实验结果**
LayoutLMv3 在所有任务上均达到了 SOTA 性能：
- **表单理解（FUNSD）**：F1 提升至 92.08（比 StructuralLM 高出 7%）。
- **收据理解（CORD）**：F1 提升至 97.46。
- **文档分类（RVL-CDIP）**：准确率提升至 95.93。
- **文档视觉问答（DocVQA）**：ANLS 提升至 83.37。
- **文档布局分析（PubLayNet）**：mAP 达到 95.1（超越 ResNet 和 DiT）。

### **3. 消融实验（Ablation Study）**
研究了不同图像嵌入方法和预训练目标的影响：
- **移除 MIM 目标后，图像分类（RVL-CDIP）和文档布局分析（PubLayNet）性能显著下降。**
- **移除 WPA 目标后，跨模态对齐能力下降，所有任务性能均有所降低。**
- **采用线性投影替代 CNN 后，参数减少 75%，但仍能保持 SOTA 结果。**

**结论与未来展望：**
1. **LayoutLMv3 采用统一的文本和图像掩码目标，有效解决了多模态预训练目标的不一致问题。**
2. **通过采用 ViT 风格的线性投影，移除了 CNN 依赖，使得模型更轻量级。**
3. **通过 WPA 任务实现了跨模态对齐，提高了文本和图像信息的融合能力。**
4. **未来工作包括扩展到更大规模的预训练模型，并探索少样本和零样本学习能力，以提升文档 AI 的实际应用价值。**

---

## **LayoutLMv3 相较于 LayoutLMv2 的改进点：**
| 特点 | LayoutLMv2 | LayoutLMv3 |
| --- | --- | --- |
| **图像特征提取** | CNN (ResNeXt-101) | 线性投影（ViT 方式） |
| **预训练目标** | MLM + MIM | MLM + MIM + WPA |
| **位置编码** | 词级 2D 位置编码 | 段级 2D 位置编码 |
| **跨模态对齐** | 遮蔽部分文本学习对齐 | WPA 明确对齐任务 |
| **计算成本** | 高（CNN） | 低（无 CNN） |

## **总结：**
LayoutLMv3 通过一体化的预训练目标和更高效的特征提取方式，成为新一代文档 AI 预训练模型，在多个任务上均取得了 SOTA 结果。

