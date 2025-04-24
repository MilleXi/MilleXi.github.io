---
layout: post
title: "0基础AI项目实践入门"
date:   2025-04-19
tags: [lecture]
comments: true
author: MilleXi
--- 
笔者在学校AI社团内刚举办完CV项目实践赛，so组织了一次讲座，供更多的0基础同学入门，一起跟随笔者的脚步~使用CNN和ResNet18完成一个图像二分类任务吧！
<!-- more -->

## 一、讲座总览

### 讲座目标

本讲座旨在为0基础AI实践同学提供一次完整的计算机视觉开源项目的入门体验。通过PyTorch实现一个图像二分类系统，包括：

- 构建一个简单CNN模型
- 使用预训练ResNet18进行迁移学习
- 完成数据加载、模型训练、测试评估与可视化全过程

最终目标：**使用GitHub托管项目，掌握AI项目标准开发流程**

---

## 二、什么是 GitHub？

> GitHub 是一个面向开发者的代码托管平台，它允许你将自己的代码上传到云端，并和他人协作开发项目。它是开源软件和AI项目最常用的“集体仓库”。
> 

### 核心关键词

| 名词 | 含义 |
| --- | --- |
| `Repository`（仓库） | 项目的存储空间（代码 + 文档 + 版本历史） |
| `Commit`（提交） | 将改动保存到版本历史中 |
| `Push`（推送） | 把本地代码提交上传到GitHub |
| `Pull Request`（PR） | 请求合并代码（适用于协作） |
| `Issues`（问题追踪） | 项目的反馈系统 |
| `README.md` | 项目主页说明文档 |
| `LICENSE` | 项目的开源协议 |

> 类比：你可以把 GitHub 想象成“代码的网盘+版本管理+协作平台”。
> 

---

### 什么是开源？为什么要开源？

**开源（Open Source）** 指的是开发者将软件源代码公开，允许任何人查看、学习、修改和分发这些代码。

开源不是“免费使用”那么简单，而是一套**基于许可协议的共享与协作机制**。

**开源的好处：**

- 可以学习他人的项目结构与技术实现
- 能快速构建属于自己的模型、平台
- 方便团队协作与代码版本管理
- 有机会获得贡献记录（contribution），丰富履历

---

### 常见开源许可证（License）介绍

开源 ≠ 没有版权！**必须附带许可证才是合法开源**

以下是最常用的几种：

| License | 是否可商用 | 是否允许修改 | 是否需注明原作者 | 是否需开源修改内容 |
| --- | --- | --- | --- | --- |
| **MIT**（推荐） | ✅ | ✅ | ✅ | ❌ |
| **Apache 2.0** | ✅ | ✅ | ✅ | ❌（但需声明修改） |
| **GPL v3** | ✅ | ✅ | ✅ | ✅ |
| **BSD 3-Clause** | ✅ | ✅ | ✅ | ❌ |

> 一般教学项目、个人项目推荐使用 MIT License —— 简洁、宽松、易用。
> 

---

### 什么是 README？它是用来干什么的？

`README.md` 是 GitHub 上每一个项目仓库的“首页说明文件”，当别人点进你的仓库时，最先看到的就是它。它就像是你项目的“名片”和“使用手册”。

**它主要用来告诉别人：**

| 内容 | 举例 |
| --- | --- |
| 🔍 **项目是什么** | “这是一个使用PyTorch实现图像分类的入门项目” |
| 🛠️ **怎么安装** | “使用 pip install -r requirements.txt 安装依赖” |
| 🚀 **怎么运行** | “运行 python train.py 启动训练” |
| 📂 **项目结构** | “models/ 放模型代码，dataset/ 是数据集处理” |
| 🧠 **用到了哪些技术** | “使用了 ResNet18、迁移学习、数据增强等” |
| 📊 **最终效果 / Demo 截图** | “模型在测试集上达到92%的准确率” |
| 📄 **参考资料** | “参考论文、开源库链接” |
| 👩‍💻 **开发者信息** | “作者：YourName，欢迎PR或提Issue” |

### 什么是 .gitignore？

> .gitignore 是一个 告诉 Git 哪些文件或文件夹不应该被加入版本控制（即不被提交到仓库） 的配置文件。
> 

换句话说：

> ✅ Git 会忽略 .gitignore 中列出的内容，不会追踪它们的变更，也不会出现在 git add . 之后的提交列表里。
> 

为什么需要 `.gitignore`？

因为很多文件**是临时生成、敏感或者机器特定的**，不适合提交到 Git 仓库，比如：

| 类型 | 示例 |
| --- | --- |
| Python 缓存文件 | `__pycache__/`, `*.pyc` |
| 虚拟环境 | `venv/`, `.env/` |
| 模型文件 | `*.pth`, `*.pt`, `checkpoints/` |
| 日志与输出 | `*.log`, `*.csv`, `*.png` |
| Jupyter 自动检查点 | `.ipynb_checkpoints/` |
| 本地配置文件 | `.vscode/`, `.idea/` |
| 数据文件 | `data/`, `*.zip`, `*.h5` |

`.gitignore` 是 Git 用来**排除那些你不想提交的文件**的文件，它让你的代码仓库更干净、更专业、更安全。

---

### 项目文件结构详解

```
cv-demo/
├── data/                   # 数据集
├── dataset/                # 数据集加载模块
│   └── dataset.py          # 包含自定义 Dataset 类
├── models/                 # 模型定义模块
│   ├── cnn.py              # 简单 CNN 模型结构
│   └── resnet.py           # 加载与修改 ResNet18 模型
├── configs/
│   └── config.yaml         # 项目配置文件（模型、优化器、学习率等参数）
├── utils/
│   ├── visualizer.py       # 绘制 loss/acc 曲线或混淆矩阵
│   └── helpers.py          # 如early stop、日志记录等
├── train.py                # 训练主流程，封装训练逻辑
├── evaluate.py             # 评估模型性能并打印准确率
├── main.py                 # 可选入口（封装train和eval）
├── requirements.txt        # pip依赖文件，推荐使用 pipreqs 自动生成
├── README.md               # 项目说明文件，规范写法
└── .gitignore              # 忽略 __pycache__/、.DS_Store、.ipynb_checkpoints 等
```

---

## 三、dog vs. cat 二分类实例

### `dataset.py` - 数据加载模块

```python
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CatDogDataset(Dataset):
    """
    自定义的猫狗二分类数据集，继承自 PyTorch 的 Dataset 类。
    支持训练与测试模式，自动读取图像路径并生成标签。

    Args:
        root_dir (str): 数据集根目录，比如 'data/train' 或 'data/test'
        transform (callable, optional): 数据增强与预处理操作
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir              # 存储 train 或 test 路径
        self.transform = transform            # 图像预处理 transform

        # label 映射：猫为 0，狗为 1
        self.label_map = {'cat': 0, 'dog': 1}

        self.image_paths = []                 # 保存所有图像的路径
        self.labels = []                      # 保存对应标签

        # 遍历每个类别文件夹
        for label_name in ['cat', 'dog']:
            class_dir = os.path.join(root_dir, label_name)
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(self.label_map[label_name])

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        根据索引 idx 返回一个样本（图像, 标签）
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 打开图像
        image = Image.open(img_path).convert('RGB')  # 保证统一为 RGB 格式

        # 应用预处理/增强
        if self.transform:
            image = self.transform(image)

        return image, label
```

其中三个函数是 PyTorch 中构建自定义数据集 `Dataset` 类时必须实现的三个**核心魔法方法**。你可以把它们看作是“告诉 PyTorch 怎么管理、访问你自定义的数据”。

1. `__init__(self, ...)` —— 初始化方法（装载数据的入口）
    
    **作用：**
    
    - 设置数据集的“全局变量”（如图像路径、标签、transform）
    - 在这里做数据路径收集、标签编码等准备工作
    
    **注意：**
    
    - `self.image_paths` 存路径
    - `self.labels` 存数字标签
    - `transform` 是关键模块，用来控制预处理

2.  `__len__(self)` —— 返回数据集大小（长度）

**作用：**

- 告诉 `DataLoader` 总共有多少个样本

**注意：**

- PyTorch 训练过程中会 `for i in range(len(dataset))` 来索引数据
- 写法永远是：`return len(self.image_paths)`
1.  `__getitem__(self, idx)` —— 根据索引返回第 idx 个样本
    
    **作用：**
    
    - 返回一对 `(image_tensor, label)`，也就是模型训练的“一个样本”
    - **idx** 是下标，由 DataLoader 自动传入
    
    **注意：**
    
    - 要将图像读取、转换（transform）在这里完成
    - 每次调用 `__getitem__`，等于“抽出一个样本交给模型训练”
    - 最终返回的是模型需要的 `(X, y)`：图像张量 + 数字标签

**总结口诀**

```
__init__：准备数据表（收集路径和标签）
__len__ ：告诉总共有几条数据
__getitem__：每次按编号取出一条（返回图像 + 标签）
```

---

### `model.py` - 自定义CNN模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    一个用于图像二分类的简单卷积神经网络（CNN）模型。
    结构：Conv + ReLU + Pooling x3 + FC x2
    输入图像默认尺寸为 3x224x224
    """

    def __init__(self, num_classes=2):
        """
        初始化模型结构

        Args:
            num_classes (int): 输出类别数，默认是2（cat, dog）
        """
        super(SimpleCNN, self).__init__()

        # 卷积层1：输入通道3，输出通道16，核3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # 卷积层2
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 卷积层3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # 最大池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层1：64个通道，图像缩小3次 => 224 -> 112 -> 56 -> 28 -> flatten: 64 * 28 * 28
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        定义前向传播流程

        Args:
            x (Tensor): 输入图像张量，形状为 (N, 3, 224, 224)

        Returns:
            Tensor: 输出分类 logits，形状为 (N, num_classes)
        """
        x = self.pool(F.relu(self.conv1(x)))  # -> (N, 16, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # -> (N, 32, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # -> (N, 64, 28, 28)

        x = x.view(x.size(0), -1)  # 展平为向量

        x = F.relu(self.fc1(x))    # -> (N, 128)
        x = self.fc2(x)            # -> (N, num_classes)

        return x
```

**`torch.nn` 是 PyTorch 提供的一个神经网络构建模块库，包含了你搭建神经网络所需的所有“积木块”** —— 比如：`Linear`（全连接层）、`Conv2d`（卷积层）、`ReLU`（激活函数）、`Loss函数`、`BatchNorm`、`Dropout`，还有最关键的 `nn.Module` 类！（所有模型的基类，你要写自定义模型就继承它）

**模型结构总览**

```
Input: (3, 224, 224)

1. Conv2d(3, 16, kernel_size=3, padding=1) → ReLU → MaxPool2d(2, 2)
→ Output shape: (16, 112, 112)

2. Conv2d(16, 32, kernel_size=3, padding=1) → ReLU → MaxPool2d(2, 2)
→ Output shape: (32, 56, 56)

3. Conv2d(32, 64, kernel_size=3, padding=1) → ReLU → MaxPool2d(2, 2)
→ Output shape: (64, 28, 28)

Flatten → FC(64*28*28, 128) → ReLU → FC(128, 2)
```

**各层详细解释**

1. 卷积层 `Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)`
    - **为什么 in_channels=3**：图像是 RGB 三通道的。
    - **为什么 out_channels=16**：输出 16 个特征图（feature maps），让网络能学到比原图更抽象的特征。第一层通常不设太多通道，计算量会太大。
    - **为什么 kernel_size=3**：3x3 是现代 CNN 的经典卷积核尺寸，能较好地平衡局部感受野和计算效率。
    - **为什么 padding=1**：为保证输出尺寸不变（224x224），使用 padding=1（因为3x3卷积会缩小1圈）。
2. 激活函数 `ReLU`
    - **为什么用 ReLU 而不是 Sigmoid/Tanh？**
        - 计算简单（只保留正数）；
        - 避免梯度消失问题；
        - 实践证明效果非常好，是CNN默认激活函数。
3. 池化层 `MaxPool2d(kernel_size=2, stride=2)`
    - **功能**：将特征图尺寸减半，同时保留局部最大值，提高模型对平移、噪声的鲁棒性。
    - **为什么尺寸减半？**
        - 降低计算量；
        - 提高感受野；
        - 防止过拟合。
4. 第二、三层卷积（`Conv2d(16→32→64)`）
    - **为什么不断加通道数？**
        - 图像越深，特征越抽象，需要更多的通道来表示丰富的模式；
        - 一般采用「低通道 → 高通道」的设计策略。
    - **为什么都保持 kernel=3, padding=1？**
        - 保持一致性，避免图像太快缩小；
        - 能捕捉更丰富的空间细节。
    
    | 层次 | 操作 | 输出大小计算 | 输出尺寸 |
    | --- | --- | --- | --- |
    | Conv1 + Pool | Conv2d(3→16, kernel=3, padding=1) + MaxPool2d(2,2) | `224 → 224 → 112` | (16, 112, 112) |
    | Conv2 + Pool | Conv2d(16→32, kernel=3, padding=1) + MaxPool2d(2,2) | `112 → 112 → 56` | (32, 56, 56) |
    | Conv3 + Pool | Conv2d(32→64, kernel=3, padding=1) + MaxPool2d(2,2) | `56 → 56 → 28` | (64, 28, 28) |
    
    重点：
    
    - `Conv2d` 使用了 `padding=1`、`kernel_size=3`，**不会改变特征图尺寸**；
    - `MaxPool2d(kernel_size=2, stride=2)` 会将尺寸 **减半**。
5. 展平 `x.view(x.size(0), -1)`
    - 把 `(batch_size, 64, 28, 28)` 的 4D 张量变为二维 `(batch_size, 64*28*28)`，以供全连接层处理。
6. 全连接层 `Linear(64*28*28, 128)`
    - **为什么128？**
        - 起到“压缩特征表示”的作用，同时保留较强的表达能力；
        - 不设太高，防止过拟合；
        - 可根据训练效果调节。

7.输出层 `Linear(128, 2)`

- **输出为2个神经元**，分别对应二分类的两个类别（`cat=0`, `dog=1`）；
- **输出通常是 raw logits**，你可以配合 `nn.CrossEntropyLoss()` 一起使用，它会自动处理 softmax。

**总结设计原则**

| 模块 | 原因/作用 |
| --- | --- |
| Conv + ReLU + Pool | 提取特征，降低分辨率 |
| 增加通道数 | 捕捉更深层次的模式 |
| ReLU 激活 | 计算高效，防梯度消失 |
| 池化操作 | 减少计算、鲁棒性增强 |
| FC 层 | 汇总特征，输出结果 |
| 输出维度=2 | 对应二分类问题 |

---

### `train.py`  - 训练流程设计

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CatDogDataset
from model import SimpleCNN
import torchvision.transforms as transforms
from tqdm import tqdm  

# 使用 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用设备: {device}")

# 参数配置
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001
IMG_SIZE = 224
SAVE_PATH = 'best_model.pth'

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),  
    # 数据增强，在训练过程中，以 50% 的概率 将图像 水平翻转（即左右对调）。
    # 增加样本多样性，让模型见到更多“不同角度”的数据；提高泛化能力，减少过拟合，学到“猫是什么”而不是“猫朝哪边”；模拟现实场景	实际拍摄中物体方向可能左右都有，训练集中未必全有覆盖
    # transforms.RandomHorizontalFlip(p=0.7)70% 概率翻转。
    # 在猫狗分类、人脸检测中可以用，但是含有文字，或者方位信息重要的（道路场景）不能用。
    # 还可以transforms.RandomRotation(10)旋转增强
    transforms.ToTensor(), # 把图像从 [0, 255] 映射到 [0.0, 1.0]
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) 
    # 把每个像素值归一化到一个相对“标准”的分布范围，便于神经网络更快、更稳定地训练。
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 加载数据
train_dataset = CatDogDataset(root_dir='data/train', transform=train_transform)
test_dataset = CatDogDataset(root_dir='data/test', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# shuffle 防止模型记住顺序（比如训练集如果前半全是猫，后半全是狗，模型可能学到“顺序”，而不是图像特征），提高泛化能力（打乱样本分布后，能更好地逼近整体数据的真实分布），提升收敛稳定性（每轮训练样本顺序都不一样，能避免局部最小值或震荡问题）
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# 测试集是用来“评估模型效果”的，必须稳定、可重复。如果你在测试时也打乱数据顺序，会导致：测试结果不稳定（每次测试顺序变了）；模型对样本的预测顺序也变了，难以做准确率等指标对比；可视化结果无法复现

# 构建模型
model = SimpleCNN(num_classes=2).to(device)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
# 优化器（Optimizer）决定了模型如何更新参数，让损失函数变小，从而让模型更聪明。优化器就是根据损失函数的梯度信息，去更新模型参数（如权重和偏置） 的算法。
# Adam自适应学习率的优化器，收敛快、效果好，最常用

# 训练函数
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training', leave=False) # 进度条设置
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() # 清空之前残留的梯度值，防止梯度再次累加。
        outputs = model(images) # 正向传播
        loss = criterion(outputs, labels) # 计算损失
        loss.backward() # 计算当前梯度（反向传播）
        # 在 PyTorch 中，每次 .backward() 计算梯度时，默认是将当前梯度加到已有梯度上，而不是替换。
        optimizer.step() # 用梯度更新参数

        # 统计
        running_loss += loss.item() * images.size(0) # 当前 epoch 累计损失总和
        # loss.item()：将当前 batch 的 loss 从 Tensor 转成 Python float（否则不能累加）。images.size(0)：当前 batch 中图像的数量（也就是 batch_size）。为什么乘以 batch size？因为 loss 是对当前 batch 的“平均损失”，我们要恢复成“总损失”（总和），方便最后平均
        _, predicted = torch.max(outputs, 1)
        # 取每张图预测结果中概率最大的类别作为最终预测。outputs 是模型输出的 logits（形状：[batch_size, num_classes]）。返回 (最大值, 索引)，我们只关心索引，即类别编号。predicted 是形状为 [batch_size] 的预测标签列表
        
        correct += (predicted == labels).sum().item()
        # 统计当前 batch 中预测正确的样本个数。(predicted == labels) 得到一个布尔张量，比如 [True, False, True]。.sum() 对布尔张量求和，相当于统计正确预测个数。.item() 把张量转成 Python 数字，便于加到 correct 变量里
        
        total += labels.size(0) # 统计这个 batch 的样本数，加到总数中

        # 更新进度条
        pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# 验证函数
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Validating', leave=False)
    with torch.no_grad(): # 在不需要反向传播的阶段（如验证、测试、推理），关闭 PyTorch 的自动求导机制，节省内存和计算。
    # 在训练过程中：PyTorch 会自动构建计算图（computation graph）并保存所有中间变量（用来 .backward() 反向传播）
    # 在验证或测试时：我们只是前向传播（forward）看模型效果，不需要计算梯度，也不会调用 .backward()。
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# 训练循环
best_val_acc = 0.0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(model, test_loader, criterion)

    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"保存新模型：{SAVE_PATH}，验证准确率提升为 {val_acc:.4f}")

print("训练完成！最佳验证准确率为：", best_val_acc)
```

- 预处理中归一化操作回对图像的每个通道（R/G/B），按下面公式进行变换：
    
    $$
    \text{output} = \frac{\text{input} - \text{mean}}{\text{std}}
    $$
    
    对于 `[0.5, 0.5, 0.5]` 来说就是：
    
    $$
    \text{output} = \frac{\text{input} - 0.5}{0.5}
    $$
    
    也就是将像素值从 $0, 1$ 映射到 $-1, 1$ 区间。
    

---

### [`evaluate.py`](http://evaluate.py) - 评估与可视化

- 加载训练好的模型
- 在验证集跑一轮，输出accuracy和混淆矩阵

```python
import os
import torch
import random
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
from dataset import CatDogDataset
from model import SimpleCNN
import torchvision.transforms as transforms
from sklearn.metrics import classification_report

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 参数
BATCH_SIZE = 32
IMG_SIZE = 224
MODEL_PATH = 'best_model.pth'
NUM_VIS = 8  # 可视化图像数

# 类别标签映射
idx_to_class = {0: 'cat', 1: 'dog'}

# 测试集预处理
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 加载测试集
test_dataset = CatDogDataset(root_dir='data/test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 加载模型
model = SimpleCNN(num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 评估准确率
correct = 0
total = 0
# classification report
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = correct / total
print(f"Test Accuracy: {acc * 100:.2f}%")

# 打印分类报告
print("\n Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['cat', 'dog']))

# 可视化部分预测结果
# 反归一化函数（将[-1, 1]映射回[0, 1]）
def denormalize(tensor):
    return tensor * 0.5 + 0.5

# 随机挑选样本
sample_indices = random.sample(range(len(test_dataset)), NUM_VIS)

plt.figure(figsize=(16, 8))
for i, idx in enumerate(sample_indices):
    image, label = test_dataset[idx]
    input_tensor = image.unsqueeze(0).to(device) # 模型输入 shape = [batch_size, 3, 224, 224]，所以在维度0的位置增加一个维度，image.shape = [3, 224, 224] 变成 [1, 3, 224, 224]

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)

    # 转换为可显示格式
    image_np = denormalize(image).permute(1, 2, 0).cpu().numpy()
    # .permute(1, 2, 0)，PyTorch 的图像张量默认顺序是：(C, H, W) = (通道, 高, 宽)。而 Matplotlib/NumPy 的图像格式是：(H, W, C)，所以 .permute(1, 2, 0) 就是交换维度。
    # matplotlib 没法直接访问，需要先转到 CPU

    plt.subplot(2, NUM_VIS // 2, i + 1)
    plt.imshow(image_np)
    plt.axis('off')
    plt.title(f"Pred: {idx_to_class[pred.item()]}\nTrue: {idx_to_class[label]}", fontsize=12)

plt.tight_layout()
plt.savefig("prediction_visualization.png")
plt.show()
```

---

## 四、GitHub与开源规范教学

### Git基础操作

- `git init`、`git add .`、`git commit -m "msg"`
- `git remote add origin ...`、`git push`

---

## 五、ResNet18与迁移学习知识点

### ResNet简述

- **ResNet（残差网络）** 是一种深度神经网络结构，通过引入 **残差连接（shortcut connection）** 解决深层网络训练中的梯度消失和性能退化问题。
- 传统深层网络随着层数增加，可能会导致模型效果下降（退化），ResNet 通过让网络学习残差（即输出与输入的差值），实现了“跳跃连接”，使得网络更容易优化。
- 最基础的版本是 **ResNet18**，包含 18 层权重层，结构简单、计算量较小，适合入门级任务或小型数据集实验。尽管层数不高，但由于残差机制的引入，ResNet18 在许多任务上依然具有强大的表现力。

### 什么是预训练？

- 预训练模型是指：**模型在大规模数据集（如 ImageNet）上已经训练好**，其提取的图像特征具有很强的通用性（如边缘、纹理、轮廓等底层特征）。
- 在新任务上（比如猫狗分类），我们可以直接 **加载预训练权重**，然后：
    - **冻结卷积层，仅训练分类头**，适合小数据集；
    - 或者进行 **微调（fine-tuning）**，让模型在新任务上进一步优化。
- 预训练可以帮助模型快速收敛，并在数据量较少的情况下获得更好的性能，是现代深度学习中非常常见也非常推荐的做法。

### 微调流程

- 使用 `torchvision.models.resnet18(pretrained=True)` 加载 **ImageNet 预训练模型**；
- **替换最后的分类层**（原来是1000类 → 现在是2类）；
- **冻结前面的卷积层**，只训练最后的全连接层（这是典型的微调方式）；
- 支持 GPU 训练，可用于你的训练流程中。

```python
import torch
import torch.nn as nn
import torchvision.models as models

class FineTunedResNet18(nn.Module):
    """
    使用ImageNet预训练的ResNet18模型进行微调
    - 冻结所有卷积层
    - 替换最后的分类层为2分类
    """
    def __init__(self, num_classes=2, freeze_features=True):
        super(FineTunedResNet18, self).__init__()

        # 加载预训练的 ResNet18
        self.model = models.resnet18(pretrained=True)

        # 是否冻结特征提取部分
        if freeze_features:
            for param in self.model.parameters(): #返回模型中所有的可学习参数（通常是权重 weight 和偏置 bias）。也就是：ResNet18 的卷积层、BN层、全连接层等的参数列表。
                param.requires_grad = False

        # 替换最后的全连接层
        in_features = self.model.fc.in_features
        # 获取原始 fc 层的输入维度（通常为 512）；这个值不变，我们用它来构造一个新的输出层。
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
```

在 `train.py` 与 `evaluate.py` 中替换模型部分：

```python
# from model import SimpleCNN
from model import FineTunedResNet18
# 初始化微调模型（默认冻结特征层）
model = FineTunedResNet18(num_classes=2, freeze_features=True).to(device)  
```

---

## 六、完成README.md

# 猫狗图像二分类项目

本项目实现了基于 PyTorch 的猫狗图像二分类系统，包含从数据预处理、模型训练、评估可视化等完整流程。支持自定义 CNN 与微调的 ResNet18 两种模型结构，适合计算机视觉初学者、深度学习训练实验与迁移学习实践。

---

## 🌟 项目特点

- 支持两种模型结构：简单自定义 CNN 和 Fine-tuned ResNet18；
- 采用标准数据结构 `data_split/train`、`data_split/test`，自动划分训练测试集；
- 集成数据增强、训练可视化、准确率评估、`sklearn` 分类报告；
- 代码模块化清晰，适合教学、课程设计或作为深度学习入门项目模板。

---

## 📁 项目结构

```
project/
├── dataset.py               # 自定义 Dataset 类，支持增强与划分
├── model.py                 # 包含 SimpleCNN 和 FineTunedResNet18 两种模型
├── train.py                 # 模型训练主脚本（含 tqdm 可视化）
├── evaluate.py              # 测试模型性能，分类报告与预测结果展示
├── data_split/              # 已划分好的训练与测试数据集（cat/dog）
│   ├── train/
│   └── test/
├── best_model.pth           # 保存训练中表现最好的模型参数
└── prediction_visualization.png  # 测试集中部分预测图像可视化
```

---

## 🔧 环境依赖

请使用 Python 3.7+ 版本，并建议使用虚拟环境（如 `venv` 或 `conda`）：

```bash
pip install torch torchvision scikit-learn matplotlib tqdm
```

---

## 📦 数据准备

请将原始猫狗数据集放入 `train/` 文件夹中，图像命名格式如下：

```
cat.0.jpg, cat.1.jpg, ..., dog.0.jpg, dog.1.jpg, ...
```

运行以下脚本完成数据选择与划分：

```bash
python split_dataset.py  # 你可能已有这部分数据划分代码
```

划分后目录结构如下：

```
data/
├── train/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
```

---

## 🚀 模型训练

默认使用 `SimpleCNN`：

```bash
python train.py
```

若要切换为 ResNet18 微调训练，请在 `train.py` 中替换模型导入与初始化：

```python
from model import FineTunedResNet18
model = FineTunedResNet18(num_classes=2, freeze_features=True).to(device)
```

---

## 📊 模型评估

运行模型评估与分类报告生成：

```bash
python evaluate.p
```

控制台将输出如下信息：

```
Test Accuracy: 92.38%
Classification Report:
           precision    recall  f1-score   support
      cat       0.91      0.94      0.92       100
      dog       0.93      0.90      0.91       100
```

并生成 `prediction_visualization.png` 文件，展示预测效果。

---

## 📌 模型切换说明

| 模型类型 | 模块类名 | 特点 |
| --- | --- | --- |
| 自定义 CNN | `SimpleCNN` | 结构简单，适合初学者 |
| 预训练 ResNet18 | `FineTunedResNet18` | 利用 ImageNet 权重，精度更高，训练更快 |

---

## 📚 推荐学习路径

- 理解卷积神经网络结构；
- 学习 PyTorch 中 Dataset / DataLoader / nn.Module；
- 掌握迁移学习与微调策略；
- 熟悉模型训练流程与评估指标（Accuracy、Precision、Recall、F1）；
- 进阶可视化与 TensorBoard、混淆矩阵绘制等技巧。

---

## 🤝 贡献方式

欢迎你提交 Issue 或 Pull Request 进行改进建议、模型扩展或错误修复。

---

## 📄 许可证

本项目遵循 MIT 开源协议，详见 LICENSE 文件。

---

## 七、扩展建议

- 添加 tensorboard 可视化训练日志
- 加入 EarlyStopping 提前停止
- 引入混淆矩阵等辅助分析
- 制作 Gradio Demo 展示模型预测

留给大家自己尝试啦！