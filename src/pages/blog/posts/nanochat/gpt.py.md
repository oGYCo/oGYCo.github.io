---
layout: /src/layouts/MarkdownPostLayout.astro
title: nanochat学习-模型架构
author: oGYCo
description: "学习笔记"
image:
  url: "/images/posts/nanochat.png"
  alt: "Andrej Karpathy's nanochat"
pubDate: 2026-02-01
tags:
  [
    "AI", "Model Architecture"
  ]
languages: ["python"]
---

从这里开始就是正式的代码学习了，首先开始的是[gpt.py](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py)，也就是模型架构，karpathy在架构中添加了许多现代transformer的优化和改进，从而能够让训练的成本相对于gpt2大大降低，其中的改进和变化也能让我们看到从2019年到如今的科研界的种种进步(~~同时也增加了学习的内容和难度~~)
---

## 目录

- [目录](#目录)
- [背景知识](#背景知识)

---

## 背景知识
当你作为一个小白直接来看这个代码，当然会遇到各种各样的问题，而其中最大的问题就是学习当前知识的前置知识不够，我自己也感觉这是我学习过程中的一个非常痛苦的点。因为前置知识的不足，我们看到一个东西的时候会遇到大量的看起显然但是自己却非常无知的内容，这个时候就非常令人沮丧，进而导致学习动力的丧失，然后学习资料就这么开始在收藏夹吃灰了。。。当然了，当我们掌握了所有的前置知识的时候也就意味着我们学习的过程也就完成的差不太多了，剩下的部分我们只需要将各个部分进行相互联系即可。

所以，在正式的学习代码之前，我们需要大量的恶补前置知识

### 关于python
代码中用到的库：
- dataclasses：dataclasses.dataclass用于创建数据类，简化配置类的定义，`@dataclass`装饰器让这个类自动生成`__init__`等方法，从而让`GPTConfig`这个类存储GPT模型的所有配置参数（简单理解就是能够更简单的来定义一个用来管理数据的类）
- torch: PyTorch深度学习框架的核心库，`torch.nn`就是其中的神经网络模块

### 关于torch
PyTorch大家肯定都知道，这里就来讲解一下有关`torch.nn`和`torch.nn.functional`的部分

torch.nn就是神经网络的层的类，一个具体的深度学习神经网络模型有很多层，每一层都会有一个状态（维护层中存储的模型参数）。我们需要通过这个类来进行实例化。

而torch.nn.functional就是一系列的函数工具，用来进行数学计算的，通过向其输入内容我们可以得到结果




## 待补充
待补充内容：
- GPTconfig的滑动窗口注意力模式
- RMS Norm归一化
- RoPE（旋转位置编码）
- Value Embedding（值嵌入）
- SelfAttention和各种注意力的变体
- 激活函数
- MLP（多层感知机）（FNN）
- 残差连接
- KV cache
- flash attention
- 词嵌入
- 优化器
- logits
- 损失函数
- 模型采样生成