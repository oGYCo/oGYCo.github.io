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
    "AI", "Model Architecture", "nanochat"
  ]
languages: ["python"]
---

从这里开始就是正式的代码学习了，首先开始的是[gpt.py](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py)，也就是模型架构，karpathy在架构中添加了许多现代transformer的优化和改进，从而能够让训练的成本相对于gpt2大大降低，其中的改进和变化也能让我们看到从2019年到如今的科研界的种种进步(~~同时也增加了学习的内容和难度~~)

## 目录

- [目录](#目录)
- [背景知识](#背景知识)
  - [关于python](#关于python)
  - [关于torch](#关于torch)
  - [Attention](#attention)
  - [KV缓存与因果注意力机制](#kv缓存与因果注意力机制)
  - [MHA、MQA、GQA](#mhamqagqa)
  - [MLP(FFN)](#mlpffn)
  - [Value Embedding和门控](#value-embedding和门控)
  - [旋转位置编码（RoPE）](#旋转位置编码rope)
  - [RMS Norm（均方根归一化）](#rms-norm均方根归一化)
  - [ReLU^2激活函数](#relu2激活函数)
  - [滑动窗口注意力](#滑动窗口注意力)
- [待补充](#待补充)

---

## 背景知识
当你作为一个小白直接来看这个代码，当然会遇到各种各样的问题，而其中最大的问题就是学习当前知识的前置知识不够，我自己也感觉这是我学习过程中的一个非常痛苦的点。因为前置知识的不足，我们看到一个东西的时候会遇到大量的看起显然但是自己却非常无知的内容，这个时候就非常令人沮丧，进而导致学习动力的丧失，然后学习资料就这么开始在收藏夹吃灰了。。。当然了，当我们掌握了所有的前置知识的时候也就意味着我们学习的过程也就完成的差不太多了，剩下的部分我们只需要将各个部分进行相互联系即可。

所以，在正式的学习代码之前，我们需要大量的恶补前置知识

### 关于python
代码中用到的库：
- `dataclasses`：`dataclasses.dataclass`用于创建数据类，简化配置类的定义，`@dataclass`装饰器让这个类自动生成`__init__`等方法，从而让`GPTConfig`这个类存储GPT模型的所有配置参数（简单理解就是能够更简单的来定义一个用来管理数据的类）
- `torch`: PyTorch深度学习框架的核心库，`torch.nn`就是其中的神经网络模块

### 关于torch
PyTorch大家肯定都知道，这里就来讲解一下有关`torch.nn`和`torch.nn.functional`的部分

torch.nn就是神经网络的层的类，一个具体的深度学习神经网络模型有很多层，每一层都会有一个状态（维护层中存储的模型参数）。我们需要通过这个类来进行实例化，例如`nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)`就是创建一个线性层的实例，输入的纬度是`config.n_embd`，输出的纬度是`4 * config.n_embd`，也就是对输入进行4倍的升纬，然后没有添加bias，即经过这个层就是输入一个`n * 1`向量之后对这个向量左乘以一个`4n * n`的权重矩阵，图中箭头上面的权重也就是矩阵当中的一个参数，即`y = W * x`（注意这个图有一点不太准确的地方是中间不需要画那个蓝色的W块，箭头本身其实就是权重矩阵W，很多论文中也会用箭头的颜色深浅来表示参数的大小不同），如果是有bias的话就是再加上一个b向量，这个也是可学习的参数。
![alt text](linear.png)
线性层其实就是整个神经网络的基础，乘以一个W矩阵就是对输入向量做一次线性变化，如果加上偏置的话也就是给线性变换之后的向量加上一个偏移向量

而torch.nn.functional就是一系列的函数工具，用来进行数学计算的，通过向其输入内容我们可以得到结果

>关于注意力机制、多头注意力机制、前馈神经网络、残差连接的更加具体的理解可以看我的这篇[博客](https://ogyco.github.io/blog/posts/TransformerFromScratch/#%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6)，感觉还是讲解的比较详细和清楚

>一个小点：PyTorch中的`view()`方法和`reshape()`方法的区别 \
`view()`方法不改动内存中存储的tensor数据，而只是修改tensor的meta data，这样的速度是最快的，同时`view()`方法会要求tensor数据在内存中是连续的（**注意连续的定义是在最内层维度移动1步，内存地址也只移动1步**），比如如果对tensor进行了`transpose`操作的话`view()`方法就会报错。所以经常会看到先调用`.contiguous()`让tensor数据连续，然后再调用`view()`方法。\
`reshape()`方法则是一个更鲁棒的版本，如果tensor连续，则等同于`view`方法，如果不连续的话，会先复制一份数据让其连续再改变形状。
### Attention
简单理解attention机制就是让某个token能够关注到来自序列中其他token的信息然后更新自己，起到了一个融合信息的作用。那么其计算公式我们就可以想到，对于序列中的某个token来说，它对于序列中的每一个其他的token所需要的关注程度是不同的，那么我们就需要为每一个token计算出来一个注意力分数，然后让序列中所有token对应的注意力分数的和为1，那么每个token的注意力分数就是其注意力权重，我们再做来自每个token信息的加权和就可以得到当前token需要关注的来自整个序列的信息

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

![alt text](attention.png)
多头注意力：（~~图有点小问题但是问题不大~~）
![alt text](MHA.png)

### KV缓存与因果注意力机制
对于像GPT这样的decoder only架构，模型采用的是因果注意力，也就是当前的token只能看到它来自它前面的token的信息，现在会影响未来，但是未来不能影响过去，所以在不断的预测下一个词的时候当前的这个token在计算了QKV之后只会受到来自它前面的token的信息的更新，也就是说就算是后面token被预测出来之后当前token通过模型之后每一层计算出来的QKV也是一样的，所以我们就可以采用空间换时间的办法，直接把KV存起来，后面token通过模型的时候只需要计算这个token的QKV然后与前面token的KV缓存进行计算即可完成当前token的更新，这样就不用每一次都把整个序列一遍又一遍的喂到模型当中，每一次我只需要喂序列中的最后一个token，然后让这个token通过模型然后与前面序列的KV缓存进行交互计算即可，计算结束再把当前这个token的KV缓存起来。

那为什么不缓存Q呢？因为不需要，在推理阶段，我每一次只需要喂给模型一个token！！！也就是序列中的最后一个token，然后在做注意力计算更新的时候由这个token得到它的Q、K、V，然后用这个Q和$[缓存KV、当前token的KV]$进行交互计算就能更新这个token了。你看，整个过程并不需要来自前面的token的Q。

>注意上面说的是在推理阶段，训练阶段则不同。同时还有一个重要条件就是这种因果注意力机制，每个token只能看到它前面的token，那么在推理的时候某个token经过模型然后得到的计算的结果就是固定的，不会因为后面更新的token而受到影响，所以我们可以把KV缓存起来避免每次都进行重复的计算，也是计算中经典的空间换时间的策略。

### MHA、MQA、GQA
理解了KV缓存之后我们就可以看到为什么会出现各种注意力计算的变体了，因为KV缓存就是一堆矩阵，而且会随着序列的变长而不断增长，导致显存占用会越来越大，所以就出现了一系列的新方法来节省显存。其中一个就是对注意力的计算方式进行改变。

- MHA：多头注意力，最经典和标准的多头注意力，每一个头都有一套独立的`Wq、Wk、Wv`权重矩阵
- MQA：多查询注意力，所有的Query头都共用同一组Key和Value
- GQA：分组查询注意力，将Query头分为G组，每一组的Query头共用一套Key和Value矩阵

MHA需要存储的KV缓存最多，GQA次之，MQA需要存储的KV缓存最少，当然性能肯定是MHA最好

这么看的话，是不是跟计算机的Cache映射的三种机制很相似呢，直接映射、组相联、全相联，最后用的最多也是组相联机制

### MLP(FFN)
一般在transformer中用到的FFN就是一个MLP，也就是多层感知机，而这个感知机其实就是两个线性层加上一个激活函数的处理。

下面这张图在看了我上面提到的另外一篇博客之后再来看应该会有更好的理解
![alt text](mlp.png)

值得注意的是，Karpathy的代码中使用的激活函数操作是先经过ReLU之后再平方，为什么要这么做呢，我们先放一放

### Value Embedding和门控
通过将最开始输入的x的embedding向量经过变换以后作为一个类似于残差连接的东西加到v上，我们可以将最开始的token的初始信息传递到模型的深层中去，这样相当于建立了一条高速通道，让token在随着神经网络向着深层传递的时候依然能够获得最初的信息。不过这种做法的具体的来源我还没有找到，后续会进行补充。

然后代码中还加入了一个门控的机制，就是让模型自主学习到特定情景下需要用到多少来自初始token的embedding信息，起到一个自适应的作用。

### 旋转位置编码（RoPE）
序列问题中很重要的一个东西就是位置，模型必须要有能够区分token在序列中位置的能力，因为同样的token内容经过不同的位置排列之后会呈现出完全不同的意思。

nanochat中使用的现代大模型的通用办法，即旋转位置编码。在attention计算的过程中，我们的最终目的就是要计算出当前的token对于其他token最准确的注意力分数，即有的token需要重点关注，而有的token则并不需要怎么关注。

而Q和K是通过内积来计算出注意力分数，那么这个分数就会正比于Q、K向量之间的夹角。而同时，我们希望计算内积的时候，能够自然而然的体现token之间的相对距离。

在二维平面中，我们将向量 $\mathbf{q}$ 逆时针旋转 $m\theta$ 角度，将向量 $\mathbf{k}$ 逆时针旋转 $n\theta$ 角度。这可以通过乘以一个旋转矩阵来实现：
$$
f(\mathbf{q}, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}
$$

$$
f(\mathbf{k}, n) = \begin{pmatrix} \cos n\theta & -\sin n\theta \\ \sin n\theta & \cos n\theta \end{pmatrix} \begin{pmatrix} k_0 \\ k_1 \end{pmatrix}
$$

然后计算内积：
$$
\text{Score} = |\mathbf{q}| |\mathbf{k}| \cos(\underbrace{(\phi_q - \phi_k)}_{\text{原本的语义夹角}} + \underbrace{(m - n)\theta}_{\text{相对位置带来的偏移}})
$$
这样的话，token之间的距离关系就可以取决于相对位置了，同时还保留了原本的语义信息。

例如两个token很接近（这里是query和key接近，也就是不考虑位置的话二者算出来的注意力分数应该就比较高），那么最终的结果就会由位置信息决定，也就是相对距离的远近，而如果语义完全不相关，即使位置再近，总的分数也会因为语义分量而变得混乱。

而在实际中，向量的维度通常是很高纬度的，RoPE则采用的是两个两个一组的为其添加旋转位置编码：例如，对于维度 $d=4$ 的向量 $[x_0, x_1, x_2, x_3]$，我们将 $(x_0, x_1)$ 分为一组，$(x_2, x_3)$ 分为一组。每组在各自的子空间内进行旋转。
整体的旋转矩阵是一个分块对角矩阵：
$$
R_{\Theta, m}^d = \begin{pmatrix}
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots \\
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots \\
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots \\
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}
$$
同时为了区分不同维度的特征，每一组二维子向量使用不同的旋转频率 $\theta_i$。通常沿用 Transformer 中的设定，频率是指数递减的：
$$
\theta_i = 10000^{-2i/d}, \quad i \in [0, 1, \dots, d/2-1]
$$
即越靠近下方的纬度旋转的角度越小，也就是低频信号，用来捕捉长距离的位置关系（如果靠的很近几乎不会变化），而一开始的纬度转动的角度则很大，是高频信号，用来捕捉近距离的位置关系（靠的很近也能引起比较大的结果的变动）

回顾公式，Attention分数大约正比于（每一个q和k是二维向量也就是原向量的一部分）： 
$$
\sum_{i} \cos((m-n)\theta_i + \phi_{q_i} - \phi_{k_i})
$$ 
这种从低频到高频的叠加其实就是一种傅立叶变化的思想，最终可以组合形成那种捕捉到复杂的长短距离的波形图。

而在实际的代码实现中，不是采用乘以一个大的旋转矩阵的方式，这样太浪费显存了，会有个计算的技巧是：
假设向量 $x = [x_0, x_1, x_2, x_3, \dots]$。RoPE 的计算步骤如下：两两分组： 将 $x$ 视为复数向量，或者简单的对 $(x_{2i}, x_{2i+1})$ 操作。
应用变换：
$$
\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} x_{2i} \cos m\theta_i - x_{2i+1} \sin m\theta_i \\ x_{2i} \sin m\theta_i + x_{2i+1} \cos m\theta_i \end{pmatrix}
$$
在PyTorch中，这通常通过下面的trick实现：复制一份 $x$，将两两元素翻转并取负，得到 $\tilde{x} = [-x_1, x_0, -x_3, x_2, \dots]$。计算 $\text{RoPE}(x) = x \otimes \cos(m\Theta) + \tilde{x} \otimes \sin(m\Theta)$。（$\otimes$ 为逐元素乘法）。这样计算复杂度极低，完全是线性的 $O(d)$。

### RMS Norm（均方根归一化）
归一化的作用：通过强制将每一层的输出拉回到一个固定的尺度（例如均值为0，方差为1），它相当于在每一层都设置了一个“关卡”。无论前一层的计算结果多大或多小，经过Norm层后，都会被“重置”回一个健康的范围。对于Transformer尤为重要：在Attention机制中，由于 Softmax($QK^T$) 的存在，如果输入数值过大，Softmax 会进入“饱和区”，导致梯度几乎为 0。RMSNorm/LayerNorm 确保了输入 Softmax 的数值不会太大，保证了梯度的流动，进而避免梯度消失或者梯度爆炸的问题

假设输入向量$x$的维度为$d$，RMSNorm的计算步骤如下：
- 计算均方根:衡量向量$x$的模长（能量大小）$\epsilon$ 是为了防止分母为0的极小值
$$
RMS(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$
- 归一化:将$x$的每个元素除以RMS值，使其落在统一的尺度上。
$$
\bar{x}_i = \frac{x_i}{RMS(x)}
$$
- 通常情况下：标准的RMSNorm还会乘上一个可学习的参数向量$g$（gain/weight），nanochat中没有添加：
$$
y_i = \bar{x}_i \cdot g_i
$$

RMSNorm是LayerNorm的简化版，即没有平移减去均值的操作，hinton的论文中提到归一化的效果主要是来源于缩放而不是平移，减少了计算量的同时还能有相近的性能。

### ReLU^2激活函数
普通的ReLU在`x=0`会遇到导数不存在的情况，虽然深度学习框架能解决这个问题，但是经过平方之后，`x=0`左边的点导数为0，右边是`x^2`，`x=0右边的点的导数也是0，所以`x=0`处导数存在。

同时，右边变成了x^2，引入了更强非线性，同时增加的计算量也不大，还提升了模型的表达能力。

### 滑动窗口注意力
nanochat中采用的是滑动窗口注意力，目的是减少attention的计算量，每一个token不是与序列前面所有的token都进行attention的计算，而是只关注最近的W个token，例如第t个token只关注[t-w,w-1]的token，这样的话计算量就是不是平方增长了，而是线性增长$O(N\times W)$

但是同时呢，nanochat又不是让所有层都使用滑动窗口模式，而是采用了混合模式，即有些层是全注意力，有些层是滑动窗口注意力，防止模型丢失一开始的上下文的同时减少计算量


## 待补充
待补充内容：
- RoPE（旋转位置编码）的深度理解
- Resformer出处
- 词嵌入
- 优化器
- logits
- 损失函数
- 模型采样生成
- 线性层的直觉理解