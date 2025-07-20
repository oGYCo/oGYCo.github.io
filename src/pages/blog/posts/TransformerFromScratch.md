---
layout: /src/layouts/MarkdownPostLayout.astro
title: Transformer From Scratch
author: oGYCo
description: "A powerful architecture"
image:
  url: "/images/posts/transformer.png"
  alt: "Transformer architecture diagram showing attention mechanisms and neural network layers"
pubDate: 2025-07-20
tags:
  [
    "AI", "Model Architecture"
  ]
languages: ["python"]
---
## 参考资料
- [图解transformer](https://jalammar.github.io/illustrated-transformer/)
- [哈弗-代码实现](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
## 整体架构

- Transformer
    - Encoder * N
        - Self-Attention
        - Feed Forward Neural Network
    - Decoder * N
        - Self-Attention(Masked)
        - Encoder-Decoder Attention(Cross-Attention)
        - Feed Forward Neural Network

## 第一步—Tokenization & Embedding & Positional Encoding

### 分词

### 嵌入向量

### 位置编码

## 第二步—Encoder

过程：编码器会接受一个向量列表作为输入（最开始是嵌入向量和位置向量的和），然后经过注意力层和前馈层之后传递到下一个encoder，经过多个encoder最后得到了一个向量列表会用于decoder的cross-attention层

### 注意力机制

自注意力机制就是让每个单独的token学到语境信息并更新，即让每个token向量变成一个在语境信息中更加准确的向量，从而将每个token转化为该token在语境中的高维精确的表达。

例如：”`The animal didn't cross the street because it was too tired`”
比如该句子中的it指代的是the animal，但是一个简单的it的嵌入向量只能表达最基本的代词含义，现在需要让这个it“注意到”animal，从而将这两个词联系起来，即是计算注意力分数然后根据animal的注意力分数较高，会让之后由animal得到的V向量的权重较大，使得animal对it的更新效果更加显著。当然，一个token对自身的注意力权重通常都比较显著，以确保其核心身份信息不会丢失。但最高的注意力权重会动态地分配给当前上下文中最重要的token(s)，这其中可能包括它自己，也可能包括其他的token。例如对于 `it` 来说，`animal` 的上下文信息可能比 `it` 本身的（作为代词的）信息更重要。在这种情况下，模型可能会学到给 `animal` 分配比 `it` 自身更高的注意力权重。

- 自注意力机制通过一个**加权求和**过程，让每个token的向量表示融合来自句子中其他token（尤其是最相关的token）的上下文信息。在这个过程中，通过自身的注意力权重和至关重要的**残差连接（为什么要引入[残差连接](#残差连接)）**，它又能确保token不会“忘记”自己是谁，从而实现对原始信息的保留和对上下文信息的精确更新。
- **自注意力机制的核心任务，就是计算出那个需要被加到原始向量上的“更新向量”或“残差向量”，然后与原始向量相加就得到了最后更新的向量**
- 具体过程
    - 第一步：将原始向量与三个矩阵Wq、Wk、Wv分别相乘得到Q向量、K向量、V向量，即一层注意力的参数就是Wq、Wk、Wv三个矩阵
        
        ![Transformer QKV矩阵计算](/images/posts/t-1.png)
        
    - 第二步：然后得到了三个矩阵Q、K、V，再让Q矩阵和K矩阵相乘得到每个向量与其他每个向量的注意力分数，K矩阵需要转置，就得到了向量的点积矩阵（注意力分数矩阵），注意这里注意力分数还要进行除以根号下dk（向量的维度）（防止产生过大的注意力分数和梯度消失问题，让模型无法学习多样化的信息，即最后的softmax结果几乎是一个one-shot向量）和一个softmax操作（让权重和为1，且都是非负数，这样的权重可以被理解为一个**概率分布**，它告诉我们应该将100%的“注意力”如何分配给序列中的所有token）
        
        ![注意力分数计算公式](/images/posts/t-2.png)
        
        ![注意力机制详细流程](/images/posts/t-3.png)
        
        ![Softmax函数应用](/images/posts/t-4.png)
        
        - 梯度消失的原因，当产生的数值特别大的时候，变化几乎不会导致结果的变化，也就是梯度为0，模型就会认为不需要再继续训练了
            
            ![梯度消失问题示意图](/images/posts/t-5.png)
            
    - 第三步：然后再根据注意力分数对V向量进行加权求和
  
      ![注意力加权求和计算](/images/posts/t-6.png)
- 论文中所用的实际上是[多头注意力](#多头注意力机制)来计算，也就是算出来的q、k、v向量跟原本的向量不是同一个维度，例如原向量是512维的，现在算出来的q、k、v都是64维度的
- 在实际的计算过程中用到的是矩阵计算，如图所示：
  - 算Q、K、V矩阵：例如2个512维的向量，就是一个2 * 512的矩阵，然后乘以一个512 * 64的矩阵得到了两个64维的矩阵
     ![](/images/posts/self-attention-matrix-calculation.png)
  - 算注意力矩阵:Q为2 * 64的矩阵，乘以K的转置，也就是64 * 2的矩阵，也就得到一个2 * 2的注意力分数矩阵，这样算出来就是例如【1】【1】的数值就是第一个向量对第一个向量的注意力，【1】【2】就是第一个向量对第二个向量的注意力。然后除以根号下dk(key向量的维度)，然后再做softmax，显然这里的softmax的计算就是对注意力矩阵的每个行向量进行操作，即每个行向量是作为一组来操作的
    ![](/images/posts/self-attention-matrix-calculation-2.png)

### 多头注意力机制
#### 过程
所谓的多头注意力就是原本是将原向量乘以三个矩阵：Wq、Wk、Wv转（线性变换）成维度相同的向量然后就可以一次性完成一次的注意力层，然后多头注意力就是变成了将原向量线性变换为了低维度的3个向量，例如原向量是512维的，现在得到的q、k、v都是64维度的，然后就会有8组Wq、Wk、Wv矩阵完整8次转化，每一组最终都会得到一个包含64维度向量组成的矩阵n * 64，然后把8组的矩阵拼接起来就是一个包含512维度向量的矩阵n * 512，注意这样直接拼接起来并不好，因为信息相当于是割裂的，就需要再乘以一个512 * 512的矩阵相当于把8个64维度的**信息片段**的信息进行整合，重新变成一个**完整**的向量。
![](/images/posts/transformer_attention_heads_qkv.png)
#### 为什么这样做
为什么不直接将Q、K、V变换为512维度的呢？这样不是更加方便吗，分成8次的注意力计算有什么好处？最后还要再乘以一个矩阵来整合信息,还增加了参数量和计算量
- "It expands the model’s ability to focus on different positions."
  - 每个头关注的是不同的信息，然后这样就能学到更加丰富的内容
- "It gives the attention layer multiple 'representation subspaces'"

这么看的话，transformer就跟CNN很相似了，每个注意力头关注的是不同的地方，然后通过乘以一个大的变换矩阵就形成更大的特征提取与融合

### 前馈神经网络

### 残差连接

理论上来说，更深的网络一定不会比浅的网络的性能差，因为深的网络只需要让后面的层变成恒等变换层就会与浅层网络一样，但是实践上发现要让网络拟合出一个恒等变换是很困难的，而网络拟合一个0函数就很简单，即让F(x)=0，因为只需要让权重全部变成0就可以了。于是就引入了残差连接，即让网络学习的是输入与输出之间的差函数，这样，一个恒等变换网络就等于一个残差为0的网络加上残差连接即可。而注意力层就是学习的残差函数

残差连接改变了自注意力层的**学习目标**

- **没有残差连接时**：自注意力层需要学习一个完整的变换函数 `H(x)`，直接输出最终的目标向量。这很难。
- **有残差连接时**：自注意力层只需要学习**残差（Residual）** `F(x) = H(x) - x`。也就是说，它只需要学习“**需要做的改变量**”或者说“**更新量**”。

如果模型发现不需要做任何改变，它只需要让自注意力层输出一个零向量就行了，这比学习一个恒等变换（输入什么输出什么）要容易得多。
![残差网络的意义](/images/posts/t-7.png)
### 层归一化

## 第三步—Decoder

### 掩码注意力

### Cross-Attention

## 最后—Linear & Softmax

### Linear

### Softmax

## Something Special

### 损失函数

### 并行
在进行注意力计算的时候，某个token的计算会依赖于其他token，但在实际的计算过程中是进行矩阵乘法，所以仍然是并行计算，不过在前馈层是没有token间的依赖关系的，可以单个token直接进行计算，所以可以并行计算（也是矩阵乘法）