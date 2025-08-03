---
layout: /src/layouts/MarkdownPostLayout.astro
title: Conda 环境管理
author: oGYCo
description: ""
image:
  url: ""
  alt: ""
pubDate: 2025-08-03
tags:
  [
    "tricks"
  ]
languages: ["python"]
---

对于某个项目，我们最好使用一个独立的环境来进行配置，这个环境下会包含项目所需要用到的对应版本的库
## 使用前先初始化conda
### 自动初始化
例如在Git Bash中运行：
```bash
conda init bash
```
这样它会修改你的 ~/.bashrc 文件，加入 Conda 的初始化代码，比如：
```bash
__conda_setup="$('/path/to/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
```
path就是到miniconda3/Scripts/conda.exe的路径

然后就可以手动加载配置：
```bash
source ~/.bashrc
```
### 手动初始化：
打开并编辑.bashrc文件：

```bash
nano ~/.bashrc
```

在文件末尾添加：

```bash
__conda_setup="$('/path/to/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
```
然后`ctrl+X`再`ctrl+Y`再`enter`再加载配置即可

## 常见命令

### 新建环境

```bash
conda create -n your_env_name python=3.10
```

也可以指定安装其他的包

```bash
conda create -n your_env_name python=3.10 numpy pandas
```

### 激活环境

建立环境之后要激活

```bash
conda activate your_env_name
```
### 查看当前环境已经安装的包

```bash
conda list
```

### 退出环境到base

```bash
conda deactivate
```

### 导出环境的配置（导出为 .yml 文件）

```bash
conda env export > environment.yml
```

### 从.yml文件创建环境（克隆别人的配置）

```bash
conda env create -f environment.yml
```

### 更新包

```bash
conda update numpy
```
or

```bash
conda update --all
```

### 列出所有的环境

```bash
conda env list
```

### 删除环境

```bash
conda remove -n your_env_name --all
```