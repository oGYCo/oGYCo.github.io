---
layout: /src/layouts/MarkdownPostLayout.astro
title: Postgresql 数据库
author: oGYCo
description: ""
image:
  url: "/images/posts/126475690_p1.jpg"
  alt: ""
pubDate: 2025-10-23
tags:
  [
    "Database"
  ]
languages: ["markdown"]
---

这几天在看数据集，捣鼓了一下数据库，整理一下应该怎么用

## 服务启动

```bash
sudo service postgresql start
```

## 查看状态

```bash
sudo service postgresql status
```

## 进入数据库内部

```bash
psql "host=localhost port=5432 dbname=<db_name> user=<user_name> password=<your_password>"
```

然后我发现pycharm可以直接连接数据库然后进行可视化查看，还是挺方便的，就是需要配置一下

首先选择数据源，填入host和port、数据库、user、password就可以进行连接了，然后这时候还要设置一下Schemas（中文应该叫架构，就是把要显示的数据库的以及要显示的Schemas打勾然后就能看到具体的表的内容了，还是蛮方便的~~也就不需要DBeaver了~~，只能说还得是ide界的苹果）