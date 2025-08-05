---
layout: /src/layouts/MarkdownPostLayout.astro
title: Something about Docker
author: oGYCo
description: ""
image:
  url: "/images/posts/home-tab-build.avif"
  alt: ""
pubDate: 2025-08-03
tags:
  [
    "tricks"
  ]
languages: [""]
---
## Docker 是软件世界的“集装箱”

**打包**： 将应用程序连同它所需的所有依赖（库、配置文件、运行时环境等）一起打包。

**隔离**： 这个“集装箱”是沙箱化的，与宿主机和其他“集装箱”隔离。

**标准化**： 这个“集装箱”可以在任何支持 Docker 的机器上（开发、测试、生产）以完全相同的方式运行，彻底解决了环境一致性问题。

## Docker 的核心概念

### 镜像 (Image)

>定义： 镜像是一个只读的模板，它包含了运行一个应用所需的一切：代码、运行时、库、环境变量和配置文件 (即一道菜的菜谱，详细记录了需要哪些食材（依赖）和制作步骤)

**核心特性**：分层存储 (Layered Storage)
- FROM ubuntu:20.04 (这是基础层)
- COPY . /app (这会在上一层之上增加一个新层)
- RUN pip install -r requirements.txt (这又是一个新层)

**创建方式**： 镜像通常通过一个名为 Dockerfile 的文本文件来定义和构建。Dockerfile 包含了一系列指令，告诉 Docker 如何一步步地构建出这个镜像。比如用什么基础镜像、复制什么代码、安装哪些依赖等

一个简单的 Dockerfile 示例：
```dockerfile
# 使用官方的 Python 3.9 作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 将当前目录下的文件复制到容器的 /app 目录下
COPY . /app

# 安装 requirements.txt 中指定的依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露 5000 端口，允许外部访问
EXPOSE 5000

# 容器启动时运行的命令
CMD ["python", "app.py"]
```

#### 管理镜像的常用命令
| 操作         | 命令示例                              |
| ------------ | ------------------------------------- |
| 查看已经构建的镜像     | `docker images` / `docker image ls`   |
| 根据当前目录的dockerfile构建镜像     | `docker build -t myapp:latest .`      |
| 打标签       | `docker tag 镜像ID 新名字:标签`       |
| 删除镜像（也可以用id删除）     | `docker rmi myapp:latest`             |
| 清理无用(无标签的镜像)镜像 | `docker image prune`                  |
| 从远程仓库拉取镜像     | `docker pull ubuntu:22.04`            |
| 推送镜像到远程仓库     | `docker push myrepo/myapp:1.0`        |
| 导出镜像（导出为.tar文件）     | `docker save -o app.tar myapp:latest` |
| 导入镜像（从.tar文件导入镜像）     | `docker load -i app.tar`              |
| 查看详情     | `docker inspect myapp:latest`         |

### 容器 (Container)
>定义： 容器是镜像的运行实例（如果镜像是“类”，那么容器就是这个类的“实例”或“对象”）

**核心特性：**
- 沙箱环境： 每个容器都运行在自己独立、隔离的环境中。它拥有自己的文件系统、网络栈和进程空间，与宿主机和其他容器互不影响
- 可写层： 当一个容器从镜像启动时，Docker 会在只读的镜像层之上，添加一个可写的容器层。你在容器内做的任何修改（如创建文件、修改配置）都发生在这个可写层，不会影响到底层的镜像
- 轻量与快速： 容器直接运行在宿主机的内核上，没有自己的内核，也不需要硬件虚拟化，使得其性能开销很小

#### 管理容器的常用命令
| 操作             | 命令示例                                    | 说明                   |
| ---------------- | ------------------------------------------- | ---------------------- |
| 查看正在运行容器     | `docker ps`                                 |                        |
| 查看所有容器     | `docker ps -a`                              | 包括停止的容器         |
| 使用(nginx镜像)运行新容器       | `docker run -d --name myc -p 8080:80 nginx` | -d：后台运行，-p：端口映射（宿主机:容器）
| 启动容器         | `docker start myc`                          |                        |
| 停止容器         | `docker stop myc`                           |                        |
| 重启容器         | `docker restart myc`                        |                        |
| 删除容器         | `docker rm myc`                             | 需要停止后删除         |
| 强制删除容器     | `docker rm -f myc`                          | 即使运行中也删除       |
| 进入容器         | `docker exec -it myc /bin/bash`             | 交互式终端             |
| 查看日志         | `docker logs myc`                           |                        |
| 实时查看日志     | `docker logs -f myc`                        | 跟踪日志               |
| 查看详细信息     | `docker inspect myc`                        | JSON 格式信息          |
| 导出容器文件系统 | `docker export myc > myc.tar`               | 导出为 tar 文件        |
| 导入容器文件系统（转换为了镜像） | `docker import myc.tar`                     | 导入为镜像             |
| 拷贝文件         | `docker cp src myc:/dest`                   | 拷贝文件到容器         |

### 数据卷(Volume)
容器本身是“无状态”且“短暂”的。当一个容器被删除后，它在可写层产生的所有数据都会丢失。但我们的应用（如数据库、用户上传的文件）需要持久化存储数据

>定义： 数据卷是一个可供一个或多个容器使用的特殊目录，它绕过了容器的 Union File System，可以直接将数据写入宿主机的文件系统

**核心特性：**
- 独立生命周期： 数据卷的生命周期独立于容器。即使创建它的容器被删除了，数据卷及其中的数据依然存在
- 由 Docker 管理： Docker 在宿主机上创建一个专门的区域（通常是 /var/lib/docker/volumes/）来管理数据卷
- 容器间共享： 多个容器可以挂载同一个数据卷，从而实现数据共享

```bash
# 创建一个名为 my-data 的数据卷
docker volume create my-data

# 启动一个容器，并将 my-data 数据卷挂载到容器内的 /app/data 目录
docker run -d --name some-app -v my-data:/app/data my-image

#经过以上步骤，任何写入容器内 /app/data 目录的数据，实际上都被保存到了宿主机的 my-data 数据卷中，即使 some-app 容器被删除，数据依然安全
```

### 容器网络(Network)

现代应用通常是多服务的（例如 Web 服务 + 数据库服务 + 缓存服务）。这些服务运行在不同的容器中，它们之间需要相互通信

Docker 提供了多种网络模式，最常用的是桥接网络(Bridge Network)
- 默认桥接网络： 所有容器默认都连接到一个名为 bridge 的网络。在这个网络中，容器之间可以通过它们的 IP 地址进行通信。但这种方式不稳定，因为容器重启后 IP 可能会变
- 自定义桥接网络（推荐）： 最佳实践是为你的应用创建一个自定义的桥接网络
```bash
# 1. 创建一个自定义网络
docker network create my-app-net

# 2. 启动数据库容器，并连接到这个网络
docker run -d --name db --network my-app-net redis

# 3. 启动应用容器，也连接到这个网络
docker run -d --name web --network my-app-net -p 8080:5000 my-web-app
```
在同一个自定义网络中，容器之间可以直接通过它们的容器名（db, web）作为主机名进行通信。Docker 内置了 DNS 服务来解析这些名字。这样，你的 Web 应用代码就可以直接连接 redis://db:6379，而无需关心 db 容器的 IP 地址

## Docker Compose
>定位： 用于在**单个主机上**定义和运行多容器 Docker 应用的工具

通过一个 docker-compose.yml 的 YAML 文件来配置应用的所有服务（容器）、网络、数据卷等。然后只需一条命令 docker-compose up 即可启动整个应用

一个典型的 docker-compose.yml 文件结构如下：
```yaml
# docker-compose.yml
version: '3.8' # 推荐写上版本号

services:
  # 1. 定义 Web 应用服务
  web:
    build: ./web # 使用 web/ 目录下的 Dockerfile 来构建镜像
    container_name: my-web-app # 给容器起个固定的名字
    ports:
      - "8080:5000" # 将宿主机的 8080 端口映射到容器的 5000 端口
    volumes:
      - ./web:/app # 将本地的 web 目录挂载到容器的 /app 目录，方便开发时热更新代码
    environment:
      - FLASK_ENV=development
      - REDIS_HOST=db # 告诉应用，Redis 的主机名是 'db'
    networks:
      - app-net # 连接到 app-net 网络
    depends_on:
      - db # 确保先启动 db 服务

  # 2. 定义 Redis 服务
  db:
    image: "redis:alpine" # 直接使用 Docker Hub 上的官方 Redis 镜像
    container_name: my-redis-db # 固定的名字
    networks:
      - app-net # 也连接到 app-net 网络
    volumes:
      - redis-data:/data # 挂载一个命名数据卷来持久化 Redis 数据

# 3. 定义网络
networks:
  app-net:
    driver: bridge

# 4. 定义数据卷
volumes:
  redis-data:
    driver: local
```
### 一个服务的常用配置项

* `image: <image_name>`: 指定服务使用的镜像。可以是 Docker Hub 上的公共镜像（如 `redis:alpine`），也可以是你自己构建的镜像。
* `build: <path>`: 如果你不想使用现成的 `image`，可以用 `build` 来指定 `Dockerfile` 所在的目录。Compose 会自动帮你构建镜像。
    * `build: .` 表示使用当前目录下的 `Dockerfile`。
    * `build: context: ./dir, dockerfile: Dockerfile-alternate` 可以指定更复杂的构建上下文和 Dockerfile 名称。
* `container_name: <name>`: 指定容器的名称，而不是让 Docker 自动生成一个随机的名字。
* `ports`: 定义端口映射，相当于 `docker run -p`。
    * 格式: `["HOST_PORT:CONTAINER_PORT"]`
    * 示例: `ports: ["8080:5000"]`
* `environment`: 设置环境变量，相当于 `docker run -e`。
    * 格式: `["KEY=VALUE"]` 或一个 map。
    * 示例: `environment: [ "REDIS_HOST=db", "DEBUG=1" ]`
* `volumes`: 挂载数据卷或主机目录，相当于 `docker run -v`。
    * 挂载命名数据卷: `my-data:/app/data` (推荐用于持久化数据)
    * 挂载主机目录 (绑定挂载): `./code:/app/code` (常用于开发时同步代码)
* `networks`: 将服务连接到指定的网络。
* `depends_on`: 定义服务间的启动依赖关系。例如，`web` 服务依赖于 `db` 服务，Compose 会先启动 `db`，再启动 `web`。**注意：这只保证启动顺序，不保证 `db` 服务内部的程序已经准备好接受连接。**
* `restart`: 定义容器的重启策略。
    * `no`: 默认值，不重启。
    * `always`: 无论退出状态码是什么，总是重启。
    * `on-failure`: 只有在退出状态码非 0 时才重启。
    * `unless-stopped`: 除非手动停止，否则总是重启（推荐用于生产）。
* `command`: 覆盖容器启动时要执行的默认命令。

### 常用Docker Compose命令
- `docker compose up`: 创建并启动所有服务。加上 -d 在后台运行。如果服务已存在，会重新创建变化的。
- `docker compose down`: 停止并移除所有相关的容器、网络。
- `docker compose down -v`: 在 down 的基础上，同时移除在 volumes 中定义的命名数据卷。非常重要，用于彻底清理。
- `docker compose start`: 启动已存在的服务。
- `docker compose stop`: 停止服务，但不移除。
- `docker compose restart`: 重启服务。
- `docker compose ps`: 查看所有服务的状态（类似 docker ps）。
- `docker compose logs <service_name>`: 查看指定服务的日志。
- `docker compose logs -f web`: 实时跟踪 web 服务的日志。
- `docker compose build <service_name>`: 重新构建指定服务的镜像。如果 Dockerfile 或相关文件有变动，需要执行此命令。
- `docker compose exec <service_name> <command>`: 在正在运行的容器中执行一个命令。
- `docker compose exec web /bin/sh`: 在 web 容器中启动一个 shell，非常适合调试。








