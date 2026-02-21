# My Joint Demo - 本地运行指南 (Mac/Windows)

这份指南将帮助你在自己的 Mac 或 Windows 电脑上快速运行起来 **My Joint Demo**。

这个项目由三个部分组成：

1. **前端 (UI)** - 提供摄像头画面和交互界面
2. **Depth Service (深度服务)** - 分析躯干/手部的距离 (基于 PyTorch 运行)
3. **LLM Service (大语言模型)** - 提供 AI 教练反馈 (使用 Ollama 运行)

因为 Mac 和 Windows 对 Docker 的显卡加速支持不够完善，所以我们采用**在宿主机跑 Ollama，在 Docker 里跑深度模型和前端**的方式！

---

## 🛠️ 第一步：环境准备

1. **安装 Docker Desktop**
   请前往 Docker 官网下载并安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)，安装后请确保 Docker 正在后台运行。

2. **安装 Ollama**
   请前往 [Ollama 官网](https://ollama.com/) 下载并安装。Ollama 可以让你在本地非常方便地运行大语言模型。

3. **下载 AI 教练模型**
   打开你的终端 (Terminal / PowerShell)，运行以下命令下载系统默认的反馈模型：

   ```bash
   ollama run qwen2.5:1.5b
   ```

   *注意：如果你在 `docker-compose.yml` 中修改了 `LLM_MODEL` 环境变量，请对应下载你需要的模型，例如 `ollama run qwen3-vl:2b`。*

---

## 🚀 第二步：启动服务

1. **新建工程容器目录**
   在你的电脑上新建一个属于这个程序的文件夹（并进入它）：

   ```bash
   mkdir joint-demo
   cd joint-demo
   ```

2. **获取必要的文件**
   你需要跟分享这个文档给你的同事获取以下两个文件，并将它们放在刚建好的 `joint-demo` 文件夹下：
   - `docker-compose.yml` (配置后端服务的结构)
   - `download_models.sh` (下载 PyTorch 人工智能权重的脚本)

3. **下载 Depth 模型权重**
   深度服务需要依赖一些模型文件。在终端中执行刚刚获取的下载脚本：

   ```bash
   # 该脚本会自动下载权重并保存在当前目录下的 weights 文件夹中
   bash download_models.sh dav2-small
   ```

4. **一键启动 Docker 服务**
   启动容器：

   ```bash
   docker compose up -d
   ```

   *说明：这里不需要加 `--profile linux-gpu` 参数！因为我们直接使用你电脑本机 (Host) 的 Ollama。*

---

## 🎯 第三步：开始体验

等到 Docker 容器都启动成功后，打开你的浏览器访问：

👉 **[http://localhost:3000/](http://localhost:3000/)**

- 首次打开网页时，因为后端的 PyTorch 正在加载 Depth 的模型权重，请耐心等待几秒钟。页面上会显示 Loading 状态。
- 如果右上角的状态指示灯亮**绿灯**，说明服务一切正常！并且界面会自动检测并显示你本地 Ollama 正在运行的默认模型名称。
- 如果你的 Ollama 下载了多个模型，想要**强制指定**使用特定的 AI 教练模型 (例如 `qwen3-vl:2b`)，请打开 `docker-compose.yml`，在 `joint-demo` 服务下的 `environment:` 中填入：`- LLM_MODEL=qwen3-vl:2b`。修改后，只需再次运行 `docker compose up -d` 重建容器，刷新网页即可生效。
