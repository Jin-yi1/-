# 📚 个人知识库助手

一个基于 **Streamlit + ChromaDB + DeepSeek** 的单文件 RAG 知识库应用：上传自己的文档，用自然语言提问，AI 会优先根据你的资料回答。

无需向量模型下载、无需独立后端服务，`pip install` 后一条命令即可运行。

## ✨ 功能

- **多格式文档入库**：支持 `txt` / `md` / `pdf` / `docx`，自动切分为 500 字的片段并写入向量库
- **RAG 检索增强问答**：每次提问检索 Top-5 相关片段，拼接进系统提示词后再交给大模型
- **对话短期记忆**：自动携带最近 5 轮对话（10 条消息）作为上下文
- **流式输出**：回答逐字显示，不用干等
- **一键复制**：每条 AI 回答下方提供可复制的代码块按钮
- **历史记录侧边栏**：展示最近 10 条提问，点击即可重新提问
- **快捷指令**：输入 `/总结` 总结上一轮对话，输入 `/清空` 清空对话历史
- **双主题**：默认夜间模式，可切换为柔和护眼的米白亮色模式
- **微信式气泡**：用户消息靠右绿色、AI 消息靠左灰白，阅读更轻松
- **异常兜底**：单个文件解析或入库失败只提示错误，不会导致服务崩溃断连

## 🧱 技术栈

| 层次 | 选型 | 说明 |
| --- | --- | --- |
| 前端 / 后端 | Streamlit | 单进程单体架构，UI 与业务逻辑全部在 `app.py` |
| 对话大模型 | DeepSeek（OpenAI 兼容协议） | `openai` SDK，`base_url=https://api.deepseek.com/v1`，流式输出 |
| 向量数据库 | ChromaDB（PersistentClient） | 本地持久化，集合名 `knowledge_base`，余弦相似度 |
| 向量化 | 自研哈希 n-gram 嵌入 | 字级 1/2/3-gram → CRC32 → 4096 维 → L2 归一化，零模型下载、支持中文 |
| 文本切分 | langchain-text-splitters | `RecursiveCharacterTextSplitter`，chunk 500 / overlap 50 |
| 文档解析 | pypdf / python-docx | 解析 PDF 与 Word；纯文本先 UTF-8，失败回退 GBK |
| 存储 | 系统临时目录 | `%TEMP%/kb_app_chroma_db`，重启或清理临时目录后数据会丢失 |

## 📂 项目结构

```text
knowledge-base-assistant/
├── app.py                  # 全部应用代码（页面、检索、问答、样式）
├── requirements.txt        # 依赖清单
├── .streamlit/
│   └── config.toml         # Streamlit 运行配置（headless、关闭统计上报）
├── .gitignore              # 排除缓存、密钥、本地向量库
└── README.md
```

## 🚀 本地运行

1. 安装依赖（建议使用虚拟环境）：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

2. 启动应用：

```bash
streamlit run app.py
```

3. 浏览器打开 `http://localhost:8501`，在左侧边栏填入 DeepSeek API Key（[platform.deepseek.com](https://platform.deepseek.com/) 获取，`sk-` 开头），然后上传文档 → 点击「📥 加入知识库」→ 开始提问。

> API Key 只在当前会话中使用，不会写入任何文件。若提示包含非英文字符，通常是输入法混入了全角字符，请切换英文输入法重新输入。

## ☁️ 部署到 Streamlit Community Cloud

1. 将本仓库推送到 GitHub（保持 `app.py` 与 `requirements.txt` 在根目录）。
2. 打开 [share.streamlit.io](https://share.streamlit.io/)，选择该仓库，主文件填 `app.py`。
3. 部署完成后在页面左侧填入 API Key 即可使用。

## ⚠️ 说明与限制

- 向量库保存在**临时目录**，重新部署或系统清理后知识库会被清空，需要重新上传文档。
- 哈希 n-gram 嵌入不需要下载模型、速度快也支持中文，但语义理解能力弱于专业嵌入模型，适合轻量个人场景。
- 知识库数据按会话共享（单机部署时所有访客共享同一份向量库）。

## 📄 License

如需开源，建议补充一个 `LICENSE` 文件（例如 MIT）。
