import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.api.types import EmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document
import io
import os
import zlib
import numpy as np

st.set_page_config(page_title="个人知识库助手", page_icon="📚")

# ---------- 轻量哈希嵌入（零下载，支持中文） ----------
class HashingEmbedding(EmbeddingFunction):
    def __init__(self, dim=4096):
        self.dim = dim

    def __call__(self, input):
        out = []
        for doc in input:
            vec = np.zeros(self.dim, dtype=np.float32)
            text = doc.lower()
            for n in (1, 2, 3):
                length = len(text) - n + 1
                for i in range(length):
                    idx = zlib.crc32(text[i : i + n].encode("utf-8")) % self.dim
                    vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            out.append(vec)
        return out

    @staticmethod
    def name():
        return "hashing_ngram"

    @staticmethod
    def build_from_config(config):
        return HashingEmbedding(dim=config.get("dim", 4096))

    def get_config(self):
        return {"dim": self.dim}

    def default_space(self):
        return "cosine"


@st.cache_resource
def get_collection():
    db_path = os.path.join(os.environ.get("TEMP", "."), "kb_app_chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    ef = HashingEmbedding(dim=4096)
    collection = client.get_or_create_collection(
        name="knowledge_base", embedding_function=ef
    )
    return collection


collection = get_collection()


# ---------- 文件解析 ----------
def parse_uploaded_file(file, ext):
    """解析上传的文件为纯文本。解析失败会抛异常。"""
    if ext in (".txt", ".md"):
        data = file.getvalue()
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("gbk", errors="ignore")
    elif ext == ".pdf":
        reader = PdfReader(io.BytesIO(file.getvalue()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".docx":
        doc = Document(io.BytesIO(file.getvalue()))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"不支持的文件类型：{ext}")


# ---------- 初始化 session_state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # 默认夜间模式

# ---------- 主题 CSS 注入（暗色/亮色双向强制） ----------
if st.session_state.dark_mode:
    # 夜间模式
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            background-color: #0e1117 !important;
            color: #e8e6e3 !important;
        }
        [data-testid="stSidebar"] * { color: #e8e6e3 !important; }
        [data-testid="stMarkdownContainer"] p, h1, h2, h3, h4, h5, h6 {
            color: #e8e6e3 !important;
        }
        input, textarea, [data-baseweb="input"] {
            background-color: #262730 !important;
            color: #e8e6e3 !important;
        }
        /* 聊天气泡：AI 左侧灰蓝，用户右侧暖绿 */
        [data-testid="stChatMessage"] {
            border-radius: 14px; padding: 6px 14px; margin: 4px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    # 护眼亮色：米白背景 + 低对比文字
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            background-color: #f5f1e8 !important;
            color: #4a4543 !important;
        }
        [data-testid="stSidebar"] * { color: #4a4543 !important; }
        [data-testid="stMarkdownContainer"] p, h1, h2, h3, h4, h5, h6 {
            color: #4a4543 !important;
        }
        input, textarea, [data-baseweb="input"] {
            background-color: #efe9dd !important;
            color: #4a4543 !important;
        }
        [data-testid="stChatMessage"] {
            border-radius: 14px; padding: 6px 14px; margin: 4px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------- 微信风格气泡：用户右、AI 左（两种主题共用） ----------
# 说明：不同 Streamlit 版本头像 testid 不同，这里并列多个选择器保证命中
st.markdown(
    """
    <style>
    /* 用户气泡：靠右、绿色（微信风） */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        background-color: rgba(149, 236, 105, 0.25) !important;
        border-radius: 14px;
    }
    /* AI 气泡：靠左、灰白 */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        flex-direction: row;
        background-color: rgba(127, 127, 127, 0.12) !important;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- 侧边栏：外观 + 模型配置 + 知识库 + 历史记录 ----------
with st.sidebar:
    st.toggle("🌙 暗色模式", key="dark_mode")

    st.divider()
    st.header("⚙️ 模型配置")
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        help="在 DeepSeek 开放平台（platform.deepseek.com）获取，sk- 开头",
    )
    model = st.selectbox("模型", ["deepseek-chat", "deepseek-reasoner"])
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("📁 知识库")
    uploaded_files = st.file_uploader(
        "上传文档（txt / md / pdf / docx）",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if st.button("📥 加入知识库"):
        if not uploaded_files:
            st.warning("请先上传文件")
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""],
            )
            total = 0
            for file in uploaded_files:
                filename = file.name
                ext = os.path.splitext(filename)[1].lower()
                try:
                    raw = parse_uploaded_file(file, ext)
                except Exception as e:
                    st.error(f"❌ 文件解析失败：{filename}（{e}）")
                    continue
                if not raw or not raw.strip():
                    st.warning(f"⚠️ {filename} 内容为空，已跳过")
                    continue
                try:
                    existing = collection.get(where={"source": filename})
                    if existing and existing.get("ids"):
                        collection.delete(ids=existing["ids"])
                    docs = text_splitter.create_documents(
                        texts=[raw],
                        metadatas=[{"source": filename}],
                    )
                    if docs:
                        collection.add(
                            ids=[f"{filename}::{i}" for i in range(len(docs))],
                            documents=[d.page_content for d in docs],
                            metadatas=[d.metadata for d in docs],
                        )
                        total += len(docs)
                except Exception as e:
                    st.error(f"❌ 入库失败：{filename}（{e}）")
            if total:
                st.success(f"已加入 {total} 个文档片段")
            st.rerun()
    if st.button("🗑️ 清空知识库"):
        if collection.count() > 0:
            collection.delete(where={"source": {"$ne": ""}})
        st.success("知识库已清空")
        st.rerun()

    doc_count = collection.count()
    st.caption(f"当前知识库片段数：{doc_count}")

    # ---------- 历史记录（最近 10 条用户提问） ----------
    st.divider()
    st.header("🕘 历史记录")
    user_questions = [
        m["content"] for m in st.session_state.messages if m["role"] == "user"
    ]
    if not user_questions:
        st.caption("暂无历史对话")
    else:
        # 只保留最近 10 条，倒序展示（新的在前）
        recent = user_questions[-10:][::-1]
        for i, q in enumerate(recent):
            # 摘要：截断到 30 字
            summary = q if len(q) <= 30 else q[:30] + "…"
            if st.button(summary, key=f"history_{i}", use_container_width=True):
                # 点击历史记录：把该问题重新发送
                st.session_state["pending_prompt"] = q
                st.rerun()

# ---------- 对话主界面 ----------
st.title("📚 个人知识库助手")

if not api_key:
    st.info("请在左侧边栏输入 DeepSeek API Key 开始对话。")
    st.stop()

api_key = api_key.strip()
if not api_key.isascii():
    st.error(
        "API Key 中含有非英文字符（可能是中文输入法或复制时混入的全角字符）。"
        "请切换到英文输入法重新输入，或从 platform.deepseek.com 重新复制。"
    )
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",
)

# 渲染历史消息（assistant 消息带一键复制）
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        # AI 回复下方加一键复制（st.code 自带复制按钮）
        if m["role"] == "assistant":
            with st.expander("📋 复制回答"):
                st.code(m["content"], language=None)

# 处理历史记录点击后的自动重发
pending = st.session_state.pop("pending_prompt", None)

# 聊天输入框
prompt = st.chat_input("输入你的问题...") or pending

if prompt:
    # ---------- 快捷指令 ----------
    if prompt.strip() == "/清空":
        st.session_state.messages = []
        st.rerun()
    elif prompt.strip() == "/总结":
        # 总结上一轮对话（最近一轮 user+assistant）
        if len(st.session_state.messages) < 2:
            st.info("暂无可总结的对话。")
            st.stop()
        last_round = st.session_state.messages[-2:]
        summary_prompt = "请用中文简要总结我们上一轮的对话要点：\n\n" + "\n".join(
            f"{'用户' if m['role'] == 'user' else 'AI'}：{m['content']}" for m in last_round
        )
        st.session_state.messages.append({"role": "user", "content": "/总结"})
        with st.chat_message("user"):
            st.markdown("/总结")
        with st.chat_message("assistant"):
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是对话总结助手，输出简洁的中文要点。"},
                        {"role": "user", "content": summary_prompt},
                    ],
                    stream=True,
                )
                reply = st.write_stream(
                    chunk.choices[0].delta.content or "" for chunk in stream
                )
            except Exception as e:
                reply = None
                st.error(f"调用 DeepSeek 失败：{e}")
        if reply:
            st.session_state.messages.append({"role": "assistant", "content": reply})
    else:
        # ---------- 普通提问 ----------
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 检索知识库
        context = ""
        if collection.count() > 0:
            try:
                results = collection.query(query_texts=[prompt], n_results=5)
                docs = results.get("documents", [[]])[0]
                if docs:
                    context = "\n\n".join(
                        f"[片段 {i+1}]\n{doc}" for i, doc in enumerate(docs)
                    )
            except Exception as e:
                st.error(f"检索知识库失败：{e}")

        system_content = "你是个人知识库助手，请用中文简洁准确地回答。"
        if context:
            system_content += f"\n\n以下知识库内容可能相关，请优先根据这些内容回答：\n\n{context}"

        # 短期记忆：只带最近 5 轮（10 条）对话作为上下文
        memory = st.session_state.messages[-10:]

        with st.chat_message("assistant"):
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_content},
                        *memory,
                    ],
                    stream=True,
                )
                reply = st.write_stream(
                    chunk.choices[0].delta.content or "" for chunk in stream
                )
            except Exception as e:
                reply = None
                st.error(f"调用 DeepSeek 失败：{e}")

        if reply:
            st.session_state.messages.append({"role": "assistant", "content": reply})
