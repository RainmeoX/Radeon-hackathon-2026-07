# ---- ROCm 环境变量（必须在所有其他 import 之前设置）----
# W7900 / RX 7900 系列是 gfx1100，vLLM 需要 override 才能识别
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1100")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
# 多卡 TP 必需
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("RCCL_NCCL_NCHANNELS", "4")
os.environ.setdefault("RCCL_NCCL_NSOCKETS_PERCHANNEL", "8")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
os.environ.setdefault("HSA_ENABLE_INTERRUPTIBLE", "0")

import streamlit as st
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import tools  # noqa: F401  导入即触发所有工具注册到 registry
from inference.engine import InferenceEngine, InferenceConfig
from memory.manager import MemoryManager
from agent.core import RadeonAgent

# 界面模式 -> agent prompt_mode
MODE_MAP = {
    "Hardware R&D": "hardware",
    "Generic": "generic",
}

st.set_page_config(
    page_title="Bedrock · Hardware R&D Assistant",
    page_icon="🤖",
    layout="wide",
    # 深色主题通过 .streamlit/config.toml 设置（Streamlit 1.61+ 已从
    # set_page_config 移除 theme 参数）。亮/暗切换由下方 CSS 变量实现。
)

# ---------------------------------------------------------------------------
# ChatGPT / Codex 风格界面：用 CSS 变量实现 亮/暗 双色，清晰边界、区分消息
# 调色板取 ChatGPT 官方 web 设计 token（暗：主区 #343541 / 侧栏 #202123 /
# 输入框 #40414f；亮：主区 #ffffff / 侧栏 #f7f7f8）
# ---------------------------------------------------------------------------
THEMES = {
    "dark": {
        "bg_app": "#202123",
        "bg_sidebar": "#202123",
        "bg_main": "#343541",
        "bg_surface": "#40414f",
        "bg_msg_user": "#40414f",
        "bg_msg_asst": "#343541",
        "border": "#4d4d4f",
        "border_strong": "#565869",
        "text": "#ececf1",
        "text_dim": "#c5c5d2",
        "accent": "#ff6b6b",
        "accent_blue": "#58a6ff",
    },
    "light": {
        "bg_app": "#f7f7f8",
        "bg_sidebar": "#f7f7f8",
        "bg_main": "#ffffff",
        "bg_surface": "#ffffff",
        "bg_msg_user": "#f4f4f5",
        "bg_msg_asst": "#ffffff",
        "border": "#e5e5e5",
        "border_strong": "#d0d0d8",
        "text": "#353740",
        "text_dim": "#8e8ea0",
        "accent": "#e5484d",
        "accent_blue": "#2563eb",
    },
}

_theme = st.session_state.get("theme", "dark")
if _theme not in THEMES:
    _theme = "dark"
_p = THEMES[_theme]

VARS = """
:root {{
  --bg-app: {bg_app};
  --bg-sidebar: {bg_sidebar};
  --bg-main: {bg_main};
  --bg-surface: {bg_surface};
  --bg-msg-user: {bg_msg_user};
  --bg-msg-asst: {bg_msg_asst};
  --border: {border};
  --border-strong: {border_strong};
  --text: {text};
  --text-dim: {text_dim};
  --accent: {accent};
  --accent-blue: {accent_blue};
}}
""".format(**_p)

CSS = "<style>\n" + VARS + """
/* 彻底隐藏 Streamlit 默认头部/页脚，避免占位导致内容整体上移 */
#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }

/* 应用底 / 侧栏背景层 */
.stApp { background: var(--bg-app) !important; color: var(--text) !important; }

/* 主对话列：比背景亮一号，有左右边界（居中列） */
.block-container {
    max-width: 920px;
    margin: 0 auto;
    padding-top: 0;
    padding-bottom: 24px;
    background: var(--bg-main) !important;
    border-left: 1px solid var(--border) !important;
    border-right: 1px solid var(--border) !important;
}

/* 顶部品牌栏：常驻顶部，清晰分隔，仅品牌名（无标语） */
.topbar {
    position: sticky;
    top: 0;
    z-index: 50;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border);
    color: var(--text);
    padding: 14px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.topbar .brand { font-size: 18px; font-weight: 700; letter-spacing: .2px; }
.topbar .brand span { color: var(--accent); font-weight: 600; }

/* 侧栏 */
section[data-testid="stSidebar"] { background: var(--bg-sidebar) !important; }
section[data-testid="stSidebar"] > div:first-child {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: var(--bg-surface); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; font-weight: 600; width: 100%; transition: all .15s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--border-strong); background: var(--bg-main);
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stToggle label,
section[data-testid="stSidebar"] .stCheckbox label { color: var(--text) !important; }
section[data-testid="stSidebar"] .stFileUploader { border-color: var(--border) !important; }
section[data-testid="stSidebar"] .stCaption { color: var(--text-dim) !important; }
section[data-testid="stSidebar"] .stDivider { border-color: var(--border) !important; }

/* 侧栏分区小标题 */
.sb-section { font-size: 12px; font-weight: 700; letter-spacing: .6px;
    text-transform: uppercase; color: var(--text-dim); margin: 4px 0 6px; }

/* 知识库计数徽标 */
.kb-badge {
    display: inline-block;
    background: color-mix(in srgb, var(--accent-blue) 16%, transparent);
    color: var(--accent-blue);
    border: 1px solid color-mix(in srgb, var(--accent-blue) 38%, transparent);
    border-radius: 20px; padding: 2px 12px; font-size: 13px; font-weight: 600;
}

/* 聊天气泡：清晰边界 */
.stChatMessage {
    border-radius: 16px !important; padding: 14px 18px !important;
    margin-bottom: 14px !important; border: 1px solid var(--border) !important;
    background: var(--bg-msg-asst) !important; color: var(--text) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.18) !important;
}
.stChatMessage[data-testid="stChatMessageContent"] { background: transparent !important; }
/* 用户消息：右对齐、明显填充 */
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--bg-msg-user) !important; border-color: var(--border-strong) !important;
    margin-left: 60px !important;
}
/* 助手消息：左对齐，贴主区背景 */
.stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--bg-msg-asst) !important; border-color: var(--border) !important;
    margin-right: 60px !important;
}

/* 来源展开 */
.streamlit-expanderHeader { font-size: 13px; color: var(--text-dim) !important; }

/* 输入框容器：高对比圆角卡片，始终可见 */
.stChatInput {
    background: var(--bg-surface) !important; border: 1px solid var(--border-strong) !important;
    border-radius: 18px !important; box-shadow: 0 4px 18px rgba(0,0,0,0.22) !important;
    padding: 8px 12px !important;
}
.stChatInput textarea { background: transparent !important; color: var(--text) !important;
    border: none !important; font-size: 15px !important; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="topbar">
      <div class="brand">🤖 Bedrock <span>· Hardware R&D Assistant</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 空对话时显示居中的品牌提示（类似 ChatGPT 首页）
if len(st.session_state.get("messages", [])) == 0:
    st.markdown(
        """
        <div style="text-align:center; margin-top: 7vh; color: var(--text-dim);">
            <div style="font-size:48px; margin-bottom:14px;">🤖</div>
            <div style="font-size:26px; font-weight:700; color: var(--text); margin-bottom:8px;">Bedrock</div>
            <div style="font-size:14px; margin-bottom:36px;">Hardware R&D Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 资源初始化
# ---------------------------------------------------------------------------
@st.cache_resource
def init_engine():
    with open(os.path.join(REPO_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    inference_config = InferenceConfig(
        model_path=model_config.get("path", "./models/Qwen2.5-14B-Instruct"),
        n_ctx=model_config.get("n_ctx", 8192),
        temperature=model_config.get("temperature", 0.7),
        max_tokens=model_config.get("max_tokens", 4096),
        dtype=model_config.get("dtype", "float16"),
        gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.90),
        tensor_parallel_size=model_config.get("tensor_parallel_size", 1),
        pipeline_parallel_size=model_config.get("pipeline_parallel_size", 1),
    )

    if not os.path.exists(inference_config.model_path):
        raise RuntimeError(
            f"Model not found: {inference_config.model_path}\n"
            f"Run first: python scripts/download_model.py --model qwen2.5-14b"
        )

    engine = InferenceEngine(inference_config)
    return engine


@st.cache_resource
def init_memory():
    rag_config = {"chunk_size": 512, "chunk_overlap": 50, "top_k": 5}
    memory_manager = MemoryManager(
        index_path=os.path.join(REPO_ROOT, "data", "faiss_index"),
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=rag_config.get("chunk_size", 512),
        chunk_overlap=rag_config.get("chunk_overlap", 50),
        top_k=rag_config.get("top_k", 5),
    )
    return memory_manager


@st.cache_resource
def init_agent():
    engine = init_engine()
    memory_manager = init_memory()

    prompt_mode = "hardware"
    try:
        mode = st.session_state.get("mode", "Hardware R&D")
        prompt_mode = MODE_MAP.get(mode, "hardware")
    except Exception:
        pass

    agent = RadeonAgent(engine, memory_manager, prompt_mode=prompt_mode)
    return agent


# ---------------------------------------------------------------------------
# 侧栏：控制面板（分区清晰）
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div style="font-size:15px;font-weight:700;color:var(--text);">'
        '🤖 Bedrock</div>',
        unsafe_allow_html=True,
    )
    st.caption("Hardware R&D Assistant")

    st.divider()

    # ---- Appearance（亮/暗切换）----
    st.markdown('<div class="sb-section">Appearance</div>', unsafe_allow_html=True)
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    dark_on = st.toggle("Dark mode", value=(st.session_state.theme == "dark"))
    new_theme = "dark" if dark_on else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.divider()

    # ---- Assistant mode ----
    st.markdown('<div class="sb-section">Assistant mode</div>', unsafe_allow_html=True)
    if "mode" not in st.session_state:
        st.session_state.mode = "Hardware R&D"
    new_mode = st.radio(
        "Mode",
        list(MODE_MAP.keys()),
        index=list(MODE_MAP.keys()).index(st.session_state.mode),
        label_visibility="collapsed",
    )
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # ---- Knowledge base ----
    st.markdown('<div class="sb-section">Knowledge base</div>', unsafe_allow_html=True)
    try:
        doc_count = init_memory().get_document_count()
    except Exception:
        doc_count = 0
    st.markdown(
        f'Indexed chunks: <span class="kb-badge">{doc_count}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("Upload PDF / DOCX / MD / TXT — indexed automatically.")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
    )

    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = set()

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if os.path.basename(f.name) not in st.session_state.processed_docs
        ]
        if new_files:
            memory_manager = init_memory()
            save_dir = os.path.join(REPO_ROOT, "data", "documents")
            os.makedirs(save_dir, exist_ok=True)
            for uploaded_file in new_files:
                safe_name = os.path.basename(uploaded_file.name)
                file_path = os.path.join(save_dir, safe_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                added = memory_manager.add_document(file_path)
                if added > 0:
                    st.success(f"Indexed {uploaded_file.name}: {added} chunks")
                    st.session_state.processed_docs.add(safe_name)
                else:
                    st.warning(
                        f"Could not extract text from {uploaded_file.name} "
                        "(likely a scanned / image-only PDF)."
                    )
            st.rerun()

    st.divider()

    # ---- Actions ----
    st.markdown('<div class="sb-section">Actions</div>', unsafe_allow_html=True)
    if st.button("Clear document index"):
        init_memory().clear_long_term_memory()
        st.session_state.processed_docs = set()
        st.success("Document index cleared.")
        st.rerun()

    if st.button("Clear conversation"):
        init_agent().clear_memory()
        st.session_state.messages = []
        st.rerun()

    if st.button("Reload model"):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption("100% local on AMD Radeon · vLLM + ROCm")


# ---------------------------------------------------------------------------
# 对话区
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                for s in message["sources"]:
                    st.caption(f"[Doc {s['index']}] {s['source']}")
                    st.text(s["content"][:600])

if prompt := st.chat_input("Ask about your datasheet, schematic, Verilog, or registers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = init_agent()
                response, sources = agent.chat(prompt, return_sources=True)
                st.markdown(response)
                if sources:
                    with st.expander(f"Sources ({len(sources)})"):
                        for s in sources:
                            st.caption(f"[Doc {s['index']}] {s['source']}")
                            st.text(s["content"][:600])
                st.session_state.messages.append(
                    {"role": "assistant", "content": response, "sources": sources}
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
