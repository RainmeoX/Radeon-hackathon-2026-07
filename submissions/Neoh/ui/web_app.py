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
    # set_page_config 移除 theme 参数），避免浏览器暗色模式下整页发白。
)

# ---------------------------------------------------------------------------
# 现代样式（浅色、卡片化、AMD 红强调色）
# ---------------------------------------------------------------------------
CSS = """
<style>
/* 隐藏默认顶栏 / 页脚 / 右上角菜单 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden;}

/* 强制深色底（兜底：覆盖浏览器暗色模式下的白底） */
.stApp {
    background: #0d1117 !important;
    color: #e6edf3 !important;
}

/* 主内容居中限宽 */
.main .block-container {
    max-width: 920px;
    margin: 0 auto;
    padding-top: 1rem;
}

/* 顶部品牌栏（Codex 风格：深底 + 细描边） */
.topbar {
    background: linear-gradient(90deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 16px 22px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.topbar .brand { font-size: 22px; font-weight: 700; color: #e6edf3; letter-spacing: .3px; }
.topbar .brand span { color: #ff6b6b; font-weight: 600; }
.topbar .tag { font-size: 12px; color: #8b949e; text-align: right; line-height: 1.4; }

/* 侧栏 */
section[data-testid="stSidebar"] > div:first-child {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #21262d; color: #e6edf3; border: 1px solid #30363d;
    border-radius: 9px; font-weight: 600; width: 100%;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #30363d; border-color: #8b949e;
}

/* 知识库计数徽标 */
.kb-badge {
    display: inline-block; background: #1f6feb22; color: #58a6ff;
    border: 1px solid #1f6feb55; border-radius: 20px;
    padding: 2px 12px; font-size: 13px; font-weight: 600;
}

/* 聊天气泡（深色卡片 + 细描边） */
.stChatMessage {
    border-radius: 14px !important;
    padding: 12px 16px !important;
    margin-bottom: 10px !important;
    border: 1px solid #21262d !important;
    background: #161b22 !important;
    color: #e6edf3 !important;
}
.stChatMessage[data-testid="stChatMessageContent"] { background: transparent !important; }

/* 来源展开 */
.streamlit-expanderHeader { font-size: 13px; color: #8b949e !important; }

/* 输入条 */
.stChatInput { border-radius: 12px; }
.stChatInput textarea {
    background: #0d1117 !important; color: #e6edf3 !important;
    border: 1px solid #30363d !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="topbar">
      <div class="brand">🤖 Bedrock <span>· Hardware R&D Assistant</span></div>
      <div class="tag">100% local inference on AMD Radeon (ROCm)<br/>No external API · Your docs never leave this machine</div>
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
# 侧栏：控制面板
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Assistant mode")
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

    st.markdown("### Knowledge base")
    try:
        doc_count = init_memory().get_document_count()
    except Exception:
        doc_count = 0
    st.markdown(
        f'Indexed chunks: <span class="kb-badge">{doc_count}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("Upload documents — they are indexed automatically (PDF / DOCX / MD / TXT).")
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
    st.caption("All inference runs locally on AMD Radeon via vLLM + ROCm.")

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
