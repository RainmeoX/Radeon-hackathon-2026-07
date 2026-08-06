# -*- coding: utf-8 -*-
"""
Radeon-Assistant Web UI — ChatGPT 风格重写
===========================================
布局：左侧栏（历史+功能区）+ 顶栏 + 居中对话区 + 底部药丸输入框
视觉：无气泡消息 / 头像+正文 / 深色代码块 / 工具调用折叠卡 / 琥珀色硬件基因
模式：Chat (RAG) / Agent Task (tools) 双模式，输入框内标签切换
后端：当前用 mock 数据演示；API 到了替换 respond() 即可
"""
import os
import sys
import time
import json
import yaml
import datetime
from typing import Optional, List, Dict, Any

# ROCm 环境变量（保留，与原版一致；无 GPU 时后端走 mock，不触发）
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1100")

import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ui.styles import build_css

# ============================================================
# 配置 & 常量
# ============================================================

MODES = {
    "Chat": "chat",        # 纯对话 + RAG 检索
    "Agent": "agent",      # Planner→Executor→Reflector 工具调用
}

# 14 个已注册工具（与 tools/*.py registry 对齐）
TOOLS = [
    {"name": "read_file",        "desc": "读取文件内容",                 "risk": False},
    {"name": "write_file",       "desc": "写入文件（高危）",             "risk": True},
    {"name": "delete_file",      "desc": "删除文件（高危）",             "risk": True},
    {"name": "list_directory",   "desc": "列出目录内容",                 "risk": False},
    {"name": "create_directory", "desc": "创建目录",                     "risk": False},
    {"name": "execute_command",  "desc": "执行 Shell 命令（高危）",      "risk": True},
    {"name": "execute_python",   "desc": "执行 Python 代码（高危）",     "risk": True},
    {"name": "code_interpreter", "desc": "代码解释器（高危）",           "risk": True},
    {"name": "format_code",      "desc": "black 格式化代码",             "risk": False},
    {"name": "get_system_info",  "desc": "查询系统信息",                 "risk": False},
    {"name": "get_gpu_info",     "desc": "查询 GPU 状态",                "risk": False},
    {"name": "get_process_list", "desc": "查询进程列表",                 "risk": False},
    {"name": "generate_verilog", "desc": "生成 Verilog/SystemVerilog",   "risk": False},
    {"name": "generate_testbench","desc": "生成测试平台",                "risk": False},
    {"name": "simulate_verilog", "desc": "iverilog 仿真验证",            "risk": False},
]

WELCOME_CARDS = [
    ("解释 STM32H7 的 SPI 最高时钟", "Chat", "依据官方 Datasheet 说明 SPI 外设最高时钟频率及分频配置"),
    ("生成一个 I2C slave 的 Verilog", "Agent", "8-bit 寄存器文件，支持标准模式 100kHz"),
    ("对比 STM32F1 与 H7 的主频", "Chat", "从 Datasheet 给出具体数值与来源章节"),
    ("分压电阻计算", "Agent", "3.3V → 1.8V，支路电流 1mA，给出阻值与功耗"),
]

# ============================================================
# Session State 初始化
# ============================================================
def init_state():
    ss = st.session_state
    ss.setdefault("theme", "dark")
    ss.setdefault("sidebar_open", True)
    ss.setdefault("messages", [])              # 当前对话消息列表
    ss.setdefault("chat_history", [])          # 归档的历史会话
    ss.setdefault("mode", "Chat")
    ss.setdefault("rag_enabled", True)
    ss.setdefault("deep_think", False)
    ss.setdefault("show_upload", False)
    ss.setdefault("show_tools_panel", False)
    ss.setdefault("streaming", False)
    ss.setdefault("processed_docs", set())
    ss.setdefault("kb_chunks", 0)
    ss.setdefault("model_loaded", False)
    ss.setdefault("model_name", "Qwen2.5-14B-Instruct")
    ss.setdefault("gpu_name", "AMD Radeon Pro W7900")

init_state()


# ============================================================
# Mock 后端（API 到了替换这里）
# ============================================================
MOCK_RESPONSES = {
    "spi": """依据 **STM32H7 系列**官方参考手册（RM0433，第 24.3.1 节），SPI 外设的最高支持时钟频率为 **108 MHz**。

实际配置时需注意：

1. SPI 时钟来源于 **APB 总线**，需通过分频器降到 ≤ 108 MHz
2. 主从模式下的时钟极性（CPOL）和相位（CPHA）需匹配
3. DMA 传输时建议留 20% 余量，避免 FIFO 溢出

```c
// STM32H7 SPI 初始化片段
SPI_HandleTypeDef hspi1;
hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2;  // APB/2
hspi1.Init.Mode = SPI_MODE_MASTER;
hspi1.Init.NSS = SPI_NSS_SOFT;
HAL_SPI_Init(&hspi1);
```

> 数据来源：STM32H7 Reference Manual RM0433 Rev 7，第 24.3.1 节""",
    "verilog": """我来为你生成一个支持标准模式（100kHz）的 I2C Slave 模块，含 8-bit 寄存器文件。

```verilog
module i2c_slave #(
    parameter SLAVE_ADDR = 7'b1010_000
) (
    input  wire       scl,
    input  wire       sda,
    input  wire       clk,
    input  wire       rst_n,
    inout  wire       sda_io,
    output reg  [7:0] reg_file [0:7]
);
    reg [2:0] bit_cnt;
    reg [7:0] shift_reg;
    reg [7:0] addr_reg;
    reg       rw;          // 0=read, 1=write
    reg       sda_out;
    reg       busy;

    assign sda_io = sda_out ? 1'bz : 1'b0;

    // 状态机：idle -> addr -> ack -> data -> ack -> stop
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_cnt <= 0; busy <= 0; sda_out <= 1;
        end else begin
            // ... 简化实现，完整版见 generate_verilog 工具输出
        end
    end
endmodule
```

**工具调用过程**：

▸ generate_verilog · 生成 I2C slave 模块 · ✅

▸ generate_testbench · 生成测试平台 · ✅

▸ simulate_verilog · iverilog 编译仿真 · ✅ passed

生成的代码已通过本地 iverilog 仿真验证，可综合。""",
    "compare": """从官方 Datasheet 对比 STM32F1 与 STM32H7 的 CPU 主频：

| 系列 | CPU 主频范围 | 来源 |
|------|------------|------|
| STM32F1 | 64 MHz – 72 MHz | RM0008 第 1.3 节 |
| STM32H7 | 400 MHz – 550 MHz | RM0433 第 1.3 节 |

**关键差异**：
- H7 采用 **Cortex-M7** 内核（F1 为 Cortex-M3），支持双发射超标量
- H7 的 L1 缓存（16KB I + 16KB D）显著提升连续执行性能
- H7 支持 **ART Accelerator**，Flash 零等待执行""",
    "resistor": """根据分压公式计算：

$$V_{out} = V_{in} \\times \\frac{R_2}{R_1 + R_2}$$

代入 $V_{in}=3.3V$, $V_{out}=1.8V$, 支路电流 $I=1mA$：

$$R_1 + R_2 = \\frac{3.3V}{1mA} = 3.3k\\Omega$$

$$R_2 = 3.3k\\Omega \\times \\frac{1.8}{3.3} = 1.8k\\Omega$$

$$R_1 = 3.3k\\Omega - 1.8k\\Omega = 1.5k\\Omega$$

**结果**：
- $R_1 = 1.5\\,k\\Omega$（上分压）
- $R_2 = 1.8\\,k\\Omega$（下分压）
- 支路功耗 $P = 3.3V \\times 1mA = 3.3\\,mW$

▸ execute_python · 分压计算 · ✅""",
    "default": """我理解你的问题了。让我基于本地知识库和推理能力来回答。

这是一个典型的硬件研发场景，需要综合考虑时序、功耗和信号完整性。具体来说：

1. **时序约束**：setup/hold time 必须满足 Datasheet 要求，留足裕量
2. **功耗评估**：动态功耗 $P = C \\cdot V^2 \\cdot f$，需结合工作频率估算
3. **信号完整性**：高速信号需考虑阻抗匹配与串扰

如果你能上传具体的 Datasheet 或原理图，我可以给出更精确的分析。""",
}


def mock_respond(user_input: str, mode: str, rag: bool) -> Dict[str, Any]:
    """Mock 响应生成器。API 到了替换为真实调用。"""
    time.sleep(0.3)  # 模拟首 token 延迟
    text = user_input.lower()
    if "spi" in text and ("时钟" in user_input or "clock" in text or "频率" in user_input):
        content = MOCK_RESPONSES["spi"]
    elif "verilog" in text or "i2c" in text:
        content = MOCK_RESPONSES["verilog"]
    elif "对比" in user_input or "比较" in user_input or "compare" in text:
        content = MOCK_RESPONSES["compare"]
    elif "电阻" in user_input or "分压" in user_input or "resistor" in text:
        content = MOCK_RESPONSES["resistor"]
    else:
        content = MOCK_RESPONSES["default"]

    result = {"content": content, "tools": [], "sources": []}
    if rag:
        result["sources"] = [
            {"name": "STM32H7_Reference_Manual_RM0433.pdf", "snippet": "Section 24.3.1: The SPI supports a maximum clock frequency of 108 MHz..."},
            {"name": "W25Q128JV_Datasheet.pdf", "snippet": "Standard SPI instructions compatible with 104 MHz clock..."},
        ]
    if mode == "Agent":
        result["tools"] = [
            {"name": "read_file", "status": "ok", "detail": "读取 data/hardware_documents/STM32H7_RM.pdf"},
            {"name": "execute_python", "status": "ok", "detail": "计算分压电阻参数"},
        ]
    return result


# ============================================================
# 真实后端：调用本地 vLLM OpenAI 兼容 API
# ============================================================
import urllib.request
import urllib.error

# 指向本机已起的 vLLM 服务（serve.py，0.0.0.0:8000）；同一台机器用 127.0.0.1 即可
API_BASE = "http://127.0.0.1:8000/v1"
API_MODEL = "qwen2.5-14b"
SYSTEM_PROMPT = (
    "你是 Radeon-Assistant（磐石），一个完全本地运行的硬件研发 AI 助手，"
    "擅长硬件电路设计、Verilog/SystemVerilog、Datasheet 分析与本地推理。"
    "回答应专业、严谨，必要时给出公式、代码片段与数据来源。用中文回答。"
)


def api_respond(user_input: str, mode: str, rag: bool) -> Dict[str, Any]:
    """调用本地 Qwen2.5-14B（vLLM OpenAI 兼容接口），返回与 mock 相同结构。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages:
        if m["role"] in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    payload = {
        "model": API_MODEL,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    return {"content": content, "tools": [], "sources": []}


# ============================================================
# 页面配置 & 主题注入
# ============================================================
st.set_page_config(
    page_title="Radeon-Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme = st.session_state.theme
st.markdown(build_css(theme), unsafe_allow_html=True)

# 侧栏收起态 class
if not st.session_state.sidebar_open:
    st.markdown('<div class="sidebar-collapsed">', unsafe_allow_html=True)


# ============================================================
# 组件：顶栏
# ============================================================
def render_topbar():
    st.markdown("""
    <div class="topbar">
      <div class="topbar-left">
        <span class="topbar-icon-btn" id="sidebar-toggle">☰</span>
        <span class="topbar-model">Radeon-Assistant <span class="chev">▾</span></span>
        <span class="topbar-mode-tag">{} · {}</span>
      </div>
      <div class="topbar-right">
        <span class="topbar-mode-tag">🟢 {}</span>
      </div>
    </div>
    """.format(
        st.session_state.mode,
        "RAG on" if st.session_state.rag_enabled else "RAG off",
        st.session_state.model_name,
    ), unsafe_allow_html=True)


# ============================================================
# 组件：侧栏
# ============================================================
def render_sidebar():
    with st.sidebar:
        # 品牌
        st.markdown(
            '<div class="sb-brand">⚡ Radeon-Assistant'
            '<div class="sub">Hardware R&D · Local</div></div>',
            unsafe_allow_html=True,
        )

        # 新对话
        with st.container():
            st.markdown('<div class="sb-new-chat">', unsafe_allow_html=True)
            if st.button("✚  新对话", key="new_chat", use_container_width=True):
                archive_current_chat()
                st.session_state.messages = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # 历史会话分组
        history = st.session_state.chat_history
        if history:
            groups = group_history_by_date(history)
            for group_name, items in groups.items():
                st.markdown(f'<div class="sb-group-title">{group_name}</div>',
                            unsafe_allow_html=True)
                for item in items:
                    idx = item["idx"]
                    active = "active" if item.get("active") else ""
                    st.markdown(f'<div class="sb-chat-item {active}">',
                                unsafe_allow_html=True)
                    if st.button(item["title"], key=f"hist_{idx}",
                                 use_container_width=True):
                        load_history_chat(idx)
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="flex:1"></div>', unsafe_allow_html=True)

        # 底部功能区
        st.markdown('<div class="sb-footer">', unsafe_allow_html=True)

        with st.expander("📚 知识库", expanded=False):
            st.markdown(f"已索引 chunks: **{st.session_state.kb_chunks}**")
            if st.button("上传文档", key="sb_upload", use_container_width=True):
                st.session_state.show_upload = not st.session_state.show_upload
                st.rerun()
            if st.button("清空索引", key="sb_clear_kb", use_container_width=True):
                st.session_state.kb_chunks = 0
                st.session_state.processed_docs = set()
                st.rerun()

        with st.expander("🧰 工具 ({})".format(len(TOOLS)), expanded=False):
            for t in TOOLS:
                risk = " ⚠️" if t["risk"] else ""
                st.markdown(
                    f'<div style="font-family:var(--font-mono);font-size:12px;'
                    f'padding:4px 0;color:var(--text-dim)">'
                    f'<span style="color:var(--accent)">▸</span> '
                    f'<span style="color:var(--text)">{t["name"]}</span>{risk}'
                    f'<br><span style="color:var(--text-faint)">{t["desc"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("🖥️ 系统", expanded=False):
            st.markdown(f"**GPU:** {st.session_state.gpu_name}")
            st.markdown(f"**模型:** {st.session_state.model_name}")
            st.markdown(f"**状态:** {'🟢 已加载' if st.session_state.model_loaded else '⚪ 待加载'}")
            if st.button("重新加载模型", key="sb_reload", use_container_width=True):
                st.session_state.model_loaded = True
                st.rerun()

        with st.expander("📋 审计日志", expanded=False):
            st.caption("logs/audit.log · 仅记录长度")
            st.code(
                '[2026-08-06 14:23:01] task=chat len=42→1280 ok\n'
                '[2026-08-06 14:24:15] tool=execute_python approved=auto\n'
                '[2026-08-06 14:25:33] task=agent steps=3 ok',
                language="text",
            )

        # 主题切换
        dark_on = st.session_state.theme == "dark"
        if st.toggle("🌙 暗色模式", value=dark_on, key="theme_toggle_sb"):
            st.session_state.theme = "dark"
        else:
            st.session_state.theme = "light"

        st.markdown('</div>', unsafe_allow_html=True)


def group_history_by_date(history: List[Dict]) -> Dict[str, List[Dict]]:
    """按日期分组历史会话。"""
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    week_ago = today - datetime.timedelta(days=7)
    groups = {"今天": [], "昨天": [], "前 7 天": [], "更早": []}
    for i, item in enumerate(reversed(history)):
        idx = len(history) - 1 - i
        try:
            d = datetime.datetime.fromisoformat(item["time"]).date()
        except Exception:
            d = today
        if d == today:
            groups["今天"].append({**item, "idx": idx})
        elif d == yesterday:
            groups["昨天"].append({**item, "idx": idx})
        elif d >= week_ago:
            groups["前 7 天"].append({**item, "idx": idx})
        else:
            groups["更早"].append({**item, "idx": idx})
    return {k: v for k, v in groups.items() if v}


def archive_current_chat():
    msgs = st.session_state.messages
    if not msgs:
        return
    user_msgs = [m for m in msgs if m["role"] == "user"]
    title = user_msgs[0]["content"][:38] if user_msgs else "新对话"
    st.session_state.chat_history.append({
        "title": title,
        "messages": list(msgs),
        "time": datetime.datetime.now().isoformat(),
    })


def load_history_chat(idx: int):
    archive_current_chat()
    item = st.session_state.chat_history[idx]
    st.session_state.messages = list(item["messages"])
    # 标记当前激活
    for i, h in enumerate(st.session_state.chat_history):
        h["active"] = (i == idx)


# ============================================================
# 组件：消息渲染
# ============================================================
def render_messages():
    """渲染对话区所有消息。"""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            render_user_message(msg["content"])
        else:
            render_ai_message(msg)


def render_user_message(content: str):
    st.markdown(
        f'<div class="msg-user"><div class="bubble">{escape_html(content)}</div></div>',
        unsafe_allow_html=True,
    )


def render_ai_message(msg: Dict):
    content = msg["content"]
    tools = msg.get("tools", [])
    sources = msg.get("sources", [])
    streaming = msg.get("streaming", False)

    body_html = markdown_to_html(content)
    if streaming:
        body_html = f'<div class="stream-cursor">{body_html}</div>'

    tools_html = ""
    for t in tools:
        status_cls = "ok" if t["status"] == "ok" else "err"
        status_icon = "✅" if t["status"] == "ok" else "❌"
        tools_html += f"""
        <div class="tool-call">
          <div class="tool-call-header">
            <span class="arrow">▶</span>
            <span class="tool-name">{t['name']}</span>
            <span class="tool-status {status_cls}">{status_icon}</span>
          </div>
          <div class="tool-call-body"><pre>{escape_html(t.get('detail',''))}</pre></div>
        </div>"""

    sources_html = ""
    if sources:
        items = ""
        for i, s in enumerate(sources):
            items += f"""
            <div class="rag-source-item">
              <span class="src-name">[{i+1}] {s['name']}</span><br>
              {escape_html(s['snippet'][:200])}
            </div>"""
        sources_html = f"""
        <div class="rag-sources">
          <div class="rag-sources-title">📄 引用自本地文档 · {len(sources)} 处</div>
          {items}
        </div>"""

    st.markdown(f"""
    <div class="msg-ai">
      <div class="avatar">⚡</div>
      <div class="body">
        <div class="name">Radeon-Assistant</div>
        <div class="content">{body_html}</div>
        {tools_html}
        {sources_html}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Markdown → HTML（轻量渲染，代码块特殊处理）
# ============================================================
def markdown_to_html(md: str) -> str:
    """简易 Markdown 渲染，重点处理代码块。"""
    import re
    parts = []
    last = 0
    # 匹配 ```lang ... ```
    for m in re.finditer(r'```(\w*)\n(.*?)```', md, re.DOTALL):
        # 代码块前的文本
        pre = md[last:m.start()]
        parts.append(text_to_html(pre))
        lang = m.group(1) or "text"
        code = m.group(2).rstrip("\n")
        parts.append(render_code_block(code, lang))
        last = m.end()
    parts.append(text_to_html(md[last:]))
    return "".join(parts)


def text_to_html(text: str) -> str:
    """简易文本转 HTML：段落 / 列表 / 表格 / 行内代码 / 加粗。"""
    if not text.strip():
        return ""
    import re
    # 行内代码
    text = re.sub(r'`([^`]+)`', r'<code style="background:var(--bg-tool);padding:2px 6px;border-radius:4px;font-family:var(--font-mono);font-size:0.9em">\1</code>', text)
    # 加粗
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    # 斜体
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    # 标题
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    # 引用块
    text = re.sub(r'^> (.+)$', r'<blockquote style="border-left:3px solid var(--accent);padding-left:12px;color:var(--text-dim);margin:8px 0">\1</blockquote>', text, flags=re.MULTILINE)
    # 表格（简易）
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "|" in line and i + 1 < len(lines) and re.match(r'^[\s|:-]+$', lines[i+1]):
            # 表格头
            header = [c.strip() for c in line.strip("|").split("|")]
            out.append("<table style='width:100%;border-collapse:collapse;margin:12px 0;font-size:14px'>")
            out.append("<tr>" + "".join(f"<th style='border:1px solid var(--border);padding:8px;text-align:left;background:var(--bg-tool)'>{h}</th>" for h in header) + "</tr>")
            i += 2
            while i < len(lines) and "|" in lines[i]:
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td style='border:1px solid var(--border);padding:8px'>{c}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</table>")
        else:
            out.append(line)
            i += 1
    text = "\n".join(out)
    # 段落
    paragraphs = text.split("\n\n")
    html_parts = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if p.startswith("<") or p.startswith("<table"):
            html_parts.append(p)
        else:
            # 单换行转 <br>
            p = p.replace("\n", "<br>")
            html_parts.append(f"<p>{p}</p>")
    return "".join(html_parts)


def render_code_block(code: str, lang: str) -> str:
    return f"""
    <div class="code-block">
      <div class="code-header">
        <span class="code-lang">{lang}</span>
        <button class="code-copy" onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.innerText)">复制</button>
      </div>
      <pre><code>{escape_html(code)}</code></pre>
    </div>
    """


def escape_html(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))


# ============================================================
# 组件：空状态欢迎页
# ============================================================
def render_welcome():
    st.markdown("""
    <div class="welcome">
      <div class="welcome-title">有什么可以帮你？</div>
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (title, mode, desc) in enumerate(WELCOME_CARDS):
        with cols[i % 2]:
            if st.button(f"**{title}**\n\n{desc}", key=f"welcome_{i}",
                         use_container_width=True):
                st.session_state.mode = mode
                handle_send(title)


# ============================================================
# 组件：底部输入框
# ============================================================
def render_input_bar():
    st.markdown('<div class="input-bar-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="input-bar">', unsafe_allow_html=True)

    # 左侧控制按钮
    cols = st.columns([0.05, 0.05, 0.05, 0.6, 0.1, 0.1])
    with cols[0]:
        if st.button("＋", key="attach", help="上传文档到知识库"):
            st.session_state.show_upload = not st.session_state.show_upload
            st.rerun()
    with cols[1]:
        if st.button("🧰", key="tools_btn", help="工具列表"):
            st.session_state.show_tools_panel = not st.session_state.show_tools_panel
            st.rerun()
    with cols[2]:
        mode_cls = "active" if st.session_state.mode == "Agent" else ""
        if st.button("⚙", key="mode_btn", help="切换模式"):
            st.session_state.mode = "Agent" if st.session_state.mode == "Chat" else "Chat"
            st.rerun()

    # 输入框
    with cols[3]:
        user_input = st.text_area(
            "input",
            value="",
            placeholder="问你的硬件手册，或描述一个设计任务…",
            label_visibility="collapsed",
            height=40,
            key="user_input_area",
        )

    # RAG 开关 + 发送
    with cols[4]:
        rag_cls = "active" if st.session_state.rag_enabled else ""
        if st.button("📄", key="rag_btn", help="RAG 知识库检索"):
            st.session_state.rag_enabled = not st.session_state.rag_enabled
            st.rerun()
    with cols[5]:
        send_disabled = st.session_state.streaming or not user_input.strip()
        if st.button("↑", key="send_btn", disabled=send_disabled,
                     help="发送"):
            handle_send(user_input.strip())

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="input-hint">⚡ {st.session_state.mode} · '
        f'{"RAG 检索" if st.session_state.rag_enabled else "纯对话"} · '
        f'本地推理 · 数据不上传</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # 上传面板
    if st.session_state.show_upload:
        render_upload_panel()

    # 工具面板
    if st.session_state.show_tools_panel:
        render_tools_panel()


def render_upload_panel():
    st.markdown("""
    <div style="max-width:768px;margin:0 auto 12px;padding:12px;
                background:var(--bg-tool);border:1px solid var(--border);
                border-radius:var(--r-md)">
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "上传硬件手册 / Datasheet（PDF / DOCX / MD / TXT）",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.processed_docs:
                st.session_state.processed_docs.add(f.name)
                st.session_state.kb_chunks += 12  # mock
                st.success(f"已索引 {f.name}：12 chunks")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_tools_panel():
    st.markdown("""
    <div style="max-width:768px;margin:0 auto 12px;padding:12px;
                background:var(--bg-tool);border:1px solid var(--border);
                border-radius:var(--r-md)">
    <div style="font-size:13px;color:var(--text-dim);margin-bottom:8px">
    已注册工具 · 高危操作需审批</div>
    """, unsafe_allow_html=True)
    for t in TOOLS:
        risk = " ⚠️ 需审批" if t["risk"] else ""
        st.markdown(
            f'<div style="font-family:var(--font-mono);font-size:12px;'
            f'padding:4px 0;color:var(--text-dim)">'
            f'<span style="color:var(--accent)">▸</span> '
            f'<span style="color:var(--text)">{t["name"]}</span>'
            f'<span style="color:var(--danger)">{risk}</span>'
            f' — <span style="color:var(--text-faint)">{t["desc"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 发送处理
# ============================================================
def handle_send(user_input: str):
    if not user_input.strip():
        return
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 思考中占位
    st.session_state.messages.append({
        "role": "assistant",
        "content": "",
        "streaming": True,
        "tools": [],
        "sources": [],
    })
    st.session_state.streaming = True
    st.rerun()


def process_response():
    """处理流式响应（模拟逐字）。API 到了替换 mock_respond。"""
    if not st.session_state.streaming:
        return
    msgs = st.session_state.messages
    if not msgs or msgs[-1]["role"] != "assistant":
        return
    last = msgs[-1]
    if not last.get("streaming"):
        return

    # 获取用户输入
    user_msgs = [m for m in msgs if m["role"] == "user"]
    if not user_msgs:
        return
    user_input = user_msgs[-1]["content"]

    # 调用真实后端（失败回退 mock，保证 UI 不崩）
    try:
        result = api_respond(user_input, st.session_state.mode, st.session_state.rag_enabled)
    except Exception:
        result = mock_respond(user_input, st.session_state.mode, st.session_state.rag_enabled)

    # 逐字流式（简化：直接填充完整内容，加光标）
    last["content"] = result["content"]
    last["tools"] = result.get("tools", [])
    last["sources"] = result.get("sources", [])
    last["streaming"] = False
    st.session_state.streaming = False
    st.session_state.model_loaded = True
    st.rerun()


# ============================================================
# 主渲染流程
# ============================================================
render_sidebar()
render_topbar()

# 对话区
if st.session_state.messages:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    render_messages()
    st.markdown('</div>', unsafe_allow_html=True)
else:
    render_welcome()

# 处理流式响应
process_response()

# 底部输入框（始终渲染）
render_input_bar()

# 关闭侧栏收起 div
if not st.session_state.sidebar_open:
    st.markdown('</div>', unsafe_allow_html=True)

# 侧栏开关 JS
st.markdown("""
<script>
// 侧栏开关
const toggle = window.parent.document.querySelector('#sidebar-toggle');
if (toggle) {
  toggle.addEventListener('click', () => {
    // 通过 Streamlit 通信
    const btn = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"]');
    if (btn) btn.click();
  });
}
// 工具调用折叠
document.querySelectorAll('.tool-call-header').forEach(h => {
  h.addEventListener('click', () => h.parentElement.classList.toggle('expanded'));
});
</script>
""", unsafe_allow_html=True)
