# Radeon Cloud 跑通指南（Neoh / Radeon-Assistant）

> 本机是 Windows 桌面、无 AMD GPU，无法运行 ROCm 推理。**本指南用于在 Radeon Cloud（Linux + AMD GPU）上把项目一次跑通、录 Demo。**
> 流程严格对齐仓库根目录 `Radeon-Cloud-User Guide/README.md`，请配合其截图使用。

---

## 0. 前置：把代码放到云上

代码有两种上云方式（二选一）：

**方式 A — 从 fork 拉取（推荐，需先 push 到你的 fork）**
```bash
git clone https://github.com/RainmeoX/Radeon-hackathon-2026-07.git
cd Radeon-hackathon-2026-07/submissions/Neoh
```
> 若本地改动尚未 push，请先在本机 `git push -u origin main`（需要 GitHub 认证，见仓库根 README 提交说明）。

**方式 B — 手动上传**
在 JupyterLab 左侧文件浏览器点 `↑` 上传 `submissions/Neoh` 整个文件夹，然后 `cd` 进该目录。

---

## 1. 在 Radeon Cloud 创建实例（照 User Guide）

1. 打开 https://radeon-global.anruicloud.com/ → 右上角 **Login**（邮箱登录）。
2. 右上角头像 → **Profile**。
3. **My Templates** → **Add Template**：
   - **Title**：随便起，如 `neoh-rocm`。
   - **Container Image**：选择带 **ROCm + PyTorch** 的官方镜像（PyTorch ROCm 基础镜像）。
   - **Storage**：设为 **Persistent (PVC)** —— 这样 28GB 模型下载后，销毁/重建实例不丢失。
   - （可选）如需 SSH：打开底部 **SSH Access (advanced)** 开关，并在 Profile 里先添加你的 SSH 公钥。
   - 点 **Add Template**。
4. **My Templates** 列表里点该模板的 **Launch** 启动实例。
5. 等对话框显示 **Your workspace is ready (100%)**。

---

## 2. 进入环境（二选一）

- **Option A · JupyterLab Terminal**：点 **Open Notebook** → 打开 JupyterLab → 顶部 `+` → **Terminal**。
- **Option B · SSH**：`ssh <user>@<host> -p <port>`（实例详情页有现成命令）。

后续命令都在该 Terminal 里执行。

---

## 3. 安装依赖 + vLLM(ROCm)

```bash
cd submissions/Neoh        # 或你上传的目录
bash install_rocm.sh
```

`install_rocm.sh` 会：设置 gfx1100 环境变量 → `pip install -r requirements.txt` → 安装 vLLM 的 ROCm 预编译 wheel → 验证 `torch.cuda.is_available()` 与 vLLM 导入。

> ⚠️ 若验证步骤报 “torch 不是 ROCm 版”，手动装 ROCm 版 PyTorch 后重跑：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
> pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
> ```

---

## 4. 下载模型（默认 14B，约 28GB）

```bash
python scripts/download_model.py --model qwen2.5-14b
```

- 默认走 **ModelScope**（国内最快）；失败自动回退 hf-mirror → HF 官方。
- 模型落到 `./models/Qwen2.5-14B-Instruct`，与 `config.yaml` 的 `model.path` 一致。
- 想先快速冒烟测试可换 7B（`--model qwen2.5-7b`，约 15GB），但正式 Demo 建议用 14B（质量更好，W7900 48GB 放得下）。

---

## 5. 跑起来

### CLI 模式（录 Demo 最方便）
```bash
python app.py --mode cli
```
进去后试：
```
你: 你好，介绍一下你自己
task 用 Verilog 写一个带同步使能的 4 位向上计数器，并生成对应的 testbench
```

### Web 模式（Streamlit）
```bash
python app.py --mode web --port 7860
```
- **JupyterLab 内**：新开 Terminal 跑上面命令，浏览器另开 `http://localhost:7860`（部分镜像需走 JupyterLab 代理）。
- **SSH 方式**：本地执行 `ssh -L 7860:localhost:7860 <user>@<host> -p <port>`，然后浏览器开 `http://localhost:7860`。

---

## 6. 收尾

- Demo 视频建议 3–5 分钟：展示 CLI 多轮对话 + 一次 `task` 多步任务（含硬件工具生成 Verilog）+ 输出落盘。顺手报一下推理速度（tok/s）。
- 不用时务必 **Destroy Instance**（Profile → Active Instance → 红色 Destroy），按量计费。
- 想冲「推理速度优化」20 分 + 量化/蒸馏加分 20 分：额外跑一版 7B 对比 tok/s，或用 Guide 里 **Dedicated Model API（Option 2）** 以量化参数 `vllm serve` 暴露 OpenAI 兼容端点。

---

## 已知限制（如实记录，提交材料里照此写）
- Verilog/Testbench 由 LLM 本地生成（轻量方案），**未接 iverilog/Verilator 做仿真验证**；生成的代码需人工或后续接仿真器核对。
- 硬件能力是对「硬件研发场景」的使用特化（领域 prompt + 硬件工具 + 硬件 RAG 标签 + Datasheet 表格提取），非自研 EDA 内核。
- `config.yaml` 默认 14B / FP16；如需更低延迟可改 7B 或量化。
