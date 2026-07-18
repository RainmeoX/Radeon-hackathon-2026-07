@echo off
chcp 65001 >nul

echo === Radeon-Assistant vLLM (ROCm) 安装脚本 ===
echo.

echo 1. 安装基础依赖...
pip install -r requirements.txt

echo.
echo 2. 安装 vLLM (ROCm 预编译 wheel)...
pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/

echo.
echo 3. 验证安装...
python -c "from vllm import LLM; print('vLLM 安装成功')"
python -c "import torch; print(f'PyTorch ROCm 可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo === 安装完成 ===
echo 请运行: python scripts/download_model.py 下载模型
echo 然后:  python app.py --mode cli  或  python app.py --mode web
