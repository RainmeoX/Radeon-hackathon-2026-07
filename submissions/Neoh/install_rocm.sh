#!/bin/bash

echo "=== Radeon-Assistant vLLM (ROCm) 安装脚本 ==="
echo ""

# ---- 设置 ROCm 环境变量（W7900 / RX 7900 系列必需）----
echo "0. 设置 ROCm 环境变量（gfx1100）..."
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH

echo ""
echo "1. 安装基础依赖..."
pip install -r requirements.txt

echo ""
echo "2. 安装 vLLM (ROCm 预编译 wheel)..."
pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/

echo ""
echo "3. 验证安装..."
python -c "
import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1100'
import torch
print(f'PyTorch ROCm 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 数量: {torch.cuda.device_count()}')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
from vllm import LLM
print('vLLM 导入成功')
"

echo ""
echo "=== 安装完成 ==="
echo "请运行: python scripts/download_model.py 下载模型"
echo "然后:  python app.py --mode cli  或  python app.py --mode web"
echo ""
echo "⚠️  如果 GPU 不可用，请在运行前手动执行:"
echo "  export HSA_OVERRIDE_GFX_VERSION=11.0.0"
echo "  export PYTORCH_ROCM_ARCH=gfx1100"
