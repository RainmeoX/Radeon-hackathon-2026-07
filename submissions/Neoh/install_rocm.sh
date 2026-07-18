#!/bin/bash

echo "=== Radeon-Assistant ROCm 安装脚本 ==="
echo ""

echo "1. 安装基础依赖..."
pip install -r requirements.txt

echo ""
echo "2. 设置 ROCm 编译环境变量..."
export CMAKE_ARGS="-DGGML_HIPBLAS=ON"
export FORCE_CMAKE=1

# 检查 HIP_PATH 是否设置
if [ -z "$HIP_PATH" ]; then
    echo "警告: HIP_PATH 未设置，尝试自动检测..."
    if [ -d "/opt/rocm/hip" ]; then
        export HIP_PATH="/opt/rocm/hip"
        echo "自动检测到 HIP_PATH: $HIP_PATH"
    else
        echo "错误: 未找到 ROCm 安装路径"
        echo "请手动设置 HIP_PATH 环境变量"
        exit 1
    fi
fi

echo ""
echo "3. 安装 llama-cpp-python (ROCm GPU 支持)..."
pip install llama-cpp-python --no-cache-dir

echo ""
echo "4. 验证安装..."
python -c "from llama_cpp import Llama; print('llama-cpp-python 安装成功')"

echo ""
echo "=== 安装完成 ==="
echo "请运行: python scripts/download_model.py 下载模型"