@echo off
chcp 65001 >nul

echo === Radeon-Assistant ROCm 安装脚本 ===
echo.

echo 1. 安装基础依赖...
pip install -r requirements.txt

echo.
echo 2. 设置 ROCm 编译环境变量...
set CMAKE_ARGS=-DGGML_HIPBLAS=ON
set FORCE_CMAKE=1

echo.
echo 3. 安装 llama-cpp-python (ROCm GPU 支持)...
pip install llama-cpp-python --no-cache-dir

echo.
echo 4. 验证安装...
python -c "from llama_cpp import Llama; print('llama-cpp-python 安装成功')"

echo.
echo === 安装完成 ===
echo 请运行: python scripts/download_model.py 下载模型