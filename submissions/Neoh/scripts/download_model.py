import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="下载 GGUF 模型")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-7b",
        choices=["qwen2.5-7b", "qwen2.5-14b"],
        help="模型名称"
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="Q4_K_M",
        choices=["Q4_K_M", "Q5_K_M"],
        help="量化类型"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="模型保存目录"
    )
    args = parser.parse_args()

    try:
        loader = ModelLoader(models_dir=args.models_dir)
        model_path = loader.download_model(args.model, args.quant)
        logger.info(f"模型下载成功: {model_path}")
        logger.info(f"模型大小: {os.path.getsize(model_path) / (1024 * 1024 * 1024):.2f} GB")
    except Exception as e:
        logger.error(f"模型下载失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()