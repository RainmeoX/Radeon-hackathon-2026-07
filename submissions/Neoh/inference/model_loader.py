import logging
import os
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_REPOS = {
    "qwen2.5-7b": "bartowski/Qwen2.5-7B-Instruct-GGUF",
    "qwen2.5-14b": "bartowski/Qwen2.5-14B-Instruct-GGUF",
}

MODEL_PATTERNS = {
    "qwen2.5-7b": {
        "Q4_K_M": "*Q4_K_M.gguf",
        "Q5_K_M": "*Q5_K_M.gguf",
    },
    "qwen2.5-14b": {
        "Q4_K_M": "*Q4_K_M.gguf",
        "Q5_K_M": "*Q5_K_M.gguf",
    },
}

MODEL_FILENAMES = {
    "qwen2.5-7b": {
        "Q4_K_M": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "Q5_K_M": "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
    },
    "qwen2.5-14b": {
        "Q4_K_M": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "Q5_K_M": "Qwen2.5-14B-Instruct-Q5_K_M.gguf",
    },
}

MIRROR_ENDPOINTS = [
    None,
    "https://hf-mirror.com",
    "https://hf-mirror.cn",
]


class ModelLoader:
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def _download_with_huggingface_hub(self, repo_id: str, pattern: str, save_path: str) -> bool:
        try:
            from huggingface_hub import snapshot_download

            for endpoint in MIRROR_ENDPOINTS:
                try:
                    env = os.environ.copy()
                    if endpoint:
                        env["HF_ENDPOINT"] = endpoint

                    logger.info(f"Trying HuggingFace hub with endpoint: {endpoint or 'official'}")
                    downloaded_files = snapshot_download(
                        repo_id=repo_id,
                        allow_patterns=pattern,
                        local_dir=self.models_dir,
                    )

                    if os.path.exists(save_path):
                        logger.info(f"Download succeeded via HuggingFace hub")
                        return True

                    for root, dirs, files in os.walk(self.models_dir):
                        for f in files:
                            if f.endswith(".gguf"):
                                found_path = os.path.join(root, f)
                                if found_path != save_path:
                                    os.rename(found_path, save_path)
                                logger.info(f"Download succeeded via HuggingFace hub")
                                return True
                except Exception as e:
                    logger.warning(f"HuggingFace hub failed with endpoint {endpoint}: {str(e)}")
                    continue

            return False
        except ImportError:
            logger.warning("huggingface_hub not installed")
            return False

    def _download_with_curl(self, url: str, save_path: str) -> bool:
        try:
            logger.info(f"Trying curl download: {url}")
            result = subprocess.run(
                ["curl", "-L", "-o", save_path, url],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if result.returncode == 0 and os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                logger.info("Download succeeded via curl")
                return True
            logger.warning(f"curl failed: {result.stderr}")
            return False
        except Exception as e:
            logger.warning(f"curl download failed: {str(e)}")
            return False

    def _download_with_aria2c(self, url: str, save_path: str) -> bool:
        try:
            logger.info(f"Trying aria2c download: {url}")
            result = subprocess.run(
                ["aria2c", "-x", "16", "-s", "16", "-k", "1M", "-o", save_path, url],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if result.returncode == 0 and os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                logger.info("Download succeeded via aria2c")
                return True
            logger.warning(f"aria2c failed: {result.stderr}")
            return False
        except Exception as e:
            logger.warning(f"aria2c download failed: {str(e)}")
            return False

    def download_model(self, model_name: str, quant: str = "Q4_K_M") -> str:
        if model_name not in MODEL_REPOS:
            raise ValueError(f"Unsupported model: {model_name}. Supported: {list(MODEL_REPOS.keys())}")

        if quant not in MODEL_PATTERNS[model_name]:
            raise ValueError(f"Unsupported quantization: {quant}. Supported: {list(MODEL_PATTERNS[model_name].keys())}")

        repo_id = MODEL_REPOS[model_name]
        pattern = MODEL_PATTERNS[model_name][quant]
        expected_filename = MODEL_FILENAMES[model_name][quant]
        save_path = os.path.join(self.models_dir, expected_filename)

        if os.path.exists(save_path):
            logger.info(f"Model already exists: {save_path}")
            return save_path

        logger.info(f"Downloading model: {repo_id}")
        logger.info(f"Pattern: {pattern}")
        logger.info(f"Saving to: {save_path}")

        if self._download_with_huggingface_hub(repo_id, pattern, save_path):
            return save_path

        mirrors = [
            "https://hf-mirror.com",
            "https://huggingface.co",
        ]
        for mirror in mirrors:
            url = f"{mirror}/{repo_id}/resolve/main/{expected_filename}"
            if self._download_with_aria2c(url, save_path):
                return save_path
            if self._download_with_curl(url, save_path):
                return save_path

        raise RuntimeError(
            f"Failed to download model from all sources. "
            f"Please try downloading manually and place it at: {save_path}\n"
            f"Download URL: https://huggingface.co/{repo_id}/resolve/main/{expected_filename}"
        )

    def list_models(self) -> list:
        models = []
        for f in os.listdir(self.models_dir):
            if f.endswith(".gguf"):
                models.append(f)
        return models

    def get_model_path(self, model_name: str, quant: str = "Q4_K_M") -> Optional[str]:
        if model_name not in MODEL_FILENAMES:
            return None

        if quant not in MODEL_FILENAMES[model_name]:
            return None

        filename = MODEL_FILENAMES[model_name][quant]
        save_path = os.path.join(self.models_dir, filename)

        if os.path.exists(save_path):
            return save_path
        return None
