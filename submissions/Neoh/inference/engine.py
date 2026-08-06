"""vLLM 推理引擎封装（ROCm / AMD Radeon）。

在 AMD Radeon GPU 上以 vLLM 运行 Qwen2.5（14B/7B/32B）。ROCm 环境变量
（HSA_OVERRIDE_GFX_VERSION 等）由 app.py / scripts/serve.py / ui/web_app.py
在导入本模块之前设置，本模块不再设置，避免与入口逻辑耦合。

API：
- generate(prompt, **kwargs) -> str         原始文本补全
- chat(messages, **kwargs) -> str           将 messages 拼成 "role: content\\n" 文本后补全
- chat_completion(messages, **kwargs) -> str  chat 的向后兼容别名
- get_model_info() -> dict / unload()       状态查询与卸载
"""

import logging
import os
from typing import Optional, List, Dict, Any

from vllm import LLM, SamplingParams
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class InferenceConfig(BaseModel):
    # 调用方统一使用 model_path 字段名（config.yaml 的 model.path 经此传入）
    model_path: str = Field(..., description="模型目录路径（safetensors 格式）")
    n_ctx: int = Field(8192, description="上下文窗口大小（max_model_len）")
    dtype: str = Field("float16", description="模型精度：float16/bfloat16/auto")
    gpu_memory_utilization: float = Field(0.9, description="GPU 显存利用率")
    tensor_parallel_size: int = Field(1, description="张量并行 GPU 数")
    pipeline_parallel_size: int = Field(1, description="流水线并行 GPU 数")
    temperature: float = Field(0.7, description="生成温度")
    max_tokens: int = Field(2048, description="最大生成 token 数")


class InferenceEngine:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop=["<|im_end|>", "</s>"],
        )
        self._load_model()

    def _load_model(self):
        # 规范化模型路径：相对路径转绝对路径，避免 vLLM / AutoTokenizer
        # 误判为 HF repo id 而尝试联网下载。
        model_path = self.config.model_path
        if model_path.startswith("./") or model_path.startswith("../"):
            model_path = os.path.abspath(model_path)
        logger.info(f"正在加载模型: {model_path}")
        logger.info(
            f"Context: {self.config.n_ctx}, dtype: {self.config.dtype}, "
            f"TP: {self.config.tensor_parallel_size}, PP: {self.config.pipeline_parallel_size}"
        )

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = LLM(
            trust_remote_code=True,
            model=model_path,
            dtype=self.config.dtype,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            tensor_parallel_size=self.config.tensor_parallel_size,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            max_model_len=self.config.n_ctx,
        )
        logger.info("✅ 模型加载完成")

    def _build_sampling_params(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> SamplingParams:
        if max_tokens is None and temperature is None:
            return self.sampling_params
        return SamplingParams(
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            stop=["<|im_end|>", "</s>"],
        )

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """文本生成（非 chat 格式）。"""
        if not self.model:
            raise RuntimeError("LLM engine not initialized")
        sampling_params = self._build_sampling_params(max_tokens, temperature)
        try:
            outputs = self.model.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Chat 对话补全。

        不使用 vLLM 原生 chat template，而是将 messages 拼接成纯文本 prompt：
            role: content
            role: content
            assistant:
        再调用 generate() 做原始补全。
        """
        prompt = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt += f"{role}: {content}\n"
        prompt += "assistant: "
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    # 向后兼容别名：旧调用方（agent/core.py、scripts/benchmark.py）使用 chat_completion
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "status": "loaded" if self.model else "not_initialized",
            "engine": "vLLM",
            "model_path": self.config.model_path,
            "n_ctx": self.config.n_ctx,
            "dtype": self.config.dtype,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "temperature": self.config.temperature,
        }

    def unload(self):
        if self.model:
            del self.model
            self.model = None
            logger.info("Model unloaded")
