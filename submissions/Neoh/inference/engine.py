import logging
from typing import Optional, List, Dict, Any
from llama_cpp import Llama
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class InferenceConfig(BaseModel):
    model_path: str = Field(..., description="模型文件路径")
    n_gpu_layers: int = Field(-1, description="加载到GPU的层数，-1表示全部")
    n_ctx: int = Field(8192, description="上下文窗口大小")
    n_batch: int = Field(512, description="批处理大小")
    temperature: float = Field(0.7, description="温度参数")
    max_tokens: int = Field(4096, description="最大生成token数")
    chat_format: str = Field("qwen2", description="对话格式：qwen2, llama, chatml")


class InferenceEngine:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.llm: Optional[Llama] = None
        self._init_llm()

    def _init_llm(self):
        try:
            logger.info(f"Loading model from {self.config.model_path}")
            logger.info(f"GPU layers: {self.config.n_gpu_layers}, Context: {self.config.n_ctx}")
            
            self.llm = Llama(
                model_path=self.config.model_path,
                n_gpu_layers=self.config.n_gpu_layers,
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                verbose=True,
                n_threads=8,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.llm:
            raise RuntimeError("LLM engine not initialized")

        stop_tokens = self._get_stop_tokens()

        try:
            response = self.llm(
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                stop=stop_tokens,
                echo=False,
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            raise

    def _get_stop_tokens(self) -> List[str]:
        format = self.config.chat_format.lower()
        
        if format == "qwen2":
            return ["<|im_end|>", "</s>"]
        elif format == "llama":
            return ["</s>", "[INST]", "[/INST]"]
        elif format == "chatml":
            return ["<|im_end|>", "</s>"]
        else:
            return ["<|im_end|>", "</s>"]

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.llm:
            raise RuntimeError("LLM engine not initialized")

        prompt = self._build_chat_prompt(messages)
        return self.generate(prompt, **kwargs)

    def _build_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        format = self.config.chat_format.lower()
        
        if format == "qwen2":
            return self._build_qwen2_prompt(messages)
        elif format == "llama":
            return self._build_llama_prompt(messages)
        elif format == "chatml":
            return self._build_chatml_prompt(messages)
        else:
            return self._build_qwen2_prompt(messages)

    def _build_qwen2_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        prompt_parts.append("<|im_start|>assistant\n")
        return "".join(prompt_parts)

    def _build_llama_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<<SYS>>\n{content}\n<</SYS>>")
            elif role == "user":
                prompt_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                prompt_parts.append(content)
        
        return "\n".join(prompt_parts)

    def _build_chatml_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        prompt_parts.append("<|im_start|>assistant\n")
        return "".join(prompt_parts)

    def get_model_info(self) -> Dict[str, Any]:
        if not self.llm:
            return {"status": "not_initialized"}
        
        return {
            "status": "loaded",
            "model_path": self.config.model_path,
            "n_ctx": self.config.n_ctx,
            "n_gpu_layers": self.config.n_gpu_layers,
            "temperature": self.config.temperature,
        }

    def unload(self):
        if self.llm:
            del self.llm
            self.llm = None
            logger.info("Model unloaded")