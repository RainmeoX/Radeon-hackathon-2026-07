import logging
from typing import List, Dict, Any, Optional
from inference.engine import InferenceEngine
from memory.manager import MemoryManager
from tools.registry import registry
from .planner import Planner
from .executor import Executor
from .reflector import Reflector

logger = logging.getLogger(__name__)


class RadeonAgent:
    def __init__(
        self,
        engine: InferenceEngine,
        memory_manager: Optional[MemoryManager] = None,
        max_iterations: int = 10,
    ):
        self.engine = engine
        self.memory_manager = memory_manager
        self.max_iterations = max_iterations
        self.planner = Planner(engine)
        self.executor = Executor()
        self.reflector = Reflector(engine)

    def chat(self, message: str, use_rag: bool = True) -> str:
        context = ""
        
        if self.memory_manager and use_rag:
            rag_context = self.memory_manager.get_context(message)
            if rag_context:
                context = f"参考文档:\n{rag_context}\n\n"

        short_term_memory = self.memory_manager.get_short_term_memory() if self.memory_manager else []
        
        memory_text = ""
        for msg in short_term_memory[-5:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            memory_text += f"{role}: {content}\n"

        system_prompt = """你是一个基于 AMD Radeon GPU 的本地 AI 助手。请用中文回答用户的问题。

你的特点:
- 所有推理计算在本地 GPU 完成，数据隐私安全
- 支持 RAG 文档问答
- 支持工具调用和多步骤任务规划

请遵循以下原则:
1. 直接回答用户的问题，不需要解释思考过程
2. 如果问题涉及文档内容，请参考提供的参考文档
3. 如果需要执行多步骤任务，请使用工具调用
"""

        messages = [{"role": "system", "content": system_prompt}]
        
        if memory_text:
            messages.append({"role": "user", "content": f"对话历史:\n{memory_text}"})
        
        if context:
            messages.append({"role": "user", "content": f"{context}"})
        
        messages.append({"role": "user", "content": message})

        response = self.engine.chat_completion(messages)

        if self.memory_manager:
            self.memory_manager.add_short_term_memory({"role": "user", "content": message})
            self.memory_manager.add_short_term_memory({"role": "assistant", "content": response})

        return response

    def run_task(self, task: str) -> Dict[str, Any]:
        logger.info(f"Starting task: {task}")
        
        tool_descriptions = registry.get_tool_descriptions()
        steps = self.planner.plan(task, tool_descriptions)
        
        if not steps:
            return {"success": False, "error": "任务规划失败"}
        
        logger.info(f"Generated {len(steps)} steps")
        
        results = []
        for step in steps:
            logger.info(f"Executing step {step.get('step')}: {step.get('description')}")
            
            result = self.executor.execute(step)
            results.append(result)
            
            if not result.get("success", False):
                logger.warning(f"Step {step.get('step')} failed")
                break

        reflection = self.reflector.reflect(task, steps, results)

        final_result = {
            "success": reflection.get("completed", False),
            "task": task,
            "steps": steps,
            "results": results,
            "reflection": reflection,
        }

        if self.memory_manager:
            self.memory_manager.add_short_term_memory({
                "role": "user",
                "content": f"任务: {task}",
            })
            self.memory_manager.add_short_term_memory({
                "role": "assistant",
                "content": f"任务结果: {'完成' if final_result['success'] else '未完成'} - {reflection.get('reason', '')}",
            })

        return final_result

    def summarize_conversation(self) -> str:
        if not self.memory_manager:
            return "没有对话历史"

        memory = self.memory_manager.get_short_term_memory()
        if not memory:
            return "没有对话历史"

        messages_text = ""
        for msg in memory:
            role = msg.get("role", "")
            content = msg.get("content", "")
            messages_text += f"{role}: {content}\n"

        prompt = f"""请简要总结以下对话内容:

{messages_text}

总结要求:
- 用中文回答
- 不超过 200 字
- 涵盖主要问题和答案
"""

        return self.engine.generate(prompt, max_tokens=500)

    def clear_memory(self):
        if self.memory_manager:
            self.memory_manager.clear_short_term_memory()
            logger.info("Memory cleared")