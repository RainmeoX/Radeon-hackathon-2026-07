import logging
from typing import List, Dict, Any
from inference.engine import InferenceEngine

logger = logging.getLogger(__name__)


class Planner:
    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def plan(self, task: str, tool_descriptions: str) -> List[Dict[str, Any]]:
        prompt = f"""你是一个任务规划专家。请将以下任务分解为可执行的步骤列表。

任务: {task}

可用工具:
{tool_descriptions}

请按照以下格式输出步骤列表（JSON格式）:
[
  {{"step": 1, "description": "步骤描述", "tool": "工具名称或None", "arguments": {{"参数名": "值"}}}},
  ...
]

注意:
- 如果不需要工具调用，tool 字段设为 null
- 参数值需要根据任务内容合理推断
- 步骤应该清晰、可执行
- 最多生成 10 个步骤
"""

        response = self.engine.generate(prompt, max_tokens=2000)
        
        try:
            import json
            steps = json.loads(response)
            if isinstance(steps, list):
                return steps
            return []
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse plan: {response}")
            return self._parse_fallback_plan(response)

    def _parse_fallback_plan(self, text: str) -> List[Dict[str, Any]]:
        steps = []
        lines = text.strip().split("\n")
        step_num = 1
        
        for line in lines:
            line = line.strip()
            if line.startswith(("-", "*", "1.", "2.", "3.")):
                description = line.lstrip("-*0123456789. ").strip()
                steps.append({
                    "step": step_num,
                    "description": description,
                    "tool": None,
                    "arguments": {},
                })
                step_num += 1
                if step_num > 10:
                    break
        
        return steps