import logging
from typing import List, Dict, Any
from inference.engine import InferenceEngine

logger = logging.getLogger(__name__)


class Reflector:
    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def reflect(self, task: str, steps: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        results_text = ""
        for i, (step, result) in enumerate(zip(steps, results)):
            success = result.get("success", False)
            output = str(result.get("output", "无输出"))
            results_text += f"步骤 {i+1}: {step.get('description', '')}\n结果: {'成功' if success else '失败'}\n输出: {output}\n\n"

        prompt = f"""你是一个结果评估专家。请评估以下任务的执行结果是否完成。

任务: {task}

执行步骤和结果:
{results_text}

请按照以下格式输出评估结果（JSON格式）:
{{
  "completed": true/false,
  "reason": "完成或未完成的原因",
  "suggestion": "如果未完成，建议的下一步操作"
}}

注意:
- completed 为 true 表示任务已成功完成
- completed 为 false 表示任务未完成或需要进一步操作
- reason 需要简明扼要地说明评估依据
"""

        response = self.engine.generate(prompt, max_tokens=1000)
        
        try:
            import json
            result = json.loads(response)
            if isinstance(result, dict):
                return result
            return {"completed": False, "reason": "评估失败", "suggestion": "重新执行"}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse reflection: {response}")
            return self._parse_fallback_reflection(response)

    def _parse_fallback_reflection(self, text: str) -> Dict[str, Any]:
        if "完成" in text or "成功" in text:
            return {"completed": True, "reason": text, "suggestion": ""}
        else:
            return {"completed": False, "reason": text, "suggestion": "请重新执行任务"}