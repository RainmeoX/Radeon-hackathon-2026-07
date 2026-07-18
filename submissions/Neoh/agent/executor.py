import logging
from typing import Dict, Any
from tools.registry import registry

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self):
        pass

    def execute(self, step: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = step.get("tool")
        
        if tool_name is None or tool_name == "None" or tool_name == "null":
            return {
                "success": True,
                "output": f"执行步骤: {step.get('description', '')}",
                "step": step.get("step"),
            }
        
        try:
            arguments = step.get("arguments", {})
            result = registry.call_tool(tool_name, **arguments)
            
            return {
                "success": result.get("success", True),
                "output": result,
                "step": step.get("step"),
                "tool": tool_name,
            }
        except Exception as e:
            logger.error(f"Execution failed for step {step.get('step')}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "step": step.get("step"),
                "tool": tool_name,
            }