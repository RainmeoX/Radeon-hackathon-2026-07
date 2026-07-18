import logging
import subprocess
from typing import Dict, Any
from .registry import ToolDefinition, registry

logger = logging.getLogger(__name__)


def execute_command(command: str, timeout: int = 60) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"命令执行超时 ({timeout}秒): {command}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_python(code: str) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Python 代码执行超时"}
    except Exception as e:
        return {"success": False, "error": str(e)}


registry.register_tool(ToolDefinition(
    name="execute_command",
    description="执行系统命令",
    parameters={
        "command": {"type": "string", "description": "要执行的命令"},
        "timeout": {"type": "integer", "description": "超时时间（秒）"},
    },
    function=execute_command,
    requires_approval=True,
))

registry.register_tool(ToolDefinition(
    name="execute_python",
    description="执行 Python 代码",
    parameters={
        "code": {"type": "string", "description": "要执行的 Python 代码"},
    },
    function=execute_python,
    requires_approval=True,
))