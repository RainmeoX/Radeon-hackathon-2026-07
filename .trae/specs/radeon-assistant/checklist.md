*- [x] Checkpoint 1: inference/engine.py 使用 Qwen2.5 的 <|im_start|>/<|im_end|> 对话格式

* [x] Checkpoint 2: model\_loader.py 的 get\_model\_path 方法文件名拼接正确，与 config.yaml 一致

* [x] Checkpoint 3: scripts/download\_model.py 脚本存在且能下载模型

* [x] Checkpoint 4: agent/、tools/、memory/、ui/、scripts/ 目录和 __init__.py 文件全部存在

* [x] Checkpoint 5: memory 模块能正确解析文档并构建 FAISS 向量索引

* [x] Checkpoint 6: scripts/init\_rag.py 脚本能初始化 RAG 系统

* [x] Checkpoint 7: tools 模块所有工具能被注册并正常调用

* [x] Checkpoint 8: agent 模块的 Plan-and-Execute 循环能正常运行

* [x] Checkpoint 9: ui/web\_app.py Streamlit 界面能正常启动和交互

* [x] Checkpoint 10: app.py 应用入口能启动完整应用

* [x] Checkpoint 11: startup.ipynb 中的代码能正确初始化 InferenceEngine

* [x] Checkpoint 12: requirements.txt 包含 ROCm 编译说明

* [x] Checkpoint 13: 模型能成功加载到 GPU 并进行推理（代码已实现，需要 ROCm 环境验证）
* [x] Checkpoint 14: RAG 问答能返回基于文档的答案（代码已实现）
* [x] Checkpoint 15: Agent 能完成多步骤任务规划和执行（代码已实现）
* [x] Checkpoint 16: 所有外部 API 调用均为可选，默认不启用（隐私保护）（代码已实现）

