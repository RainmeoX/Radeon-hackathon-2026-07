# Radeon-Assistant - Implementation Plan

## [x] Task 1: 修复 inference/engine.py 对话格式
- **Priority**: high
- **Depends On**: None
- **Description**: 
  - 将 _build_chat_prompt 方法从 Llama/Mistral 格式（<<SYS>>, [INST]）改为 Qwen2.5 格式（<|im_start|>/<|im_end|>）
  - 添加 chat_format 参数支持，让 llama-cpp-python 自动处理对话格式
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-1.1: InferenceEngine 能正确解析 chat_completion 的 messages 参数
  - `programmatic` TR-1.2: 生成的 prompt 使用 <|im_start|>/<|im_end|> 格式
  - `human-judgment` TR-1.3: 模型输出响应自然、格式正确

## [x] Task 2: 修复 model_loader.py 文件名逻辑

## [x] Task 4: 创建目录骨架和 __init__.py
- **Priority**: high
- **Depends On**: None
- **Description**: 
  - 修复 get_model_path 方法的文件名拼接逻辑（'qwen2.5-7b' 经过 replace('-','') 变成 'qwen2.57b' 的 bug）
  - 统一下载URL、保存文件名、config引用三者一致
  - 使用精确文件名而非字符串拼接
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-2.1: get_model_path 能正确返回已下载模型的路径
  - `programmatic` TR-2.2: 下载的文件名与 config.yaml 中的 model.path 一致

## [x] Task 3: 创建 scripts/download_model.py 脚本
- **Priority**: high
- **Depends On**: Task 2
- **Description**: 
  - 创建模型下载脚本，调用 ModelLoader 下载 Qwen2.5-7B-Instruct-Q4_K_M 模型
  - 添加命令行参数支持（模型名称、量化类型）
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-3.1: 执行脚本后模型文件存在于 ./models/ 目录
  - `programmatic` TR-3.2: 脚本支持 --help 参数显示使用说明

## [/] Task 4: 创建目录骨架和 __init__.py
- **Priority**: high
- **Depends On**: None
- **Description**: 
  - 创建 agent/、tools/、memory/、ui/、scripts/ 目录
  - 在每个目录下创建 __init__.py 文件
- **Acceptance Criteria Addressed**: AC-1, AC-3, AC-4, AC-5, AC-6, AC-7
- **Test Requirements**:
  - `programmatic` TR-4.1: 所有目录和 __init__.py 文件存在
  - `programmatic` TR-4.2: 各模块可被正确导入

## [ ] Task 5: 实现 memory 模块（FAISS 向量存储）
- **Priority**: high
- **Depends On**: Task 4
- **Description**: 
  - 实现 memory/vector_store.py - FAISS 向量数据库封装
  - 实现 memory/manager.py - 记忆管理器
  - 实现文档解析器（支持 PDF/MD/TXT）
- **Acceptance Criteria Addressed**: AC-3, AC-4
- **Test Requirements**:
  - `programmatic` TR-5.1: 文档能被正确解析和分块
  - `programmatic` TR-5.2: 文档向量能被正确存入 FAISS 索引
  - `programmatic` TR-5.3: 相似度检索能返回 Top-K 结果

## [ ] Task 6: 创建 scripts/init_rag.py 脚本
- **Priority**: high
- **Depends On**: Task 5
- **Description**: 
  - 创建 RAG 初始化脚本，处理文档向量化和索引构建
  - 支持批量导入目录下的所有文档
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-6.1: 执行脚本后 FAISS 索引文件存在于 ./data/faiss_index/
  - `programmatic` TR-6.2: 脚本能正确处理多个文档

## [ ] Task 7: 实现 tools 模块
- **Priority**: high
- **Depends On**: Task 4
- **Description**: 
  - 实现 tools/registry.py - 工具注册中心
  - 实现 tools/file_tools.py - 文件操作工具
  - 实现 tools/shell_tools.py - 命令执行工具
  - 实现 tools/code_tools.py - 代码解释工具
  - 实现 tools/system_tools.py - 系统信息工具
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `programmatic` TR-7.1: 工具能被正确注册和发现
  - `programmatic` TR-7.2: 文件操作工具能读写文件
  - `programmatic` TR-7.3: 命令执行工具能执行简单命令
  - `programmatic` TR-7.4: 系统信息工具能获取 CPU/GPU/内存信息

## [ ] Task 8: 实现 agent 模块
- **Priority**: high
- **Depends On**: Task 4, Task 7
- **Description**: 
  - 实现 agent/planner.py - 任务规划器
  - 实现 agent/executor.py - 任务执行器
  - 实现 agent/reflector.py - 结果反思器
  - 实现 agent/core.py - Agent 主循环（基于 langgraph）
- **Acceptance Criteria Addressed**: AC-5, AC-6
- **Test Requirements**:
  - `programmatic` TR-8.1: Planner 能将复杂任务分解为步骤列表
  - `programmatic` TR-8.2: Executor 能正确调用工具执行步骤
  - `programmatic` TR-8.3: Reflector 能评估执行结果
  - `human-judgment` TR-8.4: Agent 能完成多步骤任务

## [ ] Task 9: 实现 ui/web_app.py Streamlit 界面
- **Priority**: medium
- **Depends On**: Task 4, Task 8
- **Description**: 
  - 创建 Streamlit Web 应用，包含聊天界面和文档上传功能
  - 集成 Agent 核心模块，支持交互式对话
- **Acceptance Criteria Addressed**: AC-7
- **Test Requirements**:
  - `human-judgment` TR-9.1: 界面美观、响应迅速
  - `human-judgment` TR-9.2: 能发送消息并接收响应
  - `human-judgment` TR-9.3: 能上传文档并进行 RAG 问答

## [ ] Task 10: 创建 app.py 应用入口
- **Priority**: medium
- **Depends On**: Task 9
- **Description**: 
  - 创建应用入口文件，初始化配置、推理引擎和 Agent
  - 启动 Streamlit 服务
- **Acceptance Criteria Addressed**: AC-1, AC-7
- **Test Requirements**:
  - `programmatic` TR-10.1: 应用能正常启动
  - `programmatic` TR-10.2: 能加载配置文件并初始化组件

## [ ] Task 11: 修复 startup.ipynb 实例化问题
- **Priority**: high
- **Depends On**: Task 1
- **Description**: 
  - 修复 InferenceEngine() 无参数实例化问题，改为从 config.yaml 读取配置
  - 更新测试 Agent 的代码示例
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `human-judgment` TR-11.1: Notebook 中的代码能正常执行
  - `programmatic` TR-11.2: InferenceEngine 能从配置文件正确初始化

## [x] Task 12: 更新 requirements.txt 添加 ROCm 编译说明

## [x] Task 11: 修复 startup.ipynb 实例化问题

## [x] Task 5: 实现 memory 模块（FAISS 向量存储）

## [x] Task 6: 创建 scripts/init_rag.py 脚本

## [x] Task 7: 实现 tools 模块

## [x] Task 8: 实现 agent 模块

## [x] Task 9: 实现 ui/web_app.py Streamlit 界面

## [x] Task 10: 创建 app.py 应用入口
- **Priority**: high
- **Depends On**: None
- **Description**: 
  - 添加 llama-cpp-python 的 ROCm 后端编译说明
  - 可能需要添加安装脚本或注释说明 CMAKE_ARGS 环境变量设置
- **Acceptance Criteria Addressed**: AC-8
- **Test Requirements**:
  - `human-judgment` TR-12.1: 文档清晰说明 ROCm 编译步骤
  - `programmatic` TR-12.2: 安装后 llama-cpp-python 支持 HIP 后端