# Radeon-Assistant - Product Requirement Document

## Overview
- **Summary**: 基于 AMD Radeon GPU 与 ROCm 软件栈构建的完全本地化 AI Agent 系统，支持 LLM 推理、RAG 文档问答、工具调用和多步骤任务规划。
- **Purpose**: 在本地 GPU 上实现私有 AI Agent，确保数据隐私保护和低延迟响应，替代云端 API 服务。
- **Target Users**: 需要在本地环境运行 AI Agent 的开发者、企业用户和隐私敏感场景用户。

## Goals
- 构建完整的本地 AI Agent 系统，所有推理计算在 Radeon GPU 上完成
- 实现 RAG 文档问答功能，支持私有文档上传和向量化检索
- 实现工具调用能力，包括文件操作、命令执行、代码解释等
- 实现多步骤任务规划与执行循环（Plan-and-Execute）
- 提供 Streamlit Web 界面，支持交互式对话和文档管理

## Non-Goals (Out of Scope)
- 模型微调训练（使用 RAG 替代）
- 邮件接收功能（仅支持发送，且为可选配置）
- 日历管理功能
- Web 搜索功能（离线优先设计）
- 多用户认证系统

## Background & Context
- 项目处于极早期阶段，仅完成了 inference 模块的基础框架
- 使用 Qwen2.5-7B-Instruct GGUF 量化模型进行本地推理
- 基于 ROCm + llama.cpp 实现 GPU 加速推理
- 项目规划文档 [PROJECT_PLAN.md](file:///C:/Users/26927/Desktop/Radeon-hackathon-2026-07/submissions/Neoh/PROJECT_PLAN.md) 定义了完整架构

## Functional Requirements
- **FR-1**: 核心推理引擎 - 支持 Qwen2.5-7B-Instruct 模型的加载和推理，使用正确的对话格式
- **FR-2**: 模型下载管理 - 支持从 HuggingFace 镜像下载 GGUF 模型，文件名与配置一致
- **FR-3**: RAG 系统 - 支持文档上传、解析、向量化、FAISS 存储和相似度检索
- **FR-4**: 工具系统 - 支持文件操作、命令执行、代码解释、系统信息查询等工具的注册和调用
- **FR-5**: Agent 核心编排 - 实现 Planner/Executor/Reflector/Reviser 的 Plan-and-Execute 循环
- **FR-6**: 记忆管理 - 支持短期对话记忆和长期向量记忆
- **FR-7**: Web UI - 提供 Streamlit 界面进行交互式对话和文档管理
- **FR-8**: ROCm 环境支持 - 正确配置 llama-cpp-python 的 HIP 后端编译参数

## Non-Functional Requirements
- **NFR-1**: 隐私保护 - 所有推理在本地执行，无外部 API 调用
- **NFR-2**: GPU 加速 - 首 Token 延迟 < 1.5s，生成速度 > 50 tokens/s
- **NFR-3**: 容错降级 - GPU 不可用时自动切换到 CPU 模式
- **NFR-4**: 安全审计 - 高危操作（删除文件、执行命令）需要用户确认

## Constraints
- **Technical**: ROCm 7.0+，Python 3.10+，llama-cpp-python 需 ROCm 编译
- **Business**: 黑客松时间限制，优先实现核心功能
- **Dependencies**: FAISS、sentence-transformers、langgraph、streamlit

## Assumptions
- 用户已安装 ROCm 7.0+ 和 AMD GPU 驱动
- 网络环境可访问 HuggingFace 镜像（hf-mirror.com）
- 用户有足够的磁盘空间存储模型文件（约 5GB）

## Acceptance Criteria

### AC-1: 模型加载与推理
- **Given**: 模型文件已下载到指定路径，ROCm 环境已配置
- **When**: 初始化 InferenceEngine 并调用 chat_completion
- **Then**: 成功加载模型到 GPU，返回正确格式的对话响应
- **Verification**: `programmatic`
- **Notes**: 使用 Qwen2.5 的 <|im_start|>/<|im_end|> 对话格式

### AC-2: 模型下载功能
- **Given**: 网络可访问 HuggingFace 镜像
- **When**: 执行 download_model.py 脚本
- **Then**: 模型文件下载到 ./models/ 目录，文件名与 config.yaml 配置一致
- **Verification**: `programmatic`

### AC-3: RAG 文档处理
- **Given**: 用户上传 PDF/MD/TXT 文档
- **When**: 执行文档解析、向量化、FAISS 存储流程
- **Then**: 文档被成功分块并存储到向量数据库
- **Verification**: `programmatic`

### AC-4: RAG 问答
- **Given**: RAG 系统已初始化，存在文档索引
- **When**: 用户提出与文档相关的问题
- **Then**: 返回基于文档内容的答案，并附带引用来源
- **Verification**: `human-judgment`

### AC-5: 工具调用
- **Given**: 工具已注册到工具中心
- **When**: 用户请求执行文件操作、命令或代码
- **Then**: Agent 正确识别并调用相应工具，返回执行结果
- **Verification**: `programmatic`

### AC-6: 多步骤任务规划
- **Given**: 用户提出复杂多步骤任务
- **When**: Agent 执行 Plan-and-Execute 循环
- **Then**: 任务被分解为步骤并依次执行，最终返回完成结果
- **Verification**: `human-judgment`

### AC-7: Web 界面交互
- **Given**: Streamlit 服务已启动
- **When**: 用户在浏览器中访问 http://localhost:7860
- **Then**: 显示聊天界面，支持发送消息和上传文档
- **Verification**: `human-judgment`

### AC-8: ROCm GPU 加速
- **Given**: Radeon GPU 和 ROCm 环境已配置
- **When**: 执行推理任务
- **Then**: GPU 利用率 > 70%，推理速度符合预期
- **Verification**: `programmatic`

## Open Questions
- [ ] 是否需要支持多个模型切换？
- [ ] 是否需要实现文档格式转换功能？
- [ ] 是否需要添加用户认证？