"""Worker 包（SPEC 附录 A）：Celery 应用与任务定义。

阶段 0 只包含队列接线与占位任务；model_analyzer / quantization / tensorrt /
codegen / build / packaging 等执行模块按阶段 1 起逐步加入。
"""
