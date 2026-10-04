# 阶段 2 交付说明：V1.0（稳定性和质量）

SPEC 17 对阶段 2 的要求：**分割、误差分析、算子检测、数据集管理、精度验证**。

推进顺序（已与用户确认）：

| 批次 | 内容 | 状态 |
| --- | --- | --- |
| **2A** | ① 算子兼容性报告 ② 分层误差分析 + 精度报告页 ③ 数据集管理 | ① 完成；②③ 进行中 |
| **2C** | INT8 改用 Q/DQ 显式量化（阶段 1 遗留项：当前 EntropyCalibrator2 已 deprecated） | 未开始 |
| **2B** | 语义分割全链路（Adapter + mask 后处理 + C++ 模板 + mIoU 指标 + 合成分割模型回归；联网允许时再拉一个公开小模型） | 未开始 |

---

## 1. 算子兼容性报告（SPEC 7.1 步骤 6 / 7.2）— 完成

### 1.1 交付内容

| 模块 | 位置 |
| --- | --- |
| TensorRT 算子能力注册表（**后端私有知识**） | `backend/app/adapters/backends/tensorrt_capabilities.py` |
| 通过 Backend Adapter 暴露（避免通用流水线里出现 `if backend == ...`） | `backend/app/adapters/base.py`（默认空报告）、`backends/tensorrt_adapter.py`（覆盖实现） |
| 写入产物报告 | `backend/app/pipeline/stages/prepare.py` → `report/compatibility.json` |
| 查询接口 | `GET /api/tasks/{id}/compatibility` |
| 界面展示（任务详情页） | `frontend/src/views/TaskDetailView.vue` |

报告字段严格按 SPEC 7.2：`operator / domain / opset / supported / condition / severity /
suggestion`，另加 `occurrences`（出现次数）与 `registered`（是否在能力表中登记），
并有汇总：`verdict(OK/WARNINGS/BLOCKED)`、错误数、警告数、算子种类/实例数、不支持与有条件算子名单。

### 1.2 设计决策（重要）

1. **只登记有依据的条目**：能力表收录 TensorRT 10.x 公开支持矩阵里的**已知限制与明确不可用项**
   （随机算子、ONNX 序列类型、字符串/文本域算子、`ai.onnx.ml` 域算子 → error；
   `NonMaxSuppression`/`TopK`/`Resize`/`GridSample`/`ScatterND`/`Einsum`/`NonZero`/`ArgMax`/
   `Loop`/`Scan`/`If`/`RoiAlign`/`Pad`/`Cast`/`LSTM` 等 → warning + 条件 + 建议）。
   写不出依据的算子**不登记**，避免用臆测结论告诉用户"不支持"。
2. **未登记算子的判定**：以真实 Parser 结果为准——解析通过 → info（视为支持）；
   解析失败 → error 并提示查阅对应版本的算子支持矩阵。绝不把"解析失败"报成"全部支持"。
3. **opset 门控**：能力表条目可带 `min_opset`/`max_opset`，不满足时自动升级为 error
   （例：`NonMaxSuppression` 在 opset 10 判为不可用）。
4. **结论只做建议**：最终判定仍以真实 Engine 构建与运行验证为准（阶段 1 已建立该链路）。

### 1.3 验证证据

纯 CPU 用例 8 项（`tests/test_capability_report.py`）：不支持算子判 error 且有建议、有条件算子判
warning 且写清条件、opset 门控把有条件算子升级为 error、未登记算子随 Parser 结果变化、
**SPEC 7.2 字段完整性**、注册表自洽性（不支持必给建议）、真实合成 ONNX 全量联动、
适配器暴露报告（无后端耦合）。全量套件 **197 项通过**。

真实模型端到端（真实 yolov8n + FP16 + 真实 TensorRT Parser）：

```text
结论 WARNINGS｜算子种类 17｜错误 0｜警告 1｜opset 11
  warning Resize   支持=True 出现=2  仅支持部分 coordinate_transformation_mode（如 half_pixel / align_corners / asymmetric）与 nearest / linear 模式
  info    Conv     支持=True 出现=64
  info    Mul      支持=True 出现=60
  ...
```

浏览器验证（Playwright + Edge，任务详情页）：结论栏显示「WARNINGS 错误 0 / 警告 1 算子种类 17
opset 11」、问题算子表含「限制条件 / 建议」列并列出 `Resize`、其余算子折叠展示，**无控制台错误**。

### 1.4 已知限制

- 能力表按 TensorRT 10.x 编写；升级 TensorRT 大版本时需复核（已在模块 docstring 注明）。
- 只覆盖**算子级**结论；同一算子在不同 shape/属性下仍可能构建失败，这类问题由真实构建日志
  （`report/engine_build.json` + `logs/`）负责暴露。
