# 阶段 2 交付说明：V1.0（稳定性和质量）

SPEC 17 对阶段 2 的要求：**分割、误差分析、算子检测、数据集管理、精度验证**。

推进顺序（已与用户确认）：

| 批次 | 内容 | 状态 |
| --- | --- | --- |
| **2A** | ① 算子兼容性报告 ② 分层误差分析 + 精度报告页 ③ 数据集管理 | **全部完成** |
| **2C** | INT8 改用 Q/DQ 显式量化（阶段 1 遗留项：当前 EntropyCalibrator2 已 deprecated） | **完成**（实测精度优于旧路径，已设为默认） |
| **2B** | 语义分割全链路（Adapter + mask 后处理 + C++ 模板 + mIoU 指标 + 合成分割模型回归；联网允许时再拉一个公开小模型） | 🚧 进行中：后处理/指标/合成分割模型/**Model Adapter** 已完成；C++ 模板与流水线接线待做 |

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

---

## 2. 分层误差分析 + 精度报告页（SPEC 9.2）— 完成

### 2.1 交付内容

| 模块 | 位置 |
| --- | --- |
| 分层误差分析（FP32 基线 vs INT8，逐层指标 + 敏感层排序） | `backend/app/services/layer_error_analysis.py` |
| 接入 TESTING 阶段（纯 CPU，失败只标 BLOCKED 不阻断任务） | `backend/app/pipeline/stages/build.py::_layer_error_analysis` |
| 产物报告 | `report/layer_error_analysis.json`（另在 `runtime_verification.json` 内引用） |
| 查询接口 | `GET /api/tasks/{id}/precision`（端到端精度 + 分层分析；未到 TESTING 阶段返回 501） |
| 精度报告页（替换阶段 1 的占位页） | `frontend/src/views/PrecisionReportView.vue` |

### 2.2 做法与边界（写进报告，不含糊）

1. **instrument**：把选定节点的输出临时追加为图输出，其余结构/权重完全不变；
2. 用 onnxruntime 跑 **FP32 基线**；
3. 用 onnxruntime 对**同一张图**做 **INT8 静态量化**（QDQ / QInt8 / MinMax，校准数据就是本任务
   实际使用的校准集），再跑一次；
4. 逐张量算 MAE / MSE / RMSE / 最大绝对误差 / Cosine，并按**相对 RMSE**（RMSE ÷ 基线绝对峰值）
   排序——跨层比较必须用相对量，否则量纲大的层永远排第一；
5. 多个校准样本**先各自算指标再平均**，避免把不同样本拼成一个大张量掩盖单样本退化；
6. 选取规则：静态 float 张量、元素数在 [64, 4e6] 之间，超过上限时**按图顺序均匀抽样**
   （不是只取前 N 个，避免只看前半张图）；默认最多 120 层、最多 4 个输入样本。

**边界（重要）**：分层分析在 ONNX/onnxruntime 层面进行，用于**定位敏感层**；它**不是** TensorRT
Engine 的逐层数据（TRT 运行期不暴露中间张量，要拿就得在 build 阶段 markOutput 重建引擎，代价与
风险都不合适）。**端到端精度**仍以真实 Engine 与 FP32 基准的对比为准（`report/accuracy.json`），
两者不一致时以端到端为准。

默认策略：`build.layer_analysis=auto` 时**只对 INT8 任务**执行（SPEC 9.2 分析的是量化误差）；
其他精度需要时设 `build.layer_analysis=on`，关闭用 `skip`。

### 2.3 验证证据

纯 CPU 用例 10 项（`tests/test_layer_error_analysis.py`）：指标精确值、形状不一致报错、
**相对 RMSE 才能跨层比较**、候选层筛选上限、instrument 不改变原图（输出与算子数不变）、
无输入样本 → SKIPPED、**量化失败降级为 BLOCKED 且说明不影响任务结论**、真实 ONNX 全流程
（分层指标齐全 + 排序单调递减 + ranking 上限）、`/precision` 路由返回两份报告、缺报告 501。
全量套件 **207 项通过**。

真实端到端（真实 yolov8n + **INT8** + 4 张校准图 + 真实 TensorRT 熵校准 Engine，任务 75s SUCCESS）：

```text
端到端（真实 INT8 Engine vs FP32 基准）：MAE 0.671977｜RMSE 6.173026｜余弦 0.994838｜最大绝对误差 316.53
分层分析：status=SUCCESS｜分析层数 120/120｜输入样本 4
敏感层排序（相对 RMSE 降序）：
  #1 Sigmoid  relRMSE=0.12643  RMSE=0.12643  余弦=0.98261  元素=32000
  #2 Sigmoid  relRMSE=0.10248  RMSE=0.10248  余弦=0.98786  元素=128000
  #3 Sigmoid  relRMSE=0.08075  RMSE=0.08075  余弦=0.99275  元素=512000
  #4 Sigmoid  relRMSE=0.07429  RMSE=0.07429  余弦=0.99461  元素=409600
  #5 Split    relRMSE=0.05177  RMSE=0.10628  余弦=0.95407  元素=51200
```

结论合理地指向 **Sigmoid 激活层**（YOLOv8 的激活层是 INT8 混合精度的经典敏感点）——这与后续
2C（Q/DQ 显式量化）和混合精度策略的预期一致。

浏览器验证（Playwright + Edge，`/report` 页）：端到端区块显示 MAE/RMSE/余弦真实数值（无占位符）、
分层区块显示「状态：SUCCESS 分析层数：120/120 输入样本：4」、敏感层排序表含相对 RMSE 进度条与
Sigmoid 行、可展开全部层明细，**无控制台错误**。

### 2.4 已知限制

- 只分析**静态 float** 中间张量（动态形状张量无法逐层比对）；超限时均匀抽样而非全量。
- 分层分析基于 ORT 的 INT8 量化近似（QDQ/MinMax），与 TensorRT 的熵校准实现不完全等价：
  用于**排序找敏感层**是可靠的，但**不要**把它当作 TRT 的逐层数值。
- 尚未实现 SPEC 9.2 末尾的「混合精度策略 / 自动保留敏感层 FP16」——列为 2C 之后的候选。

---

## 3. 数据集管理（SPEC 8.2 / 8.3）— 完成

### 3.1 交付内容

| 模块 | 位置 |
| --- | --- |
| 数据集管理服务（清单/统计、抽样、重新校验、批量删除） | `backend/app/services/dataset_admin.py` |
| API：清单与统计、样本预览、重新校验、批量删除 | `backend/app/api/routes/datasets.py` |
| 界面：数据集管理页（列表 + 详情抽屉 + 批量删除） | `frontend/src/views/DatasetsView.vue`（路由 `/datasets`，侧边栏「数据集管理」） |

接口一览：

| 接口 | 说明 |
| --- | --- |
| `GET /api/datasets/{id}/inventory` | 分页图像清单（尺寸/通道/格式/可读性）+ 分辨率/通道/格式分布 + 异常图像列表 |
| `GET /api/datasets/{id}/samples?path=` | 读取数据集内一张样本图（供界面预览）；路径受约束，越界返回 400 |
| `POST /api/datasets/{id}/revalidate?model_id=` | 重新执行 SPEC 8.3 校验；带 `model_id` 时额外做抽样预处理检查 |
| `POST /api/datasets/delete` | 批量删除（逐个执行、逐个提交，返回每项结果） |

### 3.2 设计决策

1. **不新增数据库迁移**：校验结果写进已有的 `datasets.meta["validation"]`（该字段本来就是
   为 SPEC 8.3 预留的），避免为一份可重算的报告改表结构。
2. **统计有上限并在界面上说明**：默认只扫描前 300 张（大校准集不能把请求拖死），
   返回 `scanned` / `truncated`，界面显示「已截断，统计基于前 N 张」——**不假装是全量统计**。
3. **没有模型也能校验**：`model_id` 可选。不传时只做图像级检查（数量/分辨率/通道/格式/可读性），
   并明确写一条警告说明「抽样预处理检查已跳过」；传模型时才做 SPEC 8.3 第 3 项
   （tensor shape/dtype/range）。
4. **删除与项目删除同一策略**：正在被排队/运行中任务使用的数据集**拒绝删除**（409，说明原因），
   其余照删；删除时把其他任务上的 `dataset_id` 显式置空，避免悬空引用。
5. **预览也走安全路径**：`samples` 接口用 `safe_join` 逐段校验，`../` 之类一律拒绝（用例覆盖）。

### 3.3 验证证据

纯 CPU 用例 9 项（`tests/test_dataset_management.py`）：清单统计与异常图像、分页与扫描上限、
样本接口返回真实图片字节、**三种路径穿越尝试全部被拒**、无模型重新校验写回 meta 并给警告、
带模型重新校验执行抽样预处理检查、未知模型 404、批量删除（成功 + 不存在）、
被运行中任务占用的数据集拒绝删除且不影响其余。全量套件 **216 项通过**。

真实端到端（上传含 1 张坏图的 5 张图像校准集）：

```text
image_count=5；分辨率分布 {64x48: 2, 96x72: 2}；通道 {3: 4}；异常图像 1 张（UnidentifiedImageError）
样本预览：200 image/png 9332 B；路径穿越 "../../../qforge.db" → 400
重新校验（无模型）：image_count=5 异常=1，警告说明跳过预处理检查
重新校验（带真实 yolov8n）：抽样 4 张，tensor_shape [3,640,640]，dtype/shape 检查通过
批量删除：1 成功 / 1 不存在，释放 121468 B
```

浏览器验证（Playwright + Edge）：列表与校验状态标签、详情抽屉的统计分布与「异常图像（1）」、
**抽样预览图真实加载**（naturalWidth=64）、点「重新校验」出现成功提示、勾选后出现
「删除选中（1）」并能取消，**无控制台错误**。

### 3.4 已知限制

- 统计按前 N 张扫描（默认 300），不是全量精确统计（界面已明示）。
- 尚未提供"数据集图像删除/替换"这类细粒度编辑（当前只有整包上传与整体删除）。

---

## 4. INT8 显式量化（Q/DQ）— 完成

### 4.1 交付内容

| 模块 | 位置 |
| --- | --- |
| Q/DQ 量化器（生成带 QuantizeLinear / DequantizeLinear 的 ONNX） | `backend/app/services/qdq_quantization.py` |
| QUANTIZING 阶段：生成 Q/DQ 图并写 `report/quantization.json` | `backend/app/pipeline/stages/prepare.py` |
| BUILDING_ENGINE：解析 Q/DQ 图并**不设置校准回调** | `backend/app/adapters/backends/tensorrt_adapter.py`（`quantized_onnx_path`） |
| 配置入口 | 任务 `quantization: {"mode": "qdq"|"calibrator", "method": "minmax|entropy|percentile", "per_channel": bool}` |

默认 `mode=qdq`；`mode=calibrator` 保留旧熵校准路径用于对比。量化失败抛
`QuantizationFailedError` 并写明原因，**不静默回退**（否则报告失真）。

### 4.2 让 TensorRT 接受 Q/DQ 图的两个硬要求（实测踩出来的）

1. **激活必须对称量化**（zero point = 0）→ `ActivationSymmetric=True`。
   否则 TensorRT 报 `Non-zero zero point is not supported`（非对称量化只在 DLA 上支持）。
2. **不要量化 bias** → `QuantizeBias=False`。
   否则 onnxruntime 会为 Conv bias 生成 `DequantizeLinear`，TensorRT 10.16 的 ONNX Parser
   报 `INVALID_NODE` 直接拒绝。

这两条已写进代码注释并用回归用例盯住（`test_quantize_passes_tensorrt_compatible_options`），
避免以后被"顺手"改掉。

### 4.3 验证证据

纯 CPU 用例 11 项（`tests/test_qdq_quantization.py`）：选图逻辑（含回退分支）、Q/DQ 图确实含
QuantizeLinear/DequantizeLinear、无样本/未知校准方法/量化器抛异常都返回 BLOCKED、
校准方法白名单与映射、per_channel 记录、Adapter 接口向后兼容（默认 None）、
**TensorRT 兼容选项必须被传下去**。全量套件 **226 项通过**。

真实端到端对比（真实 yolov8n + 8 张校准图 + 真实 TensorRT 构建，两次任务都在同一台机器）：

| 路径 | 任务 | MAE | RMSE | 余弦相似度 | 最大绝对误差 |
| --- | --- | --- | --- | --- | --- |
| **Q/DQ 显式量化（新默认）** | SUCCESS 52s | **0.254139** | **1.888112** | **0.999512** | **165.002** |
| IInt8EntropyCalibrator2（旧） | SUCCESS 62s | 0.562972 | 5.197862 | 0.996344 | 301.079 |

Q/DQ 的 MAE 约为旧路径的 **1/2.2**、RMSE 约为 **1/3**、余弦相似度更高——这是把默认值切到
Q/DQ 的依据。`report/quantization.json` 记录了 `mode=qdq`、校准方法、样本数、Q/DQ 节点统计
（本机为 245 个 QuantizeLinear / 309 个 DequantizeLinear，总节点 816）与产物大小，
可证明量化确实发生。

### 4.4 已知限制与后续

- QDQ 图由 onnxruntime 生成，激活尺度算法与 TensorRT 熵校准不同，两者数值不保证可复现一致；
  换模型/换校准集后建议两条路径都跑一次做对比（报告里都有数据）。
- 校准方法默认 `minmax`；`entropy` / `percentile` 也可用（用例覆盖），但尚未做质量对比。
- **混合精度策略（自动把敏感层保留 FP16）**尚未实现——第 2 节的分层误差排序已经给出了
  "该保留哪些层"的依据，可作为下一步（需 TensorRT 逐层精度设置或 Q/DQ 选择性插入）。

---

## 5. 语义分割（2B，进行中）

SPEC 2.2 把「语义分割」列为 V1.0 扩展。检测与分割的后处理语义完全不同（检测出框 + NMS，
分割逐像素 argmax 出掩膜），因此分三步做，**已完成前三项**：

| 项 | 位置 | 状态 |
| --- | --- | --- |
| 掩膜后处理 + mIoU/IoU/Dice/像素准确率 | `backend/app/services/segmentation.py` | 完成 |
| 合成分割模型 + 可命令行再生的回归资产（模型 + 真值掩膜） | `tools/synth_segmentation.py` | 完成 |
| 分割 Model Adapter（`unet` / `segmentation`） | `backend/app/adapters/models/unet.py` | 完成 |
| 分割版 C++ 工程模板（输出掩膜 + 统计，保留实时/推送） | `backend/app/codegen/templates/task/segmentation/` | 完成（掩膜 PPM 输出 + JSON + 推送；实时摄像头循环未接） |
| 流水线接线（分割任务的运行验证/精度报告写 mIoU） | `backend/app/pipeline/stages/` | 待做 |
| 端到端回归（生成 → 编译 → 推理 → mIoU 报告） | `tests/`（GPU 标记） | 待做 |

### 5.1 已完成部分的关键决策

1. **不猜后处理语义**：分割的 `postprocessing.decoder` 必须是 `argmax`（或架构名 `unet`），
   由适配器显式声明，而不是从 ONNX shape 反推（AGENTS.md 明确禁止用 shape 猜后处理）。
   检测的 `decoder=yolov8` 用在分割任务上会被**拒绝**（有用例）。
2. **指标按整批累积混淆矩阵计算**（不是每张图算完取平均），并且 `ignore_index`（默认 255）
   在分子分母里**完全排除**；真值与预测都没出现的类别不参与 mIoU 平均，
   但"被预测到却不在真值里"的类别必须计入（IoU=0），否则假正例会凭空消失。
3. **输入不是 4D 直接报错**（不猜也不静默算错）；标签缩放用**最近邻**（标签不能插值）。
4. **接口向后兼容**：分割的掩膜统计放在 `DecodeResult.extra`，`summary()` 一并输出；
   检测路径的 `detections` / `candidates` / `preview` 字段保持不变（有用例盯住）。
5. **任务类型白名单**放行 `segmentation`（`app/models/task.py` 的 `TASK_TYPES`）。
   这使既有的"不支持的任务类型"用例失效，已把该用例改用一个确实不支持的值（`classification`）——
   是能力变化导致的期望更新，不是放宽断言。

### 5.2 验证证据

纯 CPU 用例 **28 项**（`tests/test_segmentation.py` 15 项 + `tests/test_segmentation_adapter.py` 13 项）：
argmax 正确性与维度校验、最近邻不插值、**手算核对 IoU/Dice/准确率**、忽略像素完全排除、
空类不参与平均、混淆矩阵维度校验、合成分割模型 onnxruntime 端到端（满分 → mIoU=1；
故意错一半 → 准确率 0.5 且 0<mIoU<1）、同 seed 模型逐字节一致、回归资产可命令行再生；
适配器注册与默认定义、`validate` 拒绝错误任务/错误 decoder、通道数不一致给警告、
letterbox 给警告、`decode` 产出掩膜与类别直方图、`DecodeResult` 检测字段不回归、
**API 放行 `task_type=segmentation`**。全量套件通过。

### 5.3 已知限制（本批次）

- **分割任务的真实 Engine 构建与端到端回归尚未跑过**：流水线里仍有检测专用的后处理/报告分支，
  需要用合成分割模型跑通"生成 → 编译 → 推理 → mIoU 报告"后才能宣称 2B 完成。
- 尚未接 C++ 代码生成的分割模板（当前生成器只有检测模板）。
- 尚未用真实公开分割模型验证泛化性（联网允许时补）。
