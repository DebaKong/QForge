# ONNX 自动化量化部署平台 — 软件需求与技术设计规格书

**版本：** V1.0

**用途：** 作为项目开发、模块分工、接口设计、测试验收的统一依据

**文档性质：** 在原《项目实施计划》基础上的工程化细化版本

---

## 1. 项目概述

### 1.1 项目定位

本项目定位为 Web 端一站式 ONNX 模型量化与部署平台。用户上传 ONNX 模型和校准数据集，选择任务类型、精度模式及目标后端/硬件后，平台自动完成模型校验、数据预处理、量化校准、后端编译、推理工程代码生成、自动构建与运行验证，并最终输出可部署的工程产物。

原计划的总体路线为：先聚焦 NVIDIA/TensorRT 生态完成目标检测 MVP，再扩展 2D 分割、3D 点云任务以及 OpenVINO 等后端。

### 1.2 本规格书解决的问题

- 明确系统模块边界，避免不同模块重复实现或互相耦合。
- 定义模型描述、任务配置、校准数据和最终产物的统一格式。
- 明确任务生命周期、日志、错误处理和资源限制。
- 明确自动生成 C++ 工程后必须经过编译和运行验证，避免只生成代码而无法部署。
- 为后续增加不同模型、不同任务和不同后端预留 Adapter/Template 扩展点。

---

## 2. 范围与版本边界

### 2.1 MVP 范围

| 项目     | MVP 定义                                          |
| -------- | ------------------------------------------------- |
| 任务类型 | 2D 目标检测                                       |
| 模型输入 | ONNX                                              |
| 精度模式 | FP32、FP16、INT8；其中 INT8 为静态校准量化        |
| 推理后端 | TensorRT                                          |
| 目标硬件 | NVIDIA GPU                                        |
| 校准数据 | 以图像文件为主，由平台执行统一预处理              |
| 代码产物 | C++ TensorRT 推理工程、CMakeLists.txt             |
| 容器产物 | Dockerfile、依赖说明；可选生成镜像                |
| 验证     | 模型校验、Engine 构建、C++ 编译、样例推理运行验证 |

### 2.2 非 MVP 内容

- 动态 Batch / Dynamic Shape：建议在 MVP 主链路稳定后加入。
- 语义分割：V1.0 扩展。
- 3D 点云检测：V1.5，建议先以单一 PointPillars 等模型做 PoC。
- OpenVINO：V2.0。
- RKNN：作为后续可选扩展，不作为 MVP 验收项。
- 在线推理演示、模型版本管理、批量任务、开放 API 等属于后续功能。

---

## 3. 总体系统架构

推荐系统分层如下：

```text
Web UI (Vue3 + Element Plus)
│
▼
FastAPI API Gateway
│
├── Project / Model / Dataset / Task API
├── WebSocket / Task Log
│
▼
Task Manager (Celery + Redis)
│
▼
Worker
├── Model Analyzer
├── Dataset / Calibration Loader
├── Quantization Engine
├── Backend Adapter (TensorRT)
├── Code Generator (Jinja2)
├── Build & Test
└── Artifact Packager
│
▼
File/Object Storage + Database
```

### 3.1 核心设计原则

- Web API 不直接执行长时间量化、编译和 Docker 构建任务。
- 所有耗时任务通过 Celery Worker 异步执行。
- 模型语义通过 Model Definition / Model Adapter 描述，而不是依赖 ONNX Shape 猜测后处理逻辑。
- 后端通过 Backend Adapter 隔离 TensorRT、OpenVINO 等具体实现。
- 代码生成采用“公共推理底座 + 任务模板 + 模型适配器 + 配置填充”的分层方式。
- 每个最终产物必须经过自动构建和最小运行验证。

---

## 4. 技术选型与运行环境

| 模块     | 技术                     | 要求/说明                                           |
| -------- | ------------------------ | --------------------------------------------------- |
| 前端     | Vue 3 + Element Plus     | 文件上传、任务配置、进度、日志、报告、产物下载      |
| 后端     | Python + FastAPI         | REST API；负责任务创建、查询和系统管理              |
| 任务调度 | Celery + Redis           | 异步量化、编译、测试和日志状态管理                  |
| 量化     | PPQ                      | MVP 支持静态 INT8；FP16 作为精度模式处理            |
| 模型解析 | ONNX / ONNX Runtime      | 模型合法性、图结构、输入输出信息检查                |
| TensorRT | TensorRT Python API      | ONNX → Engine                                       |
| 代码生成 | Jinja2                   | 生成 C++ / CMake / 配置文件                         |
| 容器     | Docker                   | 隔离构建和运行环境                                  |
| 数据库   | 关系型数据库             | 建议 PostgreSQL；保存任务、模型、数据集、产物元数据 |
| 文件存储 | 本地对象式目录或对象存储 | 保存模型、数据集、中间文件、日志和产物              |

### 4.1 版本矩阵必须固定

正式开发前必须在项目配置中锁定 Python、CUDA、TensorRT、ONNX、PPQ、OpenCV、CMake、GCC、Docker 等版本。文档不能只写“TensorRT 8.x”，应记录具体版本。每个 Engine 必须保存构建环境元数据。

```yaml
engine_metadata:
  backend: tensorrt
  backend_version: "<具体版本>"
  cuda_version: "<具体版本>"
  gpu_arch: "<SM版本>"
  precision: "FP16 | INT8 | FP32"
  input_profile: "<shape/profile>"
  build_time: "<timestamp>"
```

---

## 5. 核心数据模型与统一任务配置

### 5.1 TaskConfig

所有前端、后端、Worker 和代码生成模块围绕统一 TaskConfig 工作。推荐最小结构如下：

```json
{
  "task_id": "string",
  "task_type": "detection",
  "model": {
    "path": "string",
    "architecture": "yolov8"
  },
  "backend": {
    "name": "tensorrt",
    "version": "string"
  },
  "precision": "fp16",
  "preprocess": {},
  "postprocess": {},
  "calibration": {},
  "build": {},
  "validation": {}
}
```

### 5.2 数据实体

| 实体       | 职责                                 |
| ---------- | ------------------------------------ |
| Project    | 项目级组织单元                       |
| Model      | 原始 ONNX 模型及模型元数据           |
| Dataset    | 校准集/测试集及其元数据              |
| Task       | 一次完整部署流水线执行记录           |
| TaskConfig | 任务执行所需的完整配置快照           |
| Artifact   | Engine、源码、Dockerfile、报告等产物 |
| JobLog     | 阶段日志、错误日志和编译日志         |
| Backend    | 后端及版本、能力信息                 |

---

## 6. Model Definition / Model Adapter 规范

这是自动生成正确推理代码的关键模块。ONNX 图结构只能提供张量和算子信息，不能可靠表达完整的业务后处理语义。因此平台必须显式保存模型定义。

### 6.1 模型定义内容

```yaml
model:
  task: detection
  architecture: yolov8

  input:
    name: images
    shape: [1, 3, 640, 640]
    layout: NCHW
    dtype: FP32

  preprocessing:
    resize: letterbox
    color: RGB
    scale: 255.0
    mean: [0, 0, 0]
    std: [1, 1, 1]

  output:
    name: output0
    format: "[1,84,8400]"
    box_format: xywh
    has_objectness: false
    class_count: 80

  postprocessing:
    decoder: yolov8
    nms:
      type: classwise
      confidence_threshold: 0.25
      iou_threshold: 0.45
```

### 6.2 Adapter 职责

- 提供模型输入/输出语义。
- 提供预处理规则。
- 提供输出解码规则。
- 提供 NMS 或其他后处理规则。
- 向代码生成器提供模板参数。
- 执行模型特定的合法性检查。

MVP 可先支持一个明确的目标检测模型族；后续通过新增 Adapter 扩展模型，而不是修改 TensorRT 核心推理底座。

---

## 7. 模型校验与算子兼容性

### 7.1 校验层级

1. 文件格式校验：扩展名、大小、文件可读性。
2. ONNX Checker：检查 ONNX 模型格式合法性。
3. ONNX Runtime Load Test：尝试加载模型并提取输入输出信息。
4. Graph Analysis：解析节点、输入、输出、动态维度和数据类型。
5. Backend Parser Check：使用目标后端实际解析能力进行进一步检查。
6. Capability Check：根据 Backend Capability Registry 输出算子兼容性报告。

### 7.2 兼容性报告

| 字段       | 说明                      |
| ---------- | ------------------------- |
| operator   | 算子名称                  |
| domain     | 算子 domain               |
| opset      | ONNX opset                |
| supported  | 是否支持                  |
| condition  | 特殊 shape/attribute 限制 |
| severity   | 提示/警告/错误            |
| suggestion | 替换或回退建议            |

---

## 8. 预处理与校准数据规范

### 8.1 预处理参数

- 输入尺寸：固定 H/W 或 Dynamic Shape Profile。
- Tensor Layout：NCHW / NHWC。
- 颜色顺序：RGB / BGR。
- Resize：stretch / letterbox / center crop 等。
- 像素缩放公式：明确 scale、offset、mean、std。
- 输入数据类型：FP32 / FP16 / INT8 等。
- Letterbox padding 的数值和坐标映射规则。

### 8.2 MVP 校准集协议

```text
calibration.zip
└── images/
    ├── 000001.jpg
    ├── 000002.jpg
    └── ...
```

MVP 默认从 `images/` 中读取图像，由平台按照 Model Definition 的 `preprocessing` 配置完成解码、resize、normalize、layout 转换后再送入校准流程。

### 8.3 校准集校验

- 检查图片是否可读取。
- 统计图片数量、分辨率、通道数和文件类型。
- 随机抽样执行预处理并检查 tensor shape/dtype/range。
- 报告校准集异常图片。
- 后续版本可增加场景分布和代表性检查。

---

## 9. 精度模式与量化流程

| 模式 | 定义           | 主要流程                           |
| ---- | -------------- | ---------------------------------- |
| FP32 | 基准精度       | ONNX → TensorRT Engine             |
| FP16 | 半精度推理模式 | ONNX → TensorRT FP16 Engine        |
| INT8 | 静态量化       | 校准集 → Calibration → INT8 Engine |

### 9.1 INT8 流程

1. 加载并验证 ONNX。
2. 加载 Model Definition。
3. 加载并校验 calibration dataset。
4. 执行统一预处理。
5. 运行 PPQ/目标量化流程。
6. 输出量化模型或后端可接受的量化表示。
7. 构建 TensorRT INT8 Engine。
8. 记录量化参数、版本、日志和产物元数据。
9. 执行最小推理验证。

### 9.2 量化误差分析

- Graph/Layer 级误差统计。
- 建议至少支持 MAE、MSE/RMSE、Cosine Similarity 等可配置指标。
- 输出误差排序，定位敏感层。
- 支持量化前后中间结果对比。
- 后续版本可增加混合精度策略和自动敏感层保留 FP16。

---

## 10. TensorRT Engine 构建规范

### 10.1 MVP

- MVP 默认 batch=1，先保证固定输入 shape 主链路稳定。
- 动态 batch / Dynamic Shape 作为后续扩展。
- Engine 必须记录 TensorRT、CUDA、GPU 架构、精度和输入 profile。
- Engine 生成失败必须保存完整 TensorRT parser/build 日志。

### 10.2 Dynamic Shape 扩展

```yaml
optimization_profile:
  min: [1, 3, 640, 640]
  opt: [1, 3, 640, 640]
  max: [4, 3, 640, 640]
```

只有当模型和后端均支持时才允许配置 Dynamic Shape；平台必须在构建前进行合法性检查。

---

## 11. C++ 代码生成规范

### 11.1 模板分层

```text
codegen/
├── common/
│   ├── Logger
│   ├── TensorRT Runtime
│   ├── CUDA utilities
│   └── CMake
├── task/
│   └── detection/
├── model/
│   └── yolov8/
└── config/
```

### 11.2 生成物

- `src/` 与 `include/`：C++ 推理代码。
- `CMakeLists.txt`：完整构建配置。
- `config/model.yaml`：模型和预处理/后处理配置。
- `README.md`：构建、运行、输入输出说明。
- `test/`：最小测试输入。

### 11.3 生成后自动验证

1. 生成工程。
2. CMake Configure。
3. CMake Build。
4. 加载 TensorRT Engine。
5. 运行至少一个样例输入。
6. 检查输出 tensor 是否存在且 shape 正确。
7. 目标检测任务进一步执行后处理，确认输出结构合法。
8. 全部通过后才允许标记 Artifact 为可交付。

---

## 12. Docker 与部署产物

### 12.1 Docker 构建边界

- Docker build 不由 FastAPI Web 进程直接执行。
- 由受控 Worker 执行构建。
- 限制 CPU、内存、磁盘和任务超时时间。
- 默认关闭不必要的网络访问。
- 构建日志必须进入任务日志系统。

### 12.2 Artifact 目录

```text
artifact/
├── model/
│   └── model.engine
├── source/
│   ├── include/
│   ├── src/
│   └── CMakeLists.txt
├── config/
│   └── model.yaml
├── docker/
│   └── Dockerfile
├── test/
├── report/
│   ├── model_info.json
│   ├── compatibility.json
│   └── quantization_report.html
└── README.md
```

---

## 13. 任务状态机与日志

```text
CREATED
   ↓
QUEUED
   ↓
VALIDATING
   ↓
PREPROCESSING
   ↓
QUANTIZING
   ↓
BUILDING_ENGINE
   ↓
GENERATING_CODE
   ↓
BUILDING
   ↓
TESTING
   ↓
PACKAGING
   ↓
SUCCESS

任意可中断阶段 → FAILED
QUEUED/执行阶段 → CANCELLED（按实际可取消能力实现）
```

### 13.1 Task 状态字段

| 字段          | 说明         |
| ------------- | ------------ |
| task_id       | 唯一任务 ID  |
| status        | 当前状态     |
| progress      | 0~100        |
| current_stage | 当前阶段     |
| message       | 用户可读状态 |
| error_code    | 结构化错误码 |
| started_at    | 开始时间     |
| finished_at   | 结束时间     |
| worker_id     | 执行 Worker  |
| artifact_id   | 最终产物     |

### 13.2 日志

- Worker 输出统一结构化日志。
- 日志包含 task_id、stage、timestamp、level、message。
- 前端任务详情页实时展示日志。
- 失败时保留完整原始日志，避免只返回最后一行错误。
- 敏感信息不得直接写入日志。

---

## 14. 数据库与文件存储

### 14.1 推荐数据库表

- `projects`
- `models`
- `datasets`
- `tasks`
- `task_configs`
- `artifacts`
- `job_logs`
- `backends`

### 14.2 文件目录

```text
storage/
└── <task_id>/
    ├── input/
    ├── calibration/
    ├── intermediate/
    ├── engine/
    ├── source/
    ├── docker/
    ├── logs/
    └── report/
```

原始文件、中间文件和最终产物应分层管理。任务结束后的中间文件清理策略需要在系统配置中可配置。

---

## 15. 安全、资源与异常处理

| 风险          | 要求                                     |
| ------------- | ---------------------------------------- |
| 非法上传      | 扩展名 + MIME/内容检查 + 文件大小限制    |
| ZIP Bomb      | 限制压缩包解压后大小、文件数量和目录深度 |
| 路径穿越      | ZIP 解压必须限制在任务专属目录           |
| 恶意 ONNX     | 限制文件大小、解析资源和执行时间         |
| Docker 风险   | Worker 隔离、资源限制、默认最小权限      |
| GPU 资源不足  | 任务队列 + 并发限制 + 可配置资源策略     |
| 编译失败      | 保存完整编译日志并返回结构化错误码       |
| 量化失败      | 保留失败阶段、日志和中间诊断信息         |
| Engine 不兼容 | 保存构建环境元数据并在运行验证阶段检查   |

### 15.1 错误码示例

```text
MODEL_INVALID
MODEL_LOAD_FAILED
OPERATOR_UNSUPPORTED
PREPROCESS_CONFIG_INVALID
CALIBRATION_DATA_INVALID
QUANTIZATION_FAILED
TENSORRT_BUILD_FAILED
CODEGEN_FAILED
CPP_BUILD_FAILED
RUNTIME_TEST_FAILED
DOCKER_BUILD_FAILED
RESOURCE_LIMIT_EXCEEDED
TASK_TIMEOUT
```

---

## 16. 前端功能与 API 边界

| 页面     | 主要功能                                       |
| -------- | ---------------------------------------------- |
| 项目管理 | 创建/查看项目                                  |
| 新建任务 | 上传模型、校准集、选择任务/精度/后端、填写配置 |
| 模型详情 | 输入输出、算子兼容性、模型信息                 |
| 任务详情 | 状态、进度、实时日志、错误信息                 |
| 精度报告 | FP32/FP16/INT8 对比、误差分析                  |
| 产物页面 | Engine、C++ 工程、Dockerfile、报告下载         |

### 16.1 API 最小集合

```text
POST   /api/projects
GET    /api/projects/{id}

POST   /api/models
GET    /api/models/{id}

POST   /api/datasets
GET    /api/datasets/{id}

POST   /api/tasks
GET    /api/tasks/{id}
POST   /api/tasks/{id}/cancel
GET    /api/tasks/{id}/logs
GET    /api/tasks/{id}/artifacts

GET    /api/tasks/{id}/report
```

---

## 17. 阶段实施计划

| 阶段   | 目标                   | 主要交付物                                      |
| ------ | ---------------------- | ----------------------------------------------- |
| 阶段 0 | 脚手架与基础架构       | Vue/FastAPI/Redis/Celery/存储/数据库/任务状态机 |
| 阶段 1 | MVP：2D检测 + TensorRT | 上传→校验→FP16/INT8→Engine→C++→编译→测试→Docker |
| 阶段 2 | V1.0：稳定性和质量     | 分割、误差分析、算子检测、数据集管理、精度验证  |
| 阶段 3 | V1.5：3D PoC           | 先完成单一 3D 模型全流程，再扩展                |
| 阶段 4 | V2.0：多后端           | Backend Adapter + OpenVINO；后续再评估 RKNN     |

### 17.1 阶段验收原则

- 每阶段必须存在可运行版本，而不是只完成代码模块。
- 至少准备一套固定回归模型和测试数据集。
- 修改量化/代码生成逻辑后必须重新执行回归测试。
- 阶段结束时必须保存版本、环境和测试结果。
- 只有自动构建、自动运行验证通过的产物才算阶段交付物。

---

## 18. MVP 验收标准

1. 用户可以上传合法 ONNX 模型。
2. 平台能够解析并展示输入、输出、shape、dtype 和基础算子信息。
3. 用户可以上传校准图片 ZIP，并完成数据校验。
4. 用户可以选择 FP32/FP16/INT8 模式。
5. FP16 Engine 可以成功构建并通过最小推理验证。
6. INT8 Engine 可以使用校准集完成构建并通过最小推理验证。
7. 目标检测 C++ 工程能够自动生成。
8. 生成的 C++ 工程能够在规定环境中通过 CMake 编译。
9. 编译后的程序能够加载 Engine 并对测试输入完成推理。
10. Dockerfile 能够成功构建。
11. 任务详情页可以查看进度和完整日志。
12. 失败任务能够定位到具体阶段并返回结构化错误码。
13. 最终 Artifact 包含 Engine、C++ 工程、配置、Dockerfile、README 和测试/报告文件。

---

## 19. 后续扩展原则

- 新增模型：优先增加 Model Adapter，不修改核心量化/任务调度代码。
- 新增后端：实现 Backend Adapter，不直接修改业务层。
- 新增任务：增加 Task Adapter 和对应代码模板。
- 新增精度：扩展 Precision Strategy。
- 新增评估指标：扩展 Evaluation Module。
- 新增部署环境：扩展 Runtime/Container Profile。
- 平台应保持配置驱动，而不是通过大量 if/else 判断模型和后端。

---

## 20. 开发者执行说明

本规格书的核心目标不是限制实现细节，而是固定系统边界、输入输出和验收标准。资深开发者可以在不改变上述外部行为和模块职责的前提下自行决定具体类结构、数据库 ORM、消息队列实现细节和前端组件组织方式。

如果实现过程中发现某项需求存在技术冲突，应优先记录为设计变更，而不是直接改变接口或数据格式。涉及模型语义、量化策略、Engine 兼容性和最终产物格式的修改，应同步更新 TaskConfig、Model Definition 和验收测试。

---

## 附录 A：MVP 推荐目录结构

```text
onnx-deployment-platform/
├── frontend/
├── backend/
│   ├── api/
│   ├── models/
│   ├── services/
│   ├── tasks/
│   ├── storage/
│   └── config/
├── workers/
│   ├── model_analyzer/
│   ├── quantization/
│   ├── tensorrt/
│   ├── codegen/
│   ├── build/
│   └── packaging/
├── adapters/
│   ├── models/
│   └── backends/
├── templates/
│   ├── common/
│   └── detection/
├── tests/
├── docker/
└── docs/
```

---

## 附录 B：与原项目计划的关系

本规格书保留原计划的核心路线：Vue3 + Element Plus、FastAPI、Celery + Redis、PPQ、TensorRT、Jinja2、Docker，以及“阶段 0 → MVP → V1.0 → V1.5 → V2.0”的演进方式。

本版本重点对原计划中较抽象的工程要求进行明确化，包括统一任务配置、模型适配层、校准数据协议、任务状态机、构建验证、产物结构、安全边界和验收标准。新增内容中，凡属于工程实现建议而非原计划明确要求的，均应由项目负责人在正式开发前确认。
