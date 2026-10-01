"""适配器层（SPEC 6 / 19：模型语义与后端实现隔离）。

- `adapters.models`：Model Adapter，描述模型输入/输出语义、预处理、后处理与模板参数；
- `adapters.backends`：Backend Adapter，隔离 TensorRT 等具体后端实现。

业务层（pipeline / codegen / API）只面向适配器接口，不做 `if backend == "tensorrt"` 之类的分支。

导入策略：
- **模型适配器在包导入时注册**（它们只依赖 numpy，导入成本低），
  否则注册表会是空的，`get_model_adapter("yolov8")` 会误报「不支持的架构」；
- **后端适配器按需导入**（`load_backend_adapter`），避免未安装 TensorRT 时整个服务起不来。

目录说明：SPEC 附录 A 把 `adapters/` 放在仓库根与 `backend/` 同级；此处放在
`backend/app/adapters/` 内，使其作为 `app` 包的一部分可直接导入，属文档化的目录偏差。
"""

from app.adapters import models  # noqa: F401  导入以完成 Model Adapter 注册

__all__ = ["models"]
