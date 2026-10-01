"""Model Adapter 实现集合。

导入本包即完成注册（业务层通过 `app.adapters.registry` 获取适配器）。
"""

from app.adapters.models.yolov8 import YOLOv8Adapter

__all__ = ["YOLOv8Adapter"]
