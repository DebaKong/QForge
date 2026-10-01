<script setup>
import PhasePlaceholder from '@/components/PhasePlaceholder.vue'
</script>

<template>
  <PhasePlaceholder
    phase="阶段 2"
    feature="精度报告"
    detail="FP32/FP16/INT8 对比与误差分析（MAE / MSE / RMSE / Cosine Similarity）属 SPEC 17 阶段 2；后端 GET /api/tasks/{id}/report 在当前阶段返回 501 NOT_IMPLEMENTED。"
  />
</template>
