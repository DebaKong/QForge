<script setup>
// 明确的阶段占位：不用假数据冒充已实现功能
defineProps({
  phase: { type: String, required: true },
  feature: { type: String, required: true },
  detail: { type: String, default: '' },
})
</script>

<template>
  <el-card shadow="never">
    <template #header>{{ feature }}</template>
    <el-alert type="warning" :closable="false" show-icon>
      <template #title>该功能在 {{ phase }} 交付，当前阶段未实现</template>
      <div v-if="detail" class="detail">{{ detail }}</div>
    </el-alert>
  </el-card>
</template>

<style scoped>
.detail {
  margin-top: 6px;
  color: #606266;
}
</style>
