<script setup>
import { onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { api } from '@/api/client'
import PhasePlaceholder from '@/components/PhasePlaceholder.vue'

const route = useRoute()
const models = ref([])
const selected = ref(null)

onMounted(async () => {
  try {
    if (route.params.id) {
      selected.value = await api.getModel(route.params.id)
    } else {
      models.value = await api.listModels()
    }
  } catch (error) {
    ElMessage.error(error.message)
  }
})
</script>

<template>
  <div>
    <el-card v-if="selected" shadow="never">
      <template #header>模型详情</template>
      <el-descriptions :column="2" border>
        <el-descriptions-item label="名称">{{ selected.name }}</el-descriptions-item>
        <el-descriptions-item label="架构">{{ selected.architecture || '—' }}</el-descriptions-item>
        <el-descriptions-item label="任务类型">{{ selected.task_type }}</el-descriptions-item>
        <el-descriptions-item label="opset">{{ selected.opset ?? '—' }}</el-descriptions-item>
        <el-descriptions-item label="文件路径">
          {{ selected.file_path || '（阶段 1 上传后生成）' }}
        </el-descriptions-item>
        <el-descriptions-item label="SHA256">{{ selected.sha256 || '—' }}</el-descriptions-item>
      </el-descriptions>

      <el-divider content-position="left">Model Definition（SPEC 6.1）</el-divider>
      <pre class="json">{{ JSON.stringify(selected.model_definition, null, 2) }}</pre>
    </el-card>

    <el-card v-else shadow="never">
      <template #header>模型列表</template>
      <el-table :data="models" empty-text="暂无模型">
        <el-table-column prop="name" label="名称" min-width="160" />
        <el-table-column prop="architecture" label="架构" min-width="120" />
        <el-table-column prop="task_type" label="任务类型" min-width="120" />
        <el-table-column label="详情" min-width="120">
          <template #default="{ row }">
            <router-link :to="`/models/${row.id}`">查看</router-link>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <PhasePlaceholder
      phase="阶段 1"
      feature="模型解析结果"
      detail="输入/输出 shape、dtype、算子兼容性报告由阶段 1 的 ONNX Checker 与 Backend Parser 填充（SPEC 7）。"
      style="margin-top: 16px"
    />
  </div>
</template>

<style scoped>
.json {
  background: #f5f7fa;
  padding: 12px;
  border-radius: 4px;
  max-height: 320px;
  overflow: auto;
  font-size: 12px;
}
</style>
