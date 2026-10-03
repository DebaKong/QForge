<script setup>
import { ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { api } from '@/api/client'

const route = useRoute()
const models = ref([])
const selected = ref(null)
const loading = ref(false)

// 列表与详情共用同一个路由（/models/:id?），组件实例会被复用。
// 只写 onMounted 的话，从列表点"查看"时钩子不会再触发 —— 表现就是"点了不跳转、要手动刷新"。
// 因此必须监听路由参数变化（immediate 让首次进入也执行）。
async function load(id) {
  loading.value = true
  try {
    if (id) {
      selected.value = await api.getModel(id)
    } else {
      selected.value = null
      models.value = await api.listModels()
    }
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loading.value = false
  }
}

watch(() => route.params.id, (id) => load(id), { immediate: true })

function formatShape(spec) {
  if (!spec || !spec.length) return '—'
  return spec.map((item) => `${item.name} ${JSON.stringify(item.shape)} ${item.dtype}`).join('；')
}
</script>

<template>
  <div v-loading="loading">
    <el-card v-if="selected" shadow="never">
      <template #header>
        <div class="card-header">
          <span>模型详情：{{ selected.name }}</span>
          <router-link to="/models">← 返回模型列表</router-link>
        </div>
      </template>

      <el-descriptions :column="2" border>
        <el-descriptions-item label="名称">{{ selected.name }}</el-descriptions-item>
        <el-descriptions-item label="架构">{{ selected.architecture || '—' }}</el-descriptions-item>
        <el-descriptions-item label="任务类型">{{ selected.task_type }}</el-descriptions-item>
        <el-descriptions-item label="opset">{{ selected.opset ?? '—' }}</el-descriptions-item>
        <el-descriptions-item label="大小">
          {{ ((selected.size_bytes || 0) / 1024 / 1024).toFixed(2) }} MB
        </el-descriptions-item>
        <el-descriptions-item label="状态">{{ selected.status || '—' }}</el-descriptions-item>
        <el-descriptions-item label="文件路径" :span="2">
          {{ selected.file_path || '—' }}
        </el-descriptions-item>
        <el-descriptions-item label="SHA256" :span="2">
          <span class="mono">{{ selected.sha256 || '—' }}</span>
        </el-descriptions-item>
      </el-descriptions>

      <el-divider content-position="left">模型解析结果（ONNX 校验时记录）</el-divider>
      <el-descriptions :column="1" border>
        <el-descriptions-item label="输入">
          <span class="mono">{{ formatShape(selected.input_spec) }}</span>
        </el-descriptions-item>
        <el-descriptions-item label="输出">
          <span class="mono">{{ formatShape(selected.output_spec) }}</span>
        </el-descriptions-item>
      </el-descriptions>

      <el-divider content-position="left">Model Definition（SPEC 6.1）</el-divider>
      <pre class="json">{{ JSON.stringify(selected.model_definition, null, 2) }}</pre>
    </el-card>

    <el-card v-else shadow="never">
      <template #header>模型列表</template>
      <el-table :data="models" empty-text="暂无模型（先到「上传模型与数据」上传）">
        <el-table-column prop="name" label="名称" min-width="160" />
        <el-table-column prop="architecture" label="架构" min-width="120" />
        <el-table-column prop="task_type" label="任务类型" min-width="120" />
        <el-table-column prop="opset" label="opset" width="90" />
        <el-table-column label="详情" min-width="120">
          <template #default="{ row }">
            <router-link :to="`/models/${row.id}`">查看</router-link>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style scoped>
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.json {
  background: #f5f7fa;
  padding: 12px;
  border-radius: 4px;
  max-height: 320px;
  overflow: auto;
  font-size: 12px;
}
.mono {
  font-family: Consolas, 'Courier New', monospace;
  font-size: 12px;
}
</style>
