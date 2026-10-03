<script setup>
import { onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { api } from '@/api/client'

const route = useRoute()
const taskId = ref(route.params.taskId || '')
const artifacts = ref([])
const loaded = ref(false)

async function load() {
  if (!taskId.value) {
    loaded.value = true
    artifacts.value = []
    return
  }
  try {
    artifacts.value = await api.getTaskArtifacts(taskId.value)
    loaded.value = true
  } catch (error) {
    ElMessage.error(error.message)
  }
}

onMounted(load)

// 从任务详情跳过来时组件会被复用（同一路由 /artifacts/:taskId?）：
// 必须监听参数变化，否则要手动刷新才显示产物。
watch(
  () => route.params.taskId,
  (id) => {
    if (id && id !== taskId.value) {
      taskId.value = id
      load()
    }
  },
)
</script>

<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-header">
        <span>产物页面</span>
        <el-input
          v-model="taskId"
          placeholder="输入任务 ID 查询产物"
          style="width: 320px"
          @keyup.enter="load"
        >
          <template #append>
            <el-button @click="load">查询</el-button>
          </template>
        </el-input>
      </div>
    </template>

    <el-alert
      type="info"
      :closable="false"
      title="阶段 1：Engine、C++ 工程、Dockerfile、报告与 artifact.zip 均在此列出，可直接下载。"
      style="margin-bottom: 12px"
    />

    <el-table v-if="loaded" :data="artifacts" empty-text="暂无产物（任务需跑到 PACKAGING 阶段）">
      <el-table-column prop="kind" label="类型" min-width="110" />
      <el-table-column prop="relative_path" label="相对路径" min-width="260" />
      <el-table-column label="大小" min-width="110">
        <template #default="{ row }">
          {{ row.size_bytes ? (row.size_bytes / 1024).toFixed(1) + ' KB' : '—' }}
        </template>
      </el-table-column>
      <el-table-column prop="created_at" label="生成时间" min-width="200" />
      <el-table-column label="下载" width="100">
        <template #default="{ row }">
          <a :href="api.artifactDownloadUrl(taskId, row.id)" target="_blank">下载</a>
        </template>
      </el-table-column>
    </el-table>
  </el-card>
</template>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
