<script setup>
import { onMounted, ref } from 'vue'
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
      title="Engine、C++ 工程、Dockerfile 与报告属阶段 1 交付；当前阶段产物表为空是预期结果。"
      style="margin-bottom: 12px"
    />

    <el-table v-if="loaded" :data="artifacts" empty-text="暂无产物">
      <el-table-column prop="kind" label="类型" min-width="120" />
      <el-table-column prop="relative_path" label="相对路径" min-width="260" />
      <el-table-column prop="size_bytes" label="大小(字节)" min-width="120" />
      <el-table-column prop="created_at" label="生成时间" min-width="200" />
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
