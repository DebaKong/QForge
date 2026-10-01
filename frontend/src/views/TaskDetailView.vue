<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { api } from '@/api/client'

const route = useRoute()
const router = useRouter()

const tasks = ref([])
const selectedId = ref(route.params.id || '')
const task = ref(null)
const logs = ref([])
const transitions = ref(null)
const autoRefresh = ref(true)
let timer = null

const canEnqueue = computed(() => task.value?.status === 'CREATED')
const canCancel = computed(
  () => task.value && !['SUCCESS', 'FAILED', 'CANCELLED'].includes(task.value.status),
)

async function loadList() {
  try {
    tasks.value = await api.listTasks({})
  } catch (error) {
    ElMessage.error(error.message)
  }
}

async function loadDetail() {
  if (!selectedId.value) {
    task.value = null
    return
  }
  try {
    task.value = await api.getTask(selectedId.value)
    logs.value = await api.getTaskLogs(selectedId.value)
    transitions.value = await api.getTaskTransitions(selectedId.value)
  } catch (error) {
    ElMessage.error(error.message)
  }
}

function selectTask(id) {
  selectedId.value = id
  router.push(`/tasks/${id}`)
  loadDetail()
}

async function enqueue() {
  try {
    await api.enqueueTask(selectedId.value)
    ElMessage.success('已入队（阶段 0 占位 Worker 只登记领取记录）')
  } catch (error) {
    ElMessage.error(error.message)
  }
  loadDetail()
}

async function cancel() {
  try {
    await api.cancelTask(selectedId.value)
    ElMessage.success('任务已取消')
  } catch (error) {
    ElMessage.error(error.message)
  }
  loadDetail()
}

onMounted(async () => {
  await loadList()
  await loadDetail()
  timer = setInterval(() => {
    if (autoRefresh.value) loadDetail()
  }, 3000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<template>
  <el-row :gutter="16">
    <el-col :span="8">
      <el-card shadow="never">
        <template #header>任务列表</template>
        <el-table
          :data="tasks"
          highlight-current-row
          empty-text="暂无任务"
          @row-click="(row) => selectTask(row.id)"
        >
          <el-table-column prop="id" label="任务 ID" min-width="180" />
          <el-table-column prop="status" label="状态" min-width="120" />
        </el-table>
      </el-card>
    </el-col>

    <el-col :span="16">
      <el-card v-if="task" shadow="never">
        <template #header>
          <div class="card-header">
            <span>任务详情 · {{ task.id }}</span>
            <span>
              <el-switch v-model="autoRefresh" active-text="自动刷新" style="margin-right: 12px" />
              <el-button :disabled="!canEnqueue" type="primary" @click="enqueue">入队</el-button>
              <el-button :disabled="!canCancel" type="danger" plain @click="cancel">取消</el-button>
            </span>
          </div>
        </template>

        <el-descriptions :column="2" border>
          <el-descriptions-item label="状态">{{ task.status }}</el-descriptions-item>
          <el-descriptions-item label="进度">{{ task.progress }}%</el-descriptions-item>
          <el-descriptions-item label="当前阶段">{{ task.current_stage || '—' }}</el-descriptions-item>
          <el-descriptions-item label="精度">{{ task.precision }}</el-descriptions-item>
          <el-descriptions-item label="后端">{{ task.backend_name }}</el-descriptions-item>
          <el-descriptions-item label="Worker">{{ task.worker_id || '—' }}</el-descriptions-item>
          <el-descriptions-item label="错误码">{{ task.error_code || '—' }}</el-descriptions-item>
          <el-descriptions-item label="消息">{{ task.message || '—' }}</el-descriptions-item>
        </el-descriptions>

        <el-alert
          v-if="transitions"
          type="info"
          :closable="false"
          style="margin-top: 12px"
          :title="`允许的下一步：${transitions.allowed_transitions.join(' / ') || '（终态）'}`"
        />

        <el-divider content-position="left">阶段日志</el-divider>
        <el-table :data="logs" max-height="320" empty-text="暂无日志">
          <el-table-column prop="created_at" label="时间" min-width="200" />
          <el-table-column prop="stage" label="阶段" min-width="140" />
          <el-table-column prop="level" label="级别" width="90" />
          <el-table-column prop="message" label="消息" min-width="280" />
        </el-table>
      </el-card>

      <el-empty v-else description="请选择左侧任务，或先创建任务" />
    </el-col>
  </el-row>
</template>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
