<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { api } from '@/api/client'

const route = useRoute()
const router = useRouter()

const tasks = ref([])
const selectedId = ref(route.params.id || '')
const task = ref(null)
const logs = ref([])
const artifacts = ref([])
// 算子兼容性报告（SPEC 7.2）：未生成时为 null，界面明确说明而不是留空白
const compatibility = ref(null)
const problemOperators = computed(() =>
  (compatibility.value?.operators || []).filter((item) => item.severity !== 'info'),
)
const otherOperators = computed(() =>
  (compatibility.value?.operators || []).filter((item) => item.severity === 'info'),
)

function verdictType(verdict) {
  if (verdict === 'BLOCKED') return 'error'
  if (verdict === 'WARNINGS') return 'warning'
  return 'success'
}

function severityType(severity) {
  if (severity === 'error') return 'danger'
  if (severity === 'warning') return 'warning'
  return 'info'
}
const transitions = ref(null)
const autoRefresh = ref(true)
let timer = null

const canEnqueue = computed(() => task.value?.status === 'CREATED')
const canCancel = computed(
  () => task.value && !['SUCCESS', 'FAILED', 'CANCELLED'].includes(task.value.status),
)
// 最终交付物 = zip 归档（解压后可一键启动推理）
const archive = computed(
  () =>
    artifacts.value.find((item) => item.kind === 'archive') ||
    artifacts.value.find((item) => item.relative_path?.endsWith('artifact.zip')) ||
    null,
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
    artifacts.value = await api.getTaskArtifacts(selectedId.value)
    // 报告可能还不存在（任务未跑到模型校验阶段 → 501），静默视为"暂无"
    compatibility.value = await api.getTaskCompatibility(selectedId.value).catch(() => null)
  } catch (error) {
    ElMessage.error(error.message)
  }
}

function selectTask(id) {
  // 只改地址，细节由下面的 watch 统一加载（避免重复请求，也保证浏览器前进/后退一致）
  router.push(`/tasks/${id}`)
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

// 列表与详情共用路由（/tasks/:id?），组件会被复用：
// 必须监听路由参数变化，否则从别处跳进来（或浏览器前进/后退）不会重新加载，要手动刷新。
watch(
  () => route.params.id,
  async (id) => {
    selectedId.value = id || ''
    await loadDetail()
  },
  { immediate: true },
)

onMounted(async () => {
  await loadList()
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

        <el-divider content-position="left">算子兼容性（SPEC 7.2）</el-divider>
        <template v-if="compatibility">
          <el-alert
            :type="verdictType(compatibility.summary?.verdict)"
            :closable="false"
            show-icon
            :title="`结论：${compatibility.summary?.verdict}　错误 ${compatibility.summary?.error_count ?? 0} / 警告 ${compatibility.summary?.warning_count ?? 0}　算子种类 ${compatibility.summary?.operator_kinds ?? 0}　opset ${compatibility.summary?.opset ?? '—'}`"
          />
          <el-table
            :data="problemOperators"
            size="small"
            style="margin-top: 8px"
            empty-text="没有需要关注的算子（其余算子无风险提示）"
          >
            <el-table-column prop="operator" label="算子" width="150" />
            <el-table-column prop="opset" label="opset" width="70" />
            <el-table-column label="支持" width="80">
              <template #default="{ row }">
                <el-tag :type="row.supported ? 'success' : 'danger'" size="small">
                  {{ row.supported ? '是' : '否' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="级别" width="90">
              <template #default="{ row }">
                <el-tag :type="severityType(row.severity)" size="small">{{ row.severity }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="occurrences" label="出现" width="70" />
            <el-table-column prop="condition" label="限制条件" min-width="200" show-overflow-tooltip />
            <el-table-column prop="suggestion" label="建议" min-width="240" show-overflow-tooltip />
          </el-table>
          <el-collapse v-if="otherOperators.length" style="margin-top: 8px">
            <el-collapse-item :title="`其余 ${otherOperators.length} 个算子（无风险提示）`">
              <el-table :data="otherOperators" size="small">
                <el-table-column prop="operator" label="算子" min-width="150" />
                <el-table-column prop="domain" label="domain" width="110" />
                <el-table-column prop="occurrences" label="出现次数" width="100" />
              </el-table>
            </el-collapse-item>
          </el-collapse>
        </template>
        <el-alert
          v-else
          type="info"
          :closable="false"
          title="尚无算子兼容性报告（任务需先完成模型校验阶段）"
        />

        <el-divider content-position="left">产物（最终交付物是 zip，解压后跑 start.bat / start.sh）</el-divider>
        <el-alert
          v-if="archive"
          type="success"
          :closable="false"
          show-icon
          style="margin-bottom: 8px"
        >
          <template #title>
            最终交付物：{{ archive.relative_path.split('/').pop() }}（{{ archive.description }}）
            <a :href="api.artifactDownloadUrl(task.id, archive.id)" style="margin-left: 8px">
              立即下载
            </a>
            <router-link :to="`/artifacts/${task.id}`" style="margin-left: 12px">
              查看产物页面
            </router-link>
          </template>
        </el-alert>
        <el-table :data="artifacts" max-height="260" empty-text="尚无产物（任务未跑到 PACKAGING 阶段）">
          <el-table-column prop="kind" label="类型" width="100" />
          <el-table-column prop="description" label="说明" min-width="220" show-overflow-tooltip />
          <el-table-column prop="relative_path" label="相对路径" min-width="260" show-overflow-tooltip />
          <el-table-column label="大小" width="110">
            <template #default="{ row }">
              {{ row.size_bytes ? (row.size_bytes / 1024).toFixed(1) + ' KB' : '—' }}
            </template>
          </el-table-column>
          <el-table-column label="下载" width="100">
            <template #default="{ row }">
              <a :href="api.artifactDownloadUrl(task.id, row.id)" target="_blank">下载</a>
            </template>
          </el-table-column>
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
