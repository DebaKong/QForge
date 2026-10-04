<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { api } from '@/api/client'

const route = useRoute()
const tasks = ref([])
const selectedId = ref(route.params?.id || '')
const report = ref(null)
const loading = ref(false)
const unavailable = ref('')

const accuracy = computed(() => report.value?.accuracy || null)
// accuracy.json 的真实结构：engine_vs_fp32_baseline / cpp_program_vs_fp32_baseline
const engineAccuracy = computed(() => accuracy.value?.engine_vs_fp32_baseline || null)
const cppAccuracy = computed(() => accuracy.value?.cpp_program_vs_fp32_baseline || null)
const layers = computed(() => report.value?.layer_error_analysis || null)
const ranking = computed(() => layers.value?.ranking || [])
const allLayers = computed(() => layers.value?.layers || [])

function fmt(value, digits = 6) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  if (typeof value !== 'number') return String(value)
  if (value !== 0 && Math.abs(value) < 1e-4) return value.toExponential(3)
  return value.toFixed(digits)
}

// 相对 RMSE 进度条：以榜首为 100%，让"哪几层最敏感"一眼可读
function barRate(value) {
  const top = ranking.value[0]?.relative_rmse || 0
  if (!top) return 0
  return Math.min(100, Math.round((value / top) * 100))
}

async function loadTasks() {
  try {
    const items = await api.listTasks({})
    tasks.value = items
    if (!selectedId.value && items.length) {
      // 默认选最近一个跑到过 TESTING 之后的任务
      const done = items.find((item) => !['CREATED', 'QUEUED'].includes(item.status))
      selectedId.value = (done || items[0]).id
    }
  } catch (error) {
    ElMessage.error(error.message)
  }
}

async function loadReport() {
  if (!selectedId.value) {
    report.value = null
    return
  }
  loading.value = true
  unavailable.value = ''
  try {
    report.value = await api.getTaskPrecision(selectedId.value)
  } catch (error) {
    report.value = null
    // 501 = 还没跑到测试阶段；其余错误照实显示
    unavailable.value = error.message
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  await loadTasks()
  await loadReport()
})

watch(
  () => route.params?.id,
  (id) => {
    if (id && id !== selectedId.value) {
      selectedId.value = id
      loadReport()
    }
  },
)
</script>

<template>
  <div v-loading="loading">
    <el-card shadow="never">
      <template #header>
        <div class="card-header">
          <span>精度报告（SPEC 9.2）</span>
          <div class="actions">
            <el-select
              v-model="selectedId"
              placeholder="选择任务"
              style="width: 360px"
              @change="loadReport"
            >
              <el-option
                v-for="item in tasks"
                :key="item.id"
                :label="`${item.precision.toUpperCase()} · ${item.status} · ${item.id.slice(0, 8)}`"
                :value="item.id"
              />
            </el-select>
            <el-button @click="loadReport">刷新</el-button>
          </div>
        </div>
      </template>

      <el-alert
        v-if="unavailable"
        type="info"
        :closable="false"
        show-icon
        :title="`暂无精度报告：${unavailable}`"
      />

      <template v-else-if="report">
        <!-- 端到端精度：SPEC 9 定义的"与 FP32 基准对比" -->
        <el-divider content-position="left">端到端精度（真实 Engine vs FP32 基准）</el-divider>
        <template v-if="engineAccuracy">
          <el-descriptions :column="3" border>
            <el-descriptions-item label="基准">{{ engineAccuracy.baseline }}</el-descriptions-item>
            <el-descriptions-item label="本任务精度">
              {{ accuracy?.precision?.toUpperCase() || '—' }}
            </el-descriptions-item>
            <el-descriptions-item label="元素数">{{ engineAccuracy.elements }}</el-descriptions-item>
            <el-descriptions-item label="余弦相似度">
              {{ fmt(engineAccuracy.cosine_similarity) }}
            </el-descriptions-item>
            <el-descriptions-item label="MAE">{{ fmt(engineAccuracy.mae) }}</el-descriptions-item>
            <el-descriptions-item label="RMSE">{{ fmt(engineAccuracy.rmse) }}</el-descriptions-item>
            <el-descriptions-item label="最大绝对误差">
              {{ fmt(engineAccuracy.max_abs_error) }}
            </el-descriptions-item>
            <el-descriptions-item label="基准绝对峰值">
              {{ fmt(engineAccuracy.reference_abs_max) }}
            </el-descriptions-item>
            <el-descriptions-item label="实际绝对峰值">
              {{ fmt(engineAccuracy.actual_abs_max) }}
            </el-descriptions-item>
          </el-descriptions>
          <el-descriptions
            v-if="cppAccuracy"
            :column="3"
            border
            style="margin-top: 12px"
            title="生成的 C++ 程序（同一 Engine）"
          >
            <el-descriptions-item label="MAE">{{ fmt(cppAccuracy.mae) }}</el-descriptions-item>
            <el-descriptions-item label="RMSE">{{ fmt(cppAccuracy.rmse) }}</el-descriptions-item>
            <el-descriptions-item label="余弦相似度">
              {{ fmt(cppAccuracy.cosine_similarity) }}
            </el-descriptions-item>
          </el-descriptions>
        </template>
        <el-alert
          v-else
          type="warning"
          :closable="false"
          title="该任务的端到端精度报告缺失（可能未执行 Engine 推理验证）"
        />

        <!-- 分层误差分析：定位敏感层 -->
        <el-divider content-position="left">分层误差分析（定位量化敏感层）</el-divider>
        <template v-if="layers">
          <el-alert
            :type="layers.status === 'SUCCESS' ? 'success' : 'warning'"
            :closable="false"
            show-icon
            :title="`状态：${layers.status}　分析层数：${layers.analyzed_layers}/${layers.requested_layers}　输入样本：${layers.inputs_used}${layers.reason ? '　原因：' + layers.reason : ''}`"
          />
          <div class="hint">{{ layers.method }}</div>

          <template v-if="ranking.length">
            <el-table :data="ranking" size="small" style="margin-top: 8px">
              <el-table-column prop="rank" label="#" width="50" />
              <el-table-column prop="operator" label="算子" width="130" />
              <el-table-column prop="tensor" label="张量" min-width="200" show-overflow-tooltip />
              <el-table-column label="相对 RMSE（越靠前越敏感）" min-width="240">
                <template #default="{ row }">
                  <el-progress :percentage="barRate(row.relative_rmse)" :stroke-width="12" />
                  <span class="mono">{{ fmt(row.relative_rmse) }}</span>
                </template>
              </el-table-column>
              <el-table-column label="RMSE" width="120">
                <template #default="{ row }">{{ fmt(row.rmse) }}</template>
              </el-table-column>
              <el-table-column label="余弦" width="120">
                <template #default="{ row }">{{ fmt(row.cosine_similarity) }}</template>
              </el-table-column>
              <el-table-column prop="elements" label="元素数" width="100" />
            </el-table>

            <el-collapse style="margin-top: 12px">
              <el-collapse-item :title="`全部 ${allLayers.length} 层明细`">
                <el-table :data="allLayers" size="small" max-height="420">
                  <el-table-column prop="node_index" label="节点" width="70" />
                  <el-table-column prop="operator" label="算子" width="120" />
                  <el-table-column prop="tensor" label="张量" min-width="200" show-overflow-tooltip />
                  <el-table-column label="MAE" width="110">
                    <template #default="{ row }">{{ fmt(row.mae) }}</template>
                  </el-table-column>
                  <el-table-column label="RMSE" width="110">
                    <template #default="{ row }">{{ fmt(row.rmse) }}</template>
                  </el-table-column>
                  <el-table-column label="相对 RMSE" width="120">
                    <template #default="{ row }">{{ fmt(row.relative_rmse) }}</template>
                  </el-table-column>
                  <el-table-column label="最大绝对误差" width="130">
                    <template #default="{ row }">{{ fmt(row.max_abs_error) }}</template>
                  </el-table-column>
                  <el-table-column label="余弦" width="120">
                    <template #default="{ row }">{{ fmt(row.cosine_similarity) }}</template>
                  </el-table-column>
                  <el-table-column prop="elements" label="元素数" width="100" />
                </el-table>
              </el-collapse-item>
            </el-collapse>
          </template>
        </template>
        <el-alert
          v-else
          type="info"
          :closable="false"
          title="该任务尚无分层误差分析结果"
        />

        <el-alert
          v-for="note in layers?.notes || []"
          :key="note"
          type="info"
          :closable="false"
          :title="note"
          style="margin-top: 8px"
        />
      </template>
    </el-card>
  </div>
</template>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.actions {
  display: flex;
  gap: 8px;
}
.hint {
  color: #909399;
  font-size: 12px;
  margin-top: 6px;
}
.mono {
  font-family: Consolas, 'Courier New', monospace;
  font-size: 12px;
}
</style>
