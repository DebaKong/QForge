<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api } from '@/api/client'

const projects = ref([])
const selectedProject = ref('')
const datasets = ref([])
const models = ref([])
const loading = ref(false)

const selected = ref([])
const tableRef = ref(null)

// 详情抽屉
const detailVisible = ref(false)
const detailDataset = ref(null)
const inventory = ref(null)
const inventoryLoading = ref(false)
const page = ref(1)
const pageSize = ref(24)
const revalidateModelId = ref('')
const revalidating = ref(false)

const badFiles = computed(() => inventory.value?.bad_files || [])
const stats = computed(() => inventory.value?.stats || { resolutions: {}, channels: {}, formats: {} })
const samples = computed(() => (inventory.value?.items || []).filter((item) => item.ok).slice(0, 12))

function sizeText(bytes) {
  if (!bytes) return '0 B'
  const mb = bytes / 1024 / 1024
  if (mb >= 1) return `${mb.toFixed(1)} MB`
  return `${(bytes / 1024).toFixed(1)} KB`
}

function validationText(dataset) {
  const validation = dataset.meta?.validation
  if (!validation) return '未校验'
  const bad = validation.bad_file_count ?? 0
  return bad ? `已校验（异常 ${bad} 张）` : '已校验'
}

function validationType(dataset) {
  const validation = dataset.meta?.validation
  if (!validation) return 'info'
  return (validation.bad_file_count ?? 0) > 0 ? 'warning' : 'success'
}

async function loadProjects() {
  projects.value = await api.listProjects()
  if (!selectedProject.value && projects.value.length) {
    selectedProject.value = projects.value[0].id
  }
}

async function loadDatasets() {
  loading.value = true
  try {
    datasets.value = await api.listDatasets(selectedProject.value || undefined)
    models.value = selectedProject.value ? await api.listModels(selectedProject.value) : []
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loading.value = false
  }
}

async function onProjectChange() {
  selected.value = []
  tableRef.value?.clearSelection()
  await loadDatasets()
}

function onSelectionChange(rows) {
  selected.value = rows
}

async function openDetail(row) {
  detailDataset.value = row
  detailVisible.value = true
  page.value = 1
  revalidateModelId.value = ''
  await loadInventory()
}

async function loadInventory() {
  if (!detailDataset.value) return
  inventoryLoading.value = true
  try {
    inventory.value = await api.getDatasetInventory(detailDataset.value.id, {
      offset: (page.value - 1) * pageSize.value,
      limit: pageSize.value,
    })
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    inventoryLoading.value = false
  }
}

async function revalidate() {
  if (!detailDataset.value) return
  revalidating.value = true
  try {
    const report = await api.revalidateDataset(detailDataset.value.id, {
      model_id: revalidateModelId.value || undefined,
    })
    ElMessage.success(
      `重新校验完成：${report.image_count} 张图像，异常 ${report.bad_file_count ?? 0} 张`,
    )
    await loadInventory()
    await loadDatasets()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    revalidating.value = false
  }
}

async function deleteDatasets(rows) {
  if (!rows.length) return
  const names = rows.map((row) => row.name).join('、')
  const images = rows.reduce((sum, row) => sum + (row.image_count || 0), 0)
  try {
    await ElMessageBox.confirm(
      `确认删除 ${rows.length} 个数据集（${names}）？将删除其 ${images} 张图像文件，该操作不可恢复。`,
      '删除数据集',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }

  try {
    const result = await api.deleteDatasets(rows.map((row) => row.id))
    if (result.failed_count) {
      ElMessage.warning(
        `删除 ${result.deleted_count} 个，失败 ${result.failed_count} 个：` +
          result.failed.map((item) => item.reason).join('；'),
      )
    } else {
      ElMessage.success(`已删除 ${result.deleted_count} 个数据集，释放 ${sizeText(result.freed_bytes)}`)
    }
    selected.value = []
    tableRef.value?.clearSelection()
    await loadDatasets()
  } catch (error) {
    ElMessage.error(error.message)
  }
}

onMounted(async () => {
  await loadProjects()
  await loadDatasets()
})
</script>

<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-header">
        <span>数据集管理</span>
        <div class="actions">
          <el-select
            v-model="selectedProject"
            placeholder="选择项目"
            style="width: 220px"
            @change="onProjectChange"
          >
            <el-option
              v-for="item in projects"
              :key="item.id"
              :label="item.name"
              :value="item.id"
            />
          </el-select>
          <el-button :disabled="!selected.length" type="danger" plain @click="deleteDatasets(selected)">
            删除选中（{{ selected.length }}）
          </el-button>
          <el-button @click="loadDatasets">刷新</el-button>
        </div>
      </div>
    </template>

    <el-alert
      type="info"
      :closable="false"
      show-icon
      title="数据集上传在「上传模型与数据」页；这里用于查看统计、抽样预览、重新校验与删除（SPEC 8.3）。"
      style="margin-bottom: 12px"
    />

    <el-table
      ref="tableRef"
      v-loading="loading"
      :data="datasets"
      row-key="id"
      empty-text="该项目还没有数据集"
      @selection-change="onSelectionChange"
    >
      <el-table-column type="selection" width="46" reserve-selection />
      <el-table-column prop="name" label="名称" min-width="160" />
      <el-table-column prop="kind" label="类型" width="110" />
      <el-table-column prop="image_count" label="图像数" width="90" />
      <el-table-column label="校验状态" width="170">
        <template #default="{ row }">
          <el-tag :type="validationType(row)" size="small">{{ validationText(row) }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="created_at" label="创建时间" min-width="190" />
      <el-table-column label="操作" width="210" fixed="right">
        <template #default="{ row }">
          <el-button link type="primary" @click="openDetail(row)">详情</el-button>
          <el-button link @click="openDetail(row)">重新校验</el-button>
          <el-button link type="danger" @click="deleteDatasets([row])">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 数据集详情：统计 + 抽样预览 + 图像清单 + 异常图片 + 重新校验 -->
    <el-drawer v-model="detailVisible" size="72%" :title="`数据集详情：${detailDataset?.name || ''}`">
      <div v-loading="inventoryLoading">
        <el-descriptions :column="4" border>
          <el-descriptions-item label="图像总数">
            {{ inventory?.image_count ?? detailDataset?.image_count ?? 0 }}
          </el-descriptions-item>
          <el-descriptions-item label="本次扫描">
            {{ inventory?.scanned ?? 0 }}
            <span v-if="inventory?.truncated" class="hint">（已截断，统计基于前 {{ inventory?.scanned }} 张）</span>
          </el-descriptions-item>
          <el-descriptions-item label="占用空间">{{ sizeText(inventory?.total_bytes) }}</el-descriptions-item>
          <el-descriptions-item label="异常图像">
            <span :class="{ danger: badFiles.length }">{{ badFiles.length }} 张</span>
          </el-descriptions-item>
        </el-descriptions>

        <div class="inline-actions">
          <el-select
            v-model="revalidateModelId"
            placeholder="（可选）选择模型以执行抽样预处理检查"
            style="width: 340px"
            clearable
          >
            <el-option
              v-for="item in models"
              :key="item.id"
              :label="`${item.name}（${item.architecture || '未知架构'}）`"
              :value="item.id"
            />
          </el-select>
          <el-button type="primary" :loading="revalidating" @click="revalidate">重新校验</el-button>
        </div>

        <el-divider content-position="left">统计分布（SPEC 8.3）</el-divider>
        <el-row :gutter="12">
          <el-col :span="8">
            <div class="stat-title">分辨率</div>
            <el-tag v-for="(count, key) in stats.resolutions" :key="key" class="tag" type="info">
              {{ key }} × {{ count }}
            </el-tag>
            <div v-if="!Object.keys(stats.resolutions).length" class="hint">暂无</div>
          </el-col>
          <el-col :span="8">
            <div class="stat-title">通道数</div>
            <el-tag v-for="(count, key) in stats.channels" :key="key" class="tag" type="info">
              {{ key }} 通道 × {{ count }}
            </el-tag>
            <div v-if="!Object.keys(stats.channels).length" class="hint">暂无</div>
          </el-col>
          <el-col :span="8">
            <div class="stat-title">文件格式</div>
            <el-tag v-for="(count, key) in stats.formats" :key="key" class="tag" type="info">
              {{ key }} × {{ count }}
            </el-tag>
            <div v-if="!Object.keys(stats.formats).length" class="hint">暂无</div>
          </el-col>
        </el-row>

        <el-divider content-position="left">抽样预览</el-divider>
        <div class="samples">
          <div v-for="item in samples" :key="item.relative_path" class="sample">
            <img :src="api.datasetSampleUrl(detailDataset.id, item.relative_path)" :alt="item.name" />
            <div class="hint">{{ item.width }}×{{ item.height }} · {{ sizeText(item.size_bytes) }}</div>
          </div>
          <div v-if="!samples.length" class="hint">没有可预览的图像</div>
        </div>

        <el-divider content-position="left">
          异常图像（{{ badFiles.length }}）
        </el-divider>
        <el-table v-if="badFiles.length" :data="badFiles" size="small" max-height="180">
          <el-table-column prop="path" label="文件" min-width="240" show-overflow-tooltip />
          <el-table-column prop="reason" label="原因" min-width="280" show-overflow-tooltip />
        </el-table>
        <el-alert v-else type="success" :closable="false" title="没有异常图像" />

        <el-divider content-position="left">图像清单</el-divider>
        <el-table :data="inventory?.items || []" size="small" max-height="380">
          <el-table-column prop="name" label="文件" min-width="200" show-overflow-tooltip />
          <el-table-column label="尺寸" width="110">
            <template #default="{ row }">{{ row.width }}×{{ row.height }}</template>
          </el-table-column>
          <el-table-column prop="channels" label="通道" width="80" />
          <el-table-column prop="format" label="格式" width="90" />
          <el-table-column label="大小" width="100">
            <template #default="{ row }">{{ sizeText(row.size_bytes) }}</template>
          </el-table-column>
          <el-table-column label="可读" width="80">
            <template #default="{ row }">
              <el-tag :type="row.ok ? 'success' : 'danger'" size="small">
                {{ row.ok ? '是' : '否' }}
              </el-tag>
            </template>
          </el-table-column>
        </el-table>
        <el-pagination
          v-model:current-page="page"
          :page-size="pageSize"
          :total="inventory?.total_items || 0"
          layout="prev, pager, next, total"
          style="margin-top: 10px"
          @current-change="loadInventory"
        />
      </div>
    </el-drawer>
  </el-card>
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
.inline-actions {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}
.stat-title {
  font-weight: 600;
  margin-bottom: 6px;
}
.tag {
  margin: 0 6px 6px 0;
}
.samples {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}
.sample img {
  width: 120px;
  height: 90px;
  object-fit: cover;
  border: 1px solid #e4e7ed;
  border-radius: 4px;
}
.hint {
  color: #909399;
  font-size: 12px;
}
.danger {
  color: #f56c6c;
}
</style>
