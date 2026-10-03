<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api } from '@/api/client'

const projects = ref([])
const loading = ref(false)
const dialogVisible = ref(false)
const form = ref({ name: '', description: '' })

// 删除相关状态
const selected = ref([])
const deleteDialogVisible = ref(false)
const deleteTargets = ref([]) // 待删除的项目（单个=1 条，批量=多条）
const previews = ref([]) // 每个待删项目的影响面
const previewLoading = ref(false)
const deleting = ref(false)
const tableRef = ref(null)

const blocked = computed(() => previews.value.filter((item) => !item.can_delete))
const totals = computed(() =>
  previews.value.reduce(
    (sum, item) => ({
      models: sum.models + item.models,
      datasets: sum.datasets + item.datasets,
      tasks: sum.tasks + item.tasks,
      artifacts: sum.artifacts + item.artifacts,
      disk_bytes: sum.disk_bytes + item.disk_bytes,
    }),
    { models: 0, datasets: 0, tasks: 0, artifacts: 0, disk_bytes: 0 },
  ),
)

function sizeText(bytes) {
  if (!bytes) return '0 B'
  const mb = bytes / 1024 / 1024
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`
  if (mb >= 1) return `${mb.toFixed(1)} MB`
  return `${(bytes / 1024).toFixed(1)} KB`
}

async function load() {
  loading.value = true
  try {
    projects.value = await api.listProjects()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loading.value = false
  }
}

async function submit() {
  try {
    await api.createProject({ name: form.value.name, description: form.value.description || null })
    ElMessage.success('项目已创建')
    dialogVisible.value = false
    form.value = { name: '', description: '' }
    await load()
  } catch (error) {
    ElMessage.error(error.message)
  }
}

function onSelectionChange(rows) {
  selected.value = rows
}

/** 打开删除确认框：先取影响面，让用户看清楚会连带删掉什么 */
async function openDelete(rows) {
  deleteTargets.value = rows
  previews.value = []
  deleteDialogVisible.value = true
  previewLoading.value = true
  try {
    previews.value = await Promise.all(
      rows.map((row) =>
        api.previewProjectDeletion(row.id).catch((error) => ({
          project_id: row.id,
          project_name: row.name,
          can_delete: false,
          blocking_tasks: [],
          error: error.message,
          models: 0,
          datasets: 0,
          tasks: 0,
          artifacts: 0,
          disk_bytes: 0,
        })),
      ),
    )
  } finally {
    previewLoading.value = false
  }
}

/** 执行删除（不可恢复） */
async function confirmDelete() {
  if (blocked.value.length) return
  const names = deleteTargets.value.map((row) => row.name).join('、')
  try {
    await ElMessageBox.confirm(
      `确认删除 ${deleteTargets.value.length} 个项目（${names}）？该操作不可恢复。`,
      '再次确认',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return // 用户取消
  }

  deleting.value = true
  try {
    if (deleteTargets.value.length === 1) {
      const result = await api.deleteProject(deleteTargets.value[0].id)
      ElMessage.success(`已删除「${result.project_name}」，释放 ${sizeText(result.freed_bytes)}`)
    } else {
      const result = await api.deleteProjects(deleteTargets.value.map((row) => row.id))
      if (result.failed_count) {
        ElMessage.warning(
          `删除 ${result.deleted_count} 个，失败 ${result.failed_count} 个：` +
            result.failed.map((item) => item.reason).join('；'),
        )
      } else {
        ElMessage.success(
          `已删除 ${result.deleted_count} 个项目，共释放 ${sizeText(result.freed_bytes)}`,
        )
      }
    }
    deleteDialogVisible.value = false
    selected.value = []
    tableRef.value?.clearSelection()
    await load()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    deleting.value = false
  }
}

onMounted(load)
</script>

<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-header">
        <span>项目管理</span>
        <div class="actions">
          <el-button
            type="danger"
            plain
            :disabled="!selected.length"
            @click="openDelete(selected)"
          >
            删除选中（{{ selected.length }}）
          </el-button>
          <el-button type="primary" @click="dialogVisible = true">新建项目</el-button>
        </div>
      </div>
    </template>

    <el-table
      ref="tableRef"
      v-loading="loading"
      :data="projects"
      row-key="id"
      empty-text="暂无项目"
      @selection-change="onSelectionChange"
    >
      <el-table-column type="selection" width="46" reserve-selection />
      <el-table-column prop="name" label="项目名" min-width="160" />
      <el-table-column prop="description" label="描述" min-width="200" />
      <el-table-column prop="id" label="ID" min-width="240" />
      <el-table-column prop="created_at" label="创建时间" min-width="190" />
      <el-table-column label="操作" width="120" fixed="right">
        <template #default="{ row }">
          <el-button type="danger" link @click="openDelete([row])">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-dialog v-model="dialogVisible" title="新建项目" width="420px">
      <el-form label-width="70px">
        <el-form-item label="项目名">
          <el-input v-model="form.name" placeholder="唯一名称" />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="form.description" type="textarea" :rows="3" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :disabled="!form.name" @click="submit">创建</el-button>
      </template>
    </el-dialog>

    <!-- 删除确认：先把「会删掉什么」摆清楚，再让用户点 -->
    <el-dialog v-model="deleteDialogVisible" title="删除项目" width="620px">
      <div v-loading="previewLoading">
        <el-alert
          type="warning"
          :closable="false"
          show-icon
          title="删除不可恢复：会同时删掉这些项目下的模型、数据集、任务及其日志与产物文件。"
          style="margin-bottom: 12px"
        />

        <el-table :data="previews" size="small">
          <el-table-column prop="project_name" label="项目" min-width="150" />
          <el-table-column label="模型 / 数据集" width="120">
            <template #default="{ row }">{{ row.models }} / {{ row.datasets }}</template>
          </el-table-column>
          <el-table-column label="任务 / 产物" width="120">
            <template #default="{ row }">{{ row.tasks }} / {{ row.artifacts }}</template>
          </el-table-column>
          <el-table-column label="占用磁盘" width="110">
            <template #default="{ row }">{{ sizeText(row.disk_bytes) }}</template>
          </el-table-column>
          <el-table-column label="能否删除" min-width="200">
            <template #default="{ row }">
              <el-tag v-if="row.can_delete" type="success" size="small">可以删除</el-tag>
              <el-tag v-else type="danger" size="small">有运行中任务，已阻止</el-tag>
              <div v-if="!row.can_delete" class="hint">
                <template v-if="row.blocking_tasks.length">
                  请先取消：
                  <span
                    v-for="task in row.blocking_tasks"
                    :key="task.id"
                    class="mono"
                  >{{ task.id.slice(0, 8) }}({{ task.status }}) </span>
                </template>
                <template v-else>{{ row.error || '无法删除' }}</template>
              </div>
            </template>
          </el-table-column>
        </el-table>

        <el-descriptions v-if="previews.length" :column="2" border style="margin-top: 12px">
          <el-descriptions-item label="合计模型 / 数据集">
            {{ totals.models }} / {{ totals.datasets }}
          </el-descriptions-item>
          <el-descriptions-item label="合计任务 / 产物">
            {{ totals.tasks }} / {{ totals.artifacts }}
          </el-descriptions-item>
          <el-descriptions-item label="合计释放空间" :span="2">
            {{ sizeText(totals.disk_bytes) }}
          </el-descriptions-item>
        </el-descriptions>

        <el-alert
          v-if="blocked.length"
          type="error"
          :closable="false"
          show-icon
          style="margin-top: 12px"
          :title="`有 ${blocked.length} 个项目的任务仍排队/运行中：请先到「任务详情」取消任务，再回来删除。`"
        />
      </div>

      <template #footer>
        <el-button @click="deleteDialogVisible = false">取消</el-button>
        <el-button
          type="danger"
          :loading="deleting"
          :disabled="previewLoading || blocked.length > 0"
          @click="confirmDelete"
        >
          确认删除{{ deleteTargets.length > 1 ? `（${deleteTargets.length} 个）` : '' }}
        </el-button>
      </template>
    </el-dialog>
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
.hint {
  color: #909399;
  font-size: 12px;
  margin-top: 4px;
}
.mono {
  font-family: Consolas, 'Courier New', monospace;
  font-size: 12px;
}
</style>
