<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { api } from '@/api/client'

const route = useRoute()
const taskId = ref(route.params.taskId || '')
const artifacts = ref([])
const loaded = ref(false)

// 交付物 = zip 归档；其余是排障用的过程文件
const archive = computed(
  () =>
    artifacts.value.find((item) => item.kind === 'archive') ||
    artifacts.value.find((item) => item.relative_path?.endsWith('artifact.zip')) ||
    null,
)
const files = computed(() => artifacts.value.filter((item) => item !== archive.value))

function sizeText(item) {
  if (!item?.size_bytes) return '—'
  const mb = item.size_bytes / 1024 / 1024
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${(item.size_bytes / 1024).toFixed(1)} KB`
}

function download(item) {
  window.location.href = api.artifactDownloadUrl(taskId.value, item.id)
}

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
  <div v-loading="!loaded">
    <el-card shadow="never">
      <template #header>
        <div class="card-header">
          <span>任务产物</span>
          <el-input
            v-model="taskId"
            placeholder="输入任务 ID 查询产物"
            style="width: 340px"
            @keyup.enter="load"
          >
            <template #append>
              <el-button @click="load">查询</el-button>
            </template>
          </el-input>
        </div>
      </template>

      <template v-if="archive">
        <el-alert
          type="success"
          :closable="false"
          show-icon
          title="最终交付物就是这一份 zip：解压后在根目录运行 start.bat（Windows）或 ./start.sh（Linux/macOS），即可直接跑推理。"
          style="margin-bottom: 16px"
        />

        <div class="archive">
          <div class="archive-info">
            <div class="archive-name">📦 {{ archive.relative_path.split('/').pop() }}</div>
            <div class="hint">{{ archive.description }} · {{ sizeText(archive) }}</div>
            <div class="hint">SHA256：<span class="mono">{{ archive.sha256 || '—' }}</span></div>
          </div>
          <el-button type="primary" size="large" @click="download(archive)">
            下载完整产物（zip）
          </el-button>
        </div>

        <el-descriptions :column="1" border style="margin-top: 16px">
          <el-descriptions-item label="解压后的内容">
            <span class="mono">
              start.bat · start.sh · README.md · bin/ · model/model.engine · config/ · test/ ·
              source/ · docker/ · report/
            </span>
          </el-descriptions-item>
          <el-descriptions-item label="一键启动做了什么">
            找到 <span class="mono">bin/</span> 里的推理程序 → 用
            <span class="mono">model/model.engine</span> 加载 Engine → 对
            <span class="mono">test/sample.ppm</span> 跑若干次推理 → 结果写入
            <span class="mono">result.json</span>
          </el-descriptions-item>
          <el-descriptions-item label="运行前提">
            NVIDIA 驱动 + TensorRT 运行时（Windows 把 TensorRT 的
            <span class="mono">bin</span> 加入 PATH；Linux 把 <span class="mono">lib</span>
            加入 LD_LIBRARY_PATH）。不需要 OpenCV。
          </el-descriptions-item>
          <el-descriptions-item label="换机器怎么办">
            Engine 与构建平台绑定；不兼容时用 <span class="mono">source/</span> 重新编译，
            或在目标机重建 Engine（详见包内 README.md 第一、七节）。
          </el-descriptions-item>
        </el-descriptions>
      </template>

      <el-empty
        v-else-if="loaded"
        description="还没有产物（任务需要跑到 PACKAGING 阶段）"
      />
    </el-card>

    <el-collapse v-if="files.length" style="margin-top: 16px">
      <el-collapse-item :title="`逐个文件（${files.length} 项，排障用）`" name="files">
        <el-table :data="files" empty-text="暂无">
          <el-table-column prop="kind" label="类型" width="100" />
          <el-table-column prop="description" label="说明" min-width="220" />
          <el-table-column prop="relative_path" label="相对路径" min-width="320" show-overflow-tooltip />
          <el-table-column label="大小" width="110">
            <template #default="{ row }">{{ sizeText(row) }}</template>
          </el-table-column>
          <el-table-column label="下载" width="90">
            <template #default="{ row }">
              <a :href="api.artifactDownloadUrl(taskId, row.id)">下载</a>
            </template>
          </el-table-column>
        </el-table>
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.archive {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  background: #fafcff;
}
.archive-name {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 6px;
}
.hint {
  color: #909399;
  font-size: 12px;
  line-height: 1.7;
}
.mono {
  font-family: Consolas, 'Courier New', monospace;
  font-size: 12px;
}
</style>
