<script setup>
import { onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '@/api/client'

const projects = ref([])
const models = ref([])
const datasets = ref([])
const projectId = ref('')

const modelForm = ref({ name: 'yolov8n', architecture: 'yolov8', file: null })
const datasetForm = ref({ name: 'calib-100', kind: 'calibration', file: null })
const modelResult = ref(null)
const datasetResult = ref(null)
const uploadingModel = ref(false)
const uploadingDataset = ref(false)

async function loadProjects() {
  try {
    projects.value = await api.listProjects()
  } catch (error) {
    ElMessage.error(error.message)
  }
}

async function loadAssets() {
  models.value = []
  datasets.value = []
  modelResult.value = null
  datasetResult.value = null
  if (!projectId.value) return
  try {
    models.value = await api.listModels(projectId.value)
    datasets.value = await api.listDatasets(projectId.value)
  } catch (error) {
    ElMessage.error(error.message)
  }
}

function pickFile(event, target) {
  const file = event.target.files?.[0] ?? null
  target.value = file
}

async function uploadModel() {
  if (!projectId.value || !modelForm.value.file) {
    ElMessage.warning('请选择项目与 ONNX 文件')
    return
  }
  uploadingModel.value = true
  try {
    modelResult.value = await api.uploadModel(projectId.value, modelForm.value)
    ElMessage.success('模型已上传并通过校验')
    await loadAssets()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    uploadingModel.value = false
  }
}

async function uploadDataset() {
  if (!projectId.value || !datasetForm.value.file) {
    ElMessage.warning('请选择项目与 ZIP 文件')
    return
  }
  uploadingDataset.value = true
  try {
    datasetResult.value = await api.uploadDataset(projectId.value, datasetForm.value)
    ElMessage.success(`校准集已上传，解压出 ${datasetResult.value.image_count} 张图像`)
    await loadAssets()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    uploadingDataset.value = false
  }
}

onMounted(loadProjects)
watch(projectId, loadAssets)
</script>

<template>
  <div>
    <el-card shadow="never">
      <template #header>选择项目</template>
      <el-select v-model="projectId" placeholder="先选择项目" style="width: 420px">
        <el-option v-for="item in projects" :key="item.id" :label="item.name" :value="item.id" />
      </el-select>
      <span class="hint">（没有项目请先到「项目管理」创建）</span>
    </el-card>

    <el-row :gutter="16" style="margin-top: 16px">
      <el-col :span="12">
        <el-card shadow="never">
          <template #header>上传 ONNX 模型</template>
          <el-form label-width="90px">
            <el-form-item label="模型名">
              <el-input v-model="modelForm.name" />
            </el-form-item>
            <el-form-item label="架构">
              <el-input v-model="modelForm.architecture" disabled />
            </el-form-item>
            <el-form-item label="ONNX 文件">
              <input type="file" accept=".onnx" @change="pickFile($event, modelForm.file)" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="uploadingModel" @click="uploadModel">
                上传并校验
              </el-button>
            </el-form-item>
          </el-form>

          <el-alert
            v-if="modelResult"
            type="success"
            :closable="false"
            :title="`模型 ${modelResult.name} 校验通过（opset ${modelResult.opset}）`"
          />
          <el-descriptions v-if="modelResult" :column="1" border style="margin-top: 8px">
            <el-descriptions-item label="输入">
              {{ modelResult.input_spec?.[0]?.name }} {{ modelResult.input_spec?.[0]?.shape }}
            </el-descriptions-item>
            <el-descriptions-item label="输出">
              {{ modelResult.output_spec?.[0]?.name }} {{ modelResult.output_spec?.[0]?.shape }}
            </el-descriptions-item>
            <el-descriptions-item label="大小">
              {{ ((modelResult.size_bytes || 0) / 1024).toFixed(1) }} KB
            </el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card shadow="never">
          <template #header>上传校准集（ZIP）</template>
          <el-form label-width="90px">
            <el-form-item label="数据集名">
              <el-input v-model="datasetForm.name" />
            </el-form-item>
            <el-form-item label="类型">
              <el-radio-group v-model="datasetForm.kind">
                <el-radio-button value="calibration">校准集</el-radio-button>
                <el-radio-button value="test">测试集</el-radio-button>
              </el-radio-group>
            </el-form-item>
            <el-form-item label="ZIP 文件">
              <input type="file" accept=".zip" @change="pickFile($event, datasetForm.file)" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="uploadingDataset" @click="uploadDataset">
                上传并解压
              </el-button>
            </el-form-item>
          </el-form>

          <el-alert
            v-if="datasetResult"
            type="success"
            :closable="false"
            :title="`解压出 ${datasetResult.image_count} 张图像（已做路径穿越与 ZIP 炸弹校验）`"
          />
          <div class="hint" style="margin-top: 8px">
            压缩包内部结构建议：<code>images/000001.jpg</code>。
            平台会校验每个条目的路径、数量、深度与解压后总大小，不安全的一律拒绝。
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="16" style="margin-top: 16px">
      <el-col :span="12">
        <el-card shadow="never">
          <template #header>本项目的模型</template>
          <el-table :data="models" empty-text="暂无模型" max-height="260">
            <el-table-column prop="name" label="名称" min-width="120" />
            <el-table-column prop="architecture" label="架构" min-width="100" />
            <el-table-column prop="opset" label="opset" width="80" />
          </el-table>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="never">
          <template #header>本项目的数据集</template>
          <el-table :data="datasets" empty-text="暂无数据集" max-height="260">
            <el-table-column prop="name" label="名称" min-width="120" />
            <el-table-column prop="kind" label="类型" min-width="100" />
            <el-table-column prop="image_count" label="图像数" width="90" />
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.hint {
  color: #909399;
  font-size: 12px;
}
</style>
