<script setup>
import { onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { api } from '@/api/client'

const router = useRouter()
const projects = ref([])
const models = ref([])
const datasets = ref([])
const submitting = ref(false)

const form = ref({
  project_id: '',
  model_id: '',
  dataset_id: '',
  task_type: 'detection',
  precision: 'fp16',
  backend_name: 'tensorrt',
})

onMounted(async () => {
  try {
    projects.value = await api.listProjects()
  } catch (error) {
    ElMessage.error(error.message)
  }
})

watch(
  () => form.value.project_id,
  async (projectId) => {
    form.value.model_id = ''
    form.value.dataset_id = ''
    models.value = []
    datasets.value = []
    if (!projectId) return
    try {
      models.value = await api.listModels(projectId)
      datasets.value = await api.listDatasets(projectId)
    } catch (error) {
      ElMessage.error(error.message)
    }
  },
)

async function submit() {
  submitting.value = true
  try {
    const task = await api.createTask({
      ...form.value,
      model_id: form.value.model_id || null,
      dataset_id: form.value.dataset_id || null,
    })
    ElMessage.success(`任务已创建：${task.id}`)
    router.push(`/tasks/${task.id}`)
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <el-card shadow="never">
    <template #header>新建任务</template>

    <el-alert
      type="info"
      :closable="false"
      title="阶段 0 只登记任务与配置快照；文件上传、模型校验与量化流水线在阶段 1 落地。"
      style="margin-bottom: 16px"
    />

    <el-form label-width="120px" style="max-width: 620px">
      <el-form-item label="项目">
        <el-select v-model="form.project_id" placeholder="选择项目" style="width: 100%">
          <el-option v-for="item in projects" :key="item.id" :label="item.name" :value="item.id" />
        </el-select>
      </el-form-item>

      <el-form-item label="ONNX 模型">
        <el-select v-model="form.model_id" placeholder="选择模型" clearable style="width: 100%">
          <el-option v-for="item in models" :key="item.id" :label="item.name" :value="item.id" />
        </el-select>
      </el-form-item>

      <el-form-item label="校准数据集">
        <el-select v-model="form.dataset_id" placeholder="选择数据集" clearable style="width: 100%">
          <el-option v-for="item in datasets" :key="item.id" :label="item.name" :value="item.id" />
        </el-select>
      </el-form-item>

      <el-form-item label="任务类型">
        <el-input v-model="form.task_type" disabled />
      </el-form-item>

      <el-form-item label="精度模式">
        <el-radio-group v-model="form.precision">
          <el-radio-button value="fp32">FP32 基准</el-radio-button>
          <el-radio-button value="fp16">FP16 半精度</el-radio-button>
          <el-radio-button value="int8">INT8 静态量化</el-radio-button>
        </el-radio-group>
      </el-form-item>

      <el-form-item label="推理后端">
        <el-input v-model="form.backend_name" disabled />
      </el-form-item>

      <el-form-item>
        <el-button type="primary" :loading="submitting" :disabled="!form.project_id" @click="submit">
          创建任务
        </el-button>
      </el-form-item>
    </el-form>
  </el-card>
</template>
