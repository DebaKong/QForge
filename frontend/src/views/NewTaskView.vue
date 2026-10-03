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
const enqueueNow = ref(true)
// 实时推理（摄像头/视频 + 结果推送）：构建带 OpenCV 的产物
const withCamera = ref(false)

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
      // 要摄像头/推送能力就必须真的有编译产物（zip 里的 bin/），所以一并提升为 required
      build: {
        cpp_build: withCamera.value ? 'required' : 'auto',
        with_camera: withCamera.value,
      },
    })
    ElMessage.success(`任务已创建：${task.id}`)
    if (enqueueNow.value) {
      await api.enqueueTask(task.id)
      ElMessage.info('已入队，任务正在后台执行（可看实时日志）')
    }
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
      title="阶段 1：创建任务后会自动跑完整流水线（校验 → 量化 → Engine → 生成 C++ 工程 → 编译 → 真实推理 → 打包）。"
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

      <el-form-item label="实时推理">
        <el-checkbox v-model="withCamera">需要摄像头/视频实时推理（构建 OpenCV 版产物）</el-checkbox>
        <div class="hint">
          勾选后：自动以 <span class="mono">build.with_camera=true</span> 编译（需要本机有 OpenCV
          开发文件，缺失只会提示不会让任务失败），产物的 zip 会带上 OpenCV 运行库，解压后用
          <span class="mono">start_camera.bat</span> 即可边采集边把检测结果推给别的系统。
        </div>
      </el-form-item>

      <el-form-item>
        <el-checkbox v-model="enqueueNow">创建后立即入队执行</el-checkbox>
      </el-form-item>

      <el-form-item>
        <el-button type="primary" :loading="submitting" :disabled="!form.project_id" @click="submit">
          创建任务
        </el-button>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<style scoped>
.hint {
  color: #909399;
  font-size: 12px;
  line-height: 1.7;
  margin-top: 4px;
}
.mono {
  font-family: Consolas, 'Courier New', monospace;
  font-size: 12px;
}
</style>
