<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '@/api/client'

const projects = ref([])
const loading = ref(false)
const dialogVisible = ref(false)
const form = ref({ name: '', description: '' })

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

onMounted(load)
</script>

<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-header">
        <span>项目管理</span>
        <el-button type="primary" @click="dialogVisible = true">新建项目</el-button>
      </div>
    </template>

    <el-table v-loading="loading" :data="projects" empty-text="暂无项目">
      <el-table-column prop="name" label="项目名" min-width="160" />
      <el-table-column prop="description" label="描述" min-width="220" />
      <el-table-column prop="id" label="ID" min-width="260" />
      <el-table-column prop="created_at" label="创建时间" min-width="200" />
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
  </el-card>
</template>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
