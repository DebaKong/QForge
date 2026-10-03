<script setup>
import { onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '@/api/client'

const route = useRoute()
const health = ref(null)
const healthError = ref('')

onMounted(async () => {
  try {
    health.value = await api.health()
  } catch (error) {
    healthError.value = error.message
  }
})

const menu = [
  { index: '/projects', label: '项目管理' },
  { index: '/upload', label: '上传模型与数据' },
  { index: '/tasks/new', label: '新建任务' },
  { index: '/models', label: '模型详情' },
  { index: '/tasks', label: '任务详情' },
  { index: '/report', label: '精度报告' },
  { index: '/artifacts', label: '产物页面' },
]
</script>

<template>
  <el-container class="app-shell">
    <el-aside width="220px" class="app-aside">
      <div class="brand">
        <div class="brand-title">QForge</div>
        <div class="brand-sub">ONNX 量化部署平台</div>
      </div>
      <el-menu :default-active="route.path" router>
        <el-menu-item v-for="item in menu" :key="item.index" :index="item.index">
          {{ item.label }}
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header class="app-header">
        <span class="phase-tag">阶段 1：MVP（2D 检测 + TensorRT）</span>
        <span v-if="health" class="health-ok">
          后端正常 · {{ health.environment }} · {{ health.database }} · Celery eager:
          {{ health.celery_eager }}
        </span>
        <span v-else-if="healthError" class="health-bad">后端不可用：{{ healthError }}</span>
        <span v-else class="health-unknown">检测后端中…</span>
      </el-header>

      <el-main>
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<style>
body {
  margin: 0;
  font-family: 'Microsoft YaHei', system-ui, sans-serif;
  background: #f5f7fa;
}
.app-shell {
  min-height: 100vh;
}
.app-aside {
  background: #1f2d3d;
  color: #fff;
}
.brand {
  padding: 18px 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}
.brand-title {
  font-size: 20px;
  font-weight: 600;
  letter-spacing: 0.5px;
}
.brand-sub {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.65);
  margin-top: 4px;
}
/* 侧边栏是深色，但 Element Plus 的菜单默认用深色文字 —— 深色叠深色就看不清了。
   用官方 CSS 变量把菜单改成浅色文字（这是"左侧边栏文字不清楚"的原因）。 */
.app-aside .el-menu {
  background: transparent;
  border-right: none;
  --el-menu-bg-color: transparent;
  --el-menu-text-color: rgba(255, 255, 255, 0.86);
  --el-menu-hover-text-color: #ffffff;
  --el-menu-hover-bg-color: rgba(255, 255, 255, 0.10);
  --el-menu-active-color: #ffffff;
}
.app-aside .el-menu-item {
  position: relative;
  height: 44px;
  line-height: 44px;
  margin: 3px 8px;
  border-radius: 6px;
}
.app-aside .el-menu-item.is-active {
  background: rgba(64, 158, 255, 0.24);
  font-weight: 600;
}
.app-aside .el-menu-item.is-active::before {
  content: '';
  position: absolute;
  left: 0;
  top: 9px;
  bottom: 9px;
  width: 3px;
  border-radius: 0 3px 3px 0;
  background: #409eff;
}
.app-header {
  display: flex;
  align-items: center;
  gap: 12px;
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
  font-size: 13px;
}
.el-main {
  padding: 16px 20px;
}
.phase-tag {
  background: #ecf5ff;
  color: #409eff;
  padding: 2px 8px;
  border-radius: 4px;
}
.health-ok {
  color: #67c23a;
}
.health-bad {
  color: #f56c6c;
}
.health-unknown {
  color: #909399;
}
</style>
