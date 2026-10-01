import { createRouter, createWebHistory } from 'vue-router'

// SPEC 16 页面清单；阶段 1 起项目管理 / 上传 / 新建任务 / 任务详情 / 产物为真实链路。
const routes = [
  { path: '/', redirect: '/projects' },
  { path: '/projects', name: 'projects', component: () => import('@/views/ProjectsView.vue') },
  { path: '/upload', name: 'upload', component: () => import('@/views/UploadView.vue') },
  { path: '/tasks/new', name: 'task-new', component: () => import('@/views/NewTaskView.vue') },
  { path: '/models/:id?', name: 'models', component: () => import('@/views/ModelDetailView.vue') },
  { path: '/tasks/:id?', name: 'tasks', component: () => import('@/views/TaskDetailView.vue') },
  { path: '/report', name: 'report', component: () => import('@/views/PrecisionReportView.vue') },
  {
    path: '/artifacts/:taskId?',
    name: 'artifacts',
    component: () => import('@/views/ArtifactsView.vue'),
  },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
