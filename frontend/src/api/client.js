import axios from 'axios'

// 统一 API 客户端：错误一律带上后端返回的 error_code，便于界面与日志定位
const client = axios.create({ baseURL: '/api', timeout: 15000 })

client.interceptors.response.use(
  (response) => response,
  (error) => {
    const payload = error.response?.data
    const code = payload?.error_code || 'NETWORK_ERROR'
    const message = payload?.message || error.message
    return Promise.reject(Object.assign(new Error(`[${code}] ${message}`), { code, payload }))
  },
)

export const api = {
  health: () => client.get('/health').then((r) => r.data),

  listProjects: () => client.get('/projects').then((r) => r.data),
  createProject: (payload) => client.post('/projects', payload).then((r) => r.data),
  getProject: (id) => client.get(`/projects/${id}`).then((r) => r.data),

  listModels: (projectId) =>
    client.get('/models', { params: { project_id: projectId } }).then((r) => r.data),
  getModel: (id) => client.get(`/models/${id}`).then((r) => r.data),

  listDatasets: (projectId) =>
    client.get('/datasets', { params: { project_id: projectId } }).then((r) => r.data),

  listTasks: (params) => client.get('/tasks', { params }).then((r) => r.data),
  createTask: (payload) => client.post('/tasks', payload).then((r) => r.data),
  getTask: (id) => client.get(`/tasks/${id}`).then((r) => r.data),
  getTaskConfig: (id) => client.get(`/tasks/${id}/config`).then((r) => r.data),
  getTaskTransitions: (id) => client.get(`/tasks/${id}/transitions`).then((r) => r.data),
  getTaskLogs: (id) => client.get(`/tasks/${id}/logs`).then((r) => r.data),
  getTaskArtifacts: (id) => client.get(`/tasks/${id}/artifacts`).then((r) => r.data),
  enqueueTask: (id) => client.post(`/tasks/${id}/enqueue`).then((r) => r.data),
  cancelTask: (id) => client.post(`/tasks/${id}/cancel`).then((r) => r.data),
}

export default client
