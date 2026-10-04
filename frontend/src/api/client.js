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
  // 删除项目：先取「影响面」给确认框看，再执行（或批量执行）
  previewProjectDeletion: (id) =>
    client.get(`/projects/${id}/deletion-preview`).then((r) => r.data),
  deleteProject: (id) => client.delete(`/projects/${id}`).then((r) => r.data),
  deleteProjects: (projectIds) =>
    client.post('/projects/delete', { project_ids: projectIds }).then((r) => r.data),

  listModels: (projectId) =>
    client.get('/models', { params: { project_id: projectId } }).then((r) => r.data),
  getModel: (id) => client.get(`/models/${id}`).then((r) => r.data),

  // 上传走 multipart：ONNX 模型与校准集 ZIP（服务端做大小/扩展名/ZIP 炸弹/路径穿越校验）
  uploadModel: (projectId, { name, architecture, file }) => {
    const form = new FormData()
    form.append('project_id', projectId)
    form.append('name', name)
    form.append('architecture', architecture || 'yolov8')
    form.append('file', file, file.name)
    return client.post('/models/upload', form).then((r) => r.data)
  },
  uploadDataset: (projectId, { name, kind, file }) => {
    const form = new FormData()
    form.append('project_id', projectId)
    form.append('name', name)
    form.append('kind', kind || 'calibration')
    form.append('file', file, file.name)
    return client.post('/datasets/upload', form).then((r) => r.data)
  },

  listDatasets: (projectId) =>
    client.get('/datasets', { params: { project_id: projectId } }).then((r) => r.data),

  listTasks: (params) => client.get('/tasks', { params }).then((r) => r.data),
  createTask: (payload) => client.post('/tasks', payload).then((r) => r.data),
  getTask: (id) => client.get(`/tasks/${id}`).then((r) => r.data),
  getTaskConfig: (id) => client.get(`/tasks/${id}/config`).then((r) => r.data),
  getTaskTransitions: (id) => client.get(`/tasks/${id}/transitions`).then((r) => r.data),
  getTaskLogs: (id) => client.get(`/tasks/${id}/logs`).then((r) => r.data),
  getTaskArtifacts: (id) => client.get(`/tasks/${id}/artifacts`).then((r) => r.data),
  getTaskReport: (id) => client.get(`/tasks/${id}/report`).then((r) => r.data),
  // 算子兼容性报告（SPEC 7.2）：任务未跑到模型校验阶段时后端返回 501
  getTaskCompatibility: (id) => client.get(`/tasks/${id}/compatibility`).then((r) => r.data),
  // 精度报告（SPEC 9.2）：端到端精度 + 分层误差分析；未完成 TESTING 阶段时 501
  getTaskPrecision: (id) => client.get(`/tasks/${id}/precision`).then((r) => r.data),
  artifactDownloadUrl: (taskId, artifactId) =>
    `/api/tasks/${taskId}/artifacts/${artifactId}/download`,
  enqueueTask: (id) => client.post(`/tasks/${id}/enqueue`).then((r) => r.data),
  cancelTask: (id) => client.post(`/tasks/${id}/cancel`).then((r) => r.data),
}

export default client
