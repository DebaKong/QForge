// QForge 界面回归检查（真实浏览器点击，覆盖纯前端才能发现的问题）
//
// 为什么需要它：后端 API 测试全绿也可能界面点不动（例：上传按钮没反应、列表点「查看」
// 不跳转）。这类问题只有真的点一遍才发现。
//
// 依赖：Playwright（**只在跑这个脚本时才需要**，不是应用依赖）
//     npm install --no-save playwright          # 或装在任意临时目录
// 浏览器：默认用系统已安装的 Edge（channel=msedge），因此**不需要下载 chromium**；
//     想用别的浏览器就设 QFORGE_BROWSER_CHANNEL=chrome / QFORGE_BROWSER_CHANNEL=chromium。
//
// 用法：
//     node tools/ui_check.mjs <要上传的 .onnx 路径> [http://127.0.0.1:8000]
//
// 覆盖项：
//     1) 模型列表能打开
//     2) 点「查看」自动进入详情页（路由参数变化要能触发重新加载，不能要求手动刷新）
//     3) 详情页显示真实解析结果（输入/输出）
//     4) 浏览器返回自动回到列表
//     5) 选择项目 + 选择 ONNX 文件 + 点上传 能真的成功
import { basename } from 'node:path'
import { chromium } from 'playwright'

const onnxPath = process.argv[2]
const BASE = process.argv[3] || 'http://127.0.0.1:8000'
if (!onnxPath) {
  console.error('用法： node tools/ui_check.mjs <要上传的 .onnx 路径> [base-url]')
  process.exit(2)
}

const problems = []
const channel = process.env.QFORGE_BROWSER_CHANNEL || 'msedge'
const browser = await chromium.launch({ channel })
const page = await browser.newPage()
page.on('console', (msg) => {
  if (msg.type() === 'error') problems.push(`console: ${msg.text()}`)
})
page.on('pageerror', (err) => problems.push(`pageerror: ${err}`))

async function step(name, fn) {
  try {
    await fn()
    console.log(`[通过] ${name}`)
  } catch (error) {
    const line = error.message.split('\n')[0]
    console.log(`[失败] ${name} -> ${line}`)
    problems.push(`${name}: ${line}`)
  }
}

await step('打开模型列表', async () => {
  await page.goto(`${BASE}/models`, { waitUntil: 'networkidle' })
  await page.waitForSelector('text=模型列表', { timeout: 10000 })
})

await step('点「查看」自动进入详情页（无需手动刷新）', async () => {
  await page.locator('a:has-text("查看")').first().click()
  await page.waitForURL(/\/models\/[0-9a-f]{8,}/, { timeout: 10000 })
  await page.waitForSelector('text=模型详情', { timeout: 10000 })
})

await step('详情页显示真实解析结果（输入/输出）', async () => {
  await page.waitForSelector('text=模型解析结果', { timeout: 10000 })
})

await step('浏览器返回自动回到列表（无需手动刷新）', async () => {
  await page.goBack()
  await page.waitForSelector('text=模型列表', { timeout: 10000 })
})

await step('上传页选项目 + 选文件 + 上传成功', async () => {
  await page.goto(`${BASE}/upload`, { waitUntil: 'networkidle' })
  await page.locator('.el-select').first().click()
  await page.locator('.el-select-dropdown__item').first().click()
  await page.setInputFiles('input[type=file][accept=".onnx"]', onnxPath)
  await page.waitForSelector(`text=${basename(onnxPath)}`, { timeout: 5000 }) // 选中的文件名要回显
  await page.locator('button:has-text("上传并校验")').click()
  await page.waitForSelector('text=模型已上传并通过校验', { timeout: 60000 })
})

console.log(
  problems.length
    ? `\n共 ${problems.length} 个问题：\n- ${problems.join('\n- ')}`
    : '\n全部通过，无控制台错误。',
)
await browser.close()
process.exit(problems.length ? 1 : 0)
