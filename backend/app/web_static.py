"""前端静态托管（安装即用：运行期不需要 Node / Vite）。

设计要点：**不用 catch-all 路由**（那会遮蔽运行期新增的路由，实测会破坏「未处理异常信封」用例），
而是注册一个 404 处理器：

- 静态文件已构建 → 404 时先尝试按路径取真实文件，取不到就返回 `index.html`（SPA 深链接回退）；
- 未构建 → 返回一张说明页，明确告诉用户「后端是好的、前端还没构建、接口文档在哪」；
- `/api/...` 的 404 一律保持 JSON 语义，绝不被降级页伪装成 200。

复用 Starlette `StaticFiles` 的内容类型/条件请求处理，避免自己实现静态服务。
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

_FALLBACK_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>QForge 已启动</title>
  <style>
    body {{ font-family: system-ui, "Segoe UI", "Microsoft YaHei", sans-serif;
           margin: 0; padding: 48px; background: #0f172a; color: #e2e8f0; }}
    h1 {{ font-size: 22px; margin: 0 0 12px; }}
    code, pre {{ background: #1e293b; padding: 2px 6px; border-radius: 4px; }}
    pre {{ padding: 12px 16px; overflow-x: auto; }}
    a {{ color: #7dd3fc; }}
    .ok {{ color: #4ade80; }}
    li {{ margin: 6px 0; line-height: 1.6; }}
  </style>
</head>
<body>
  <h1>QForge 后端已启动 <span class="ok">●</span></h1>
  <p>接口正常，但<b>前端界面尚未构建</b>，因此这里显示的是说明页。</p>
  <ul>
    <li>API 文档：<a href="{api_prefix}/docs">{api_prefix}/docs</a> ·
        健康检查：<a href="{api_prefix}/health">{api_prefix}/health</a></li>
    <li>构建前端（需要 Node.js）：在仓库根执行 <code>qforge build-frontend</code></li>
    <li>只想用 API：上面的文档页可以直接调用全部接口，无需前端</li>
  </ul>
  <pre>qforge build-frontend   # npm ci &amp;&amp; npm run build
qforge serve --open     # 重新启动后会直接打开界面</pre>
</body>
</html>
"""


def register_frontend(app: FastAPI, directory: Path | None, *, api_prefix: str) -> None:
    """把前端（或降级说明页）挂到应用上。必须在 API 路由之后调用。"""
    api_root = "/" + api_prefix.strip("/")
    page = _FALLBACK_PAGE.format(api_prefix=api_prefix)
    static = (
        StaticFiles(directory=str(directory), html=True, check_dir=False)
        if directory is not None
        else None
    )
    if static is not None:
        logger.info("已托管前端静态文件：%s", directory)
    else:
        logger.info("未找到已构建前端，使用内置说明页（可用 qforge build-frontend 构建）")

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request: Request, exc: StarletteHTTPException) -> Response:
        path = request.url.path
        is_api = path == api_root or path.startswith(api_root + "/")
        if exc.status_code == 404 and request.method in {"GET", "HEAD"} and not is_api:
            if static is None:
                return HTMLResponse(page)
            # 先当真实文件取，取不到就交给 SPA（前端路由自己处理不存在的页面）
            for candidate in (path.lstrip("/"), "index.html"):
                try:
                    return await static.get_response(candidate or "index.html", request.scope)
                except StarletteHTTPException:
                    continue
            return HTMLResponse(page)
        return JSONResponse(
            {"detail": exc.detail},
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )
