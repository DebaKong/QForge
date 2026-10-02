"""`qforge` 命令与「前端由 API 托管」的用例。

对应 AGENTS.md 的「产品形态要求」：安装即用、一条命令启动。
这些用例只验证**行为契约**（命令能跑、路径对、降级正确），不写死某台机器的环境结论。
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.cli import main
from app.web_static import register_frontend

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_version_prints_key_paths(storage_root: Path, capsys) -> None:
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert "QForge" in out
    assert str(storage_root) in out
    assert "执行方式" in out


def test_doctor_json_is_machine_readable(storage_root: Path, capsys) -> None:
    code = main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)  # --json 只输出 JSON，便于脚本消费

    names = [item["name"] for item in payload["checks"]]
    assert "Python" in names
    assert "数据目录" in names
    assert "GPU" in names
    assert "TensorRT 运行库" in names
    # 本机是否有 GPU/Docker 不应影响用例：只断言结构与执行方式取值
    assert payload["executor_mode"] in {"local", "celery"}
    assert payload["storage_root"] == str(storage_root)
    assert isinstance(payload["ok"], bool)
    assert code in (0, 1)


def test_doctor_human_report_has_conclusion(storage_root: Path, capsys) -> None:
    code = main(["doctor"])
    out = capsys.readouterr().out
    assert "环境体检" in out
    assert "结论：" in out
    assert "执行方式：" in out
    assert code in (0, 1)


def test_init_creates_database_and_tables(storage_root: Path, capsys) -> None:
    from app.db import base as db_base

    db_base.reset_engine()  # 避免上一个用例残留的引擎指向别的临时库
    try:
        assert main(["init"]) == 0
    finally:
        db_base.reset_engine()

    out = capsys.readouterr().out
    assert "数据目录" in out

    database = storage_root / "qforge.db"
    assert database.exists(), "init 必须真的建库"
    tables = {
        row[0]
        for row in sqlite3.connect(str(database)).execute(
            "select name from sqlite_master where type='table'"
        )
    }
    assert {"projects", "models", "datasets", "tasks", "artifacts", "job_logs"} <= tables


def test_unknown_command_prints_help(capsys) -> None:
    assert main([]) == 0
    assert "qforge" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 前端静态托管
# --------------------------------------------------------------------------- #
def test_frontend_serves_built_dist_with_spa_fallback(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>QF-INDEX</html>", encoding="utf-8")
    (dist / "app.js").write_text("console.log(1)", encoding="utf-8")

    app = FastAPI()
    register_frontend(app, dist, api_prefix="/api")
    client = TestClient(app)

    assert "QF-INDEX" in client.get("/").text
    assert client.get("/app.js").status_code == 200
    # SPA 深链接必须回退到 index.html，而不是 404
    assert "QF-INDEX" in client.get("/tasks/abc123").text


def test_frontend_falls_back_to_explanation_page(tmp_path: Path) -> None:
    app = FastAPI()
    register_frontend(app, None, api_prefix="/api")
    client = TestClient(app)

    page = client.get("/")
    assert page.status_code == 200
    assert "QForge" in page.text
    assert "/api/docs" in page.text
    # 未匹配的接口路径仍必须是 404，不能被降级页伪装成 200
    assert client.get("/api/does-not-exist").status_code == 404


def test_app_root_is_never_404(client: TestClient) -> None:
    """真实应用下根路径必须可访问（有前端给界面，没有给说明页）。"""
    response = client.get("/")
    assert response.status_code == 200
    assert client.get("/api/health").status_code == 200
    # 接口文档与 OpenAPI 也必须在 /api 命名空间下可达（横幅里承诺的地址要真的能开）
    assert client.get("/api/docs").status_code == 200
    assert client.get("/api/openapi.json").status_code == 200


def test_doctor_frontend_check_honours_setting(monkeypatch, tmp_path: Path, capsys) -> None:
    """doctor 必须读 QFORGE_FRONTEND_DIR。

    容器镜像里前端在 /opt/qforge/web（镜像内路径），若只按「数据目录旁的 web/」猜，
    就会误报「未构建」——实测在容器里就是这样误报过一次。
    """
    from app.config.settings import get_settings

    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    monkeypatch.setenv("QFORGE_FRONTEND_DIR", str(web))
    get_settings.cache_clear()
    try:
        main(["doctor", "--json"])
        payload = json.loads(capsys.readouterr().out)
    finally:
        get_settings.cache_clear()

    frontend = next(item for item in payload["checks"] if item["name"] == "前端界面")
    assert frontend["status"] == "通过", frontend
    assert str(web) in frontend["detail"]


# --------------------------------------------------------------------------- #
# CUDA 开发文件自动获取（公开源）
# --------------------------------------------------------------------------- #
def test_cuda_redist_plan_selects_platform_entries() -> None:
    from app.services import cuda_redist

    manifest = {
        # 实测形态：平台项是字典，且 size 是字符串
        "cuda_cudart": {
            "windows-x86_64": {"relative_path": "cuda_cudart/windows-x86_64/x.zip", "size": "1048576"},
            "linux-x86_64": {"relative_path": "cuda_cudart/linux-x86_64/x.tar.xz", "size": "1048576"},
        },
        # 兼容形态：平台项是列表
        "cuda_crt": {"windows-x86_64": [{"relative_path": "cuda_crt/windows-x86_64/y.zip"}]},
    }

    planned = cuda_redist.plan_components(manifest, key="windows-x86_64")
    assert [item.name for item in planned] == ["cuda_cudart", "cuda_crt"]
    assert planned[0].filename == "x.zip"
    assert planned[0].archive_kind == "zip"
    assert planned[0].size_bytes == 1048576  # 字符串 size 必须被转成整数
    assert planned[0].url.startswith("https://developer.download.nvidia.com/")

    linux = cuda_redist.plan_components(manifest, key="linux-x86_64")
    assert [item.archive_kind for item in linux] == ["tar"]


def test_cuda_redist_rejects_path_traversal(tmp_path: Path) -> None:
    """解压前必须过滤越界成员（AGENTS.md：一律防路径穿越）。"""
    from app.services.cuda_redist import _safe_members

    safe = _safe_members(["include/ok.h", "../evil.h", "a/../../b.h"], tmp_path)
    assert safe == ["include/ok.h"]


def test_cuda_fetch_dry_run_does_not_download(monkeypatch, capsys, tmp_path: Path) -> None:
    from app.services import cuda_redist

    manifest = {
        "cuda_cudart": {
            "windows-x86_64": {
                "relative_path": "cuda_cudart/windows-x86_64/x.zip",
                "size": 1048576,
            }
        }
    }
    monkeypatch.setattr(cuda_redist, "fetch_manifest", lambda *a, **k: manifest)

    code = main(
        ["fetch-cuda-headers", "--dry-run", "--platform", "windows-x86_64", "--dest", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "待下载" in out
    assert "cuda_cudart" in out
    assert not list(tmp_path.rglob("*.zip")), "--dry-run 不应下载任何文件"


def test_build_frontend_publish_copies_dist(monkeypatch, tmp_path: Path, capsys) -> None:
    """安装形态下 `--source ... --publish` 必须把构建产物发布到 <数据目录>/../web。

    实测背景：安装形态里 `app.paths.project_root()` 返回 None（包在 site-packages），
    因此不能靠自动探测找前端源码，必须由安装脚本显式传入。
    """
    import types

    source = tmp_path / "frontend"
    (source / "dist").mkdir(parents=True)
    (source / "dist" / "index.html").write_text("<html>PUBLISHED</html>", encoding="utf-8")
    (source / "package.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("QFORGE_DATA_DIR", str(tmp_path / "data" / "storage"))
    monkeypatch.setattr("app.cli.shutil.which", lambda name: "npm" if name == "npm" else None)
    monkeypatch.setattr(
        "app.cli.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=0),
    )

    code = main(["build-frontend", "--source", str(source), "--publish"])
    assert code == 0, capsys.readouterr().out
    published = tmp_path / "data" / "web" / "index.html"
    assert published.is_file(), "构建产物必须被发布到 <数据目录>/../web"
    assert "PUBLISHED" in published.read_text(encoding="utf-8")


def test_build_frontend_without_source_reports_how(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("app.cli.project_root", lambda: None)
    code = main(["build-frontend"])
    out = capsys.readouterr().out
    assert code == 1
    assert "--source" in out, "必须告诉用户怎么指定前端源码目录"


# --------------------------------------------------------------------------- #
# 打包与依赖边界（决定容器镜像多大）
# --------------------------------------------------------------------------- #
def test_pyproject_declares_gpu_extra() -> None:
    """GPU 依赖必须待在可选额外项 `gpu` 里，而不是核心 dependencies。

    否则 api 镜像会被迫装上约 3.7 GB 的 TensorRT 运行库（实测镜像从 0.98 GB 涨到 9.3 GB）。
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    core = re.search(r"^dependencies = \[(.*?)^\]", pyproject, re.S | re.M)
    extras = re.search(r"^\[project\.optional-dependencies\](.*?)^\[", pyproject, re.S | re.M)
    assert core is not None and extras is not None

    for package in ("tensorrt", "cuda-python"):
        assert f'"{package}==' not in core.group(1), f"{package} 不应出现在核心 dependencies 里"
        assert f'"{package}==' in extras.group(1), f"{package} 应声明在 gpu 额外项里"


def test_dockerfile_keeps_api_image_free_of_gpu_dependencies() -> None:
    """Dockerfile 的两个目标必须保持依赖边界：api 不装 gpu 额外项，worker 才装。"""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM core AS api" in dockerfile
    assert "FROM core AS worker" in dockerfile

    api_block = dockerfile.split("FROM core AS api", 1)[1].split("FROM core AS worker", 1)[0]
    worker_block = dockerfile.split("FROM core AS worker", 1)[1]

    assert "pip install ." in api_block
    assert "[gpu]" not in api_block, "api 目标不应安装 GPU 额外项"
    assert 'pip install ".[gpu]"' in worker_block, "worker 目标必须装 GPU 额外项"


# --------------------------------------------------------------------------- #
# 脚本编码（真实踩过的坑）
# --------------------------------------------------------------------------- #
def test_powershell_scripts_are_utf8_with_bom() -> None:
    """含中文的 .ps1 必须带 UTF-8 BOM。

    实测：Windows PowerShell 5.1 会把无 BOM 的 UTF-8 脚本按系统 ANSI（GBK）读取，
    中文被打乱后会**直接导致语法错误**（install.ps1 首次运行就是这样失败的）。
    """
    scripts = sorted((REPO_ROOT / "scripts").glob("*.ps1"))
    assert scripts, "scripts/ 下应当有 PowerShell 脚本"
    for path in scripts:
        raw = path.read_bytes()
        body = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
        if body.isascii():
            continue
        assert raw.startswith(b"\xef\xbb\xbf"), f"{path.name} 含非 ASCII 字符但缺少 UTF-8 BOM"


# --------------------------------------------------------------------------- #
# 依赖声明防漂移
# --------------------------------------------------------------------------- #
def test_dependency_pins_match_requirements() -> None:
    """pyproject 的运行期依赖（含可选额外项）必须与 backend/requirements.txt 的 pin 一致。

    两处声明是「安装即用」（pyproject）与「开发环境锁定」（requirements）的分工，
    一旦漂移就会出现「安装出来的版本 ≠ 测试过的版本」，必须挡住。
    注意：GPU 依赖在 pyproject 里属于可选额外项 `gpu`，因此比较时要把额外项也算进来。
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    blocks = [
        re.search(r"^dependencies = \[(.*?)^\]", pyproject, re.S | re.M),
        re.search(r"^\[project\.optional-dependencies\](.*?)^\[", pyproject, re.S | re.M),
    ]
    assert all(block is not None for block in blocks), "pyproject.toml 未声明 dependencies/额外项"

    pins: dict[str, str] = {}
    for block in blocks:
        for match in re.finditer(r'"([A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?==([^"]+)"', block.group(1)):
            pins[match.group(1).lower()] = match.group(2)

    requirements: dict[str, str] = {}
    for raw in (REPO_ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        match = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?==(.+)$", line)
        if match:
            requirements[match.group(1).lower()] = match.group(2)

    assert requirements, "未能解析 backend/requirements.txt"
    drift = {name: (version, pins.get(name)) for name, version in requirements.items() if pins.get(name) != version}
    assert not drift, f"依赖声明漂移（requirements → pyproject）：{drift}"
