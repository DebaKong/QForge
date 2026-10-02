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


# --------------------------------------------------------------------------- #
# 依赖声明防漂移
# --------------------------------------------------------------------------- #
def test_dependency_pins_match_requirements() -> None:
    """pyproject 的运行期依赖必须与 backend/requirements.txt 的 pin 一致。

    两处声明是「安装即用」（pyproject）与「开发环境锁定」（requirements）的分工，
    一旦漂移就会出现「安装出来的版本 ≠ 测试过的版本」，必须挡住。
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies = \[(.*?)^\]", pyproject, re.S | re.M)
    assert block is not None, "pyproject.toml 未声明 dependencies"

    pins: dict[str, str] = {}
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
