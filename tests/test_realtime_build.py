"""实时推理（摄像头/推送）在平台侧的接线测试：OpenCV 探测、构建开关、交付 DLL 打包。

这些是**纯 CPU** 用例：不需要 OpenCV、不需要 GPU，用临时目录伪造布局即可。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.pipeline.stages import build as build_stage
from app.pipeline.stages import delivery as delivery_stage
from app.services import opencv_dev


def _fake_opencv(root: Path, *, version: str = "4.11.0", conda_layout: bool = False) -> Path:
    """造一个假的 OpenCV 目录树。conda 布局是 <root>/Library/{include,lib,bin}。"""
    base = root / "Library" if conda_layout else root
    include = base / "include" / "opencv2"
    (include / "core").mkdir(parents=True)
    (include / "opencv.hpp").write_text("// fake\n", encoding="utf-8")
    major, minor, patch = version.split(".")
    (include / "core" / "version.hpp").write_text(
        f"#define CV_VERSION_MAJOR {major}\n"
        f"#define CV_VERSION_MINOR {minor}\n"
        f"#define CV_VERSION_REVISION {patch}\n",
        encoding="utf-8",
    )
    lib = base / "lib"
    lib.mkdir(parents=True)
    (lib / "opencv_core4110.lib").write_bytes(b"fake")
    (base / "bin").mkdir(parents=True, exist_ok=True)
    return base


def test_probe_finds_official_layout(tmp_path: Path) -> None:
    base = _fake_opencv(tmp_path / "opencv")
    files = opencv_dev.probe(base)
    assert files is not None and files.complete
    assert files.version == "4.11.0"
    assert files.include_dir == base / "include"
    assert files.bin_dir == base / "bin"


def test_probe_returns_none_without_headers(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert opencv_dev.probe(tmp_path / "empty") is None


def test_locate_supports_conda_layout_under_search_root(tmp_path: Path, monkeypatch) -> None:
    """conda 的 <prefix>/Library 布局必须能被自动探测到（本机就是用 conda 装的）。"""
    _fake_opencv(tmp_path / "opencv-dev", conda_layout=True)
    settings = SimpleNamespace(opencv_root=None, toolchain_search_roots=[tmp_path])

    files = opencv_dev.locate(settings)  # type: ignore[arg-type]

    assert files.complete, files.problems
    assert files.root == tmp_path / "opencv-dev" / "Library"
    assert files.version == "4.11.0"


def test_locate_reports_problem_when_missing(tmp_path: Path) -> None:
    settings = SimpleNamespace(opencv_root=None, toolchain_search_roots=[tmp_path / "nothing"])
    files = opencv_dev.locate(settings)  # type: ignore[arg-type]
    assert not files.complete
    assert files.problems, "找不到时要给出可执行建议，而不是空着"


def test_locate_honours_explicit_root(tmp_path: Path) -> None:
    base = _fake_opencv(tmp_path / "opencv")
    settings = SimpleNamespace(opencv_root=base, toolchain_search_roots=[])
    files = opencv_dev.locate(settings)  # type: ignore[arg-type]
    assert files.complete and files.root == base


def test_camera_setup_off_by_default() -> None:
    context = SimpleNamespace(build_options={}, settings=SimpleNamespace())
    defines, status = build_stage._camera_setup(context)  # noqa: SLF001
    assert defines == {}
    assert status["status"] == "OFF"


def test_camera_setup_enables_opencv_when_available(tmp_path: Path, monkeypatch) -> None:
    base = _fake_opencv(tmp_path / "opencv")
    monkeypatch.setattr(
        build_stage.opencv_dev,
        "locate",
        lambda settings: opencv_dev.probe(base),
    )
    context = SimpleNamespace(build_options={"with_camera": True}, settings=SimpleNamespace())

    defines, status = build_stage._camera_setup(context)  # noqa: SLF001

    assert defines["QFORGE_WITH_OPENCV"] == "ON"
    # CMake 定义必须是正斜杠（反斜杠会被当成转义符）
    assert "\\" not in defines["QFORGE_OPENCV_ROOT"]
    assert defines["QFORGE_OPENCV_ROOT"].endswith("/opencv")
    assert status["status"] == "ENABLED"
    assert status["opencv"]["version"] == "4.11.0"


def test_camera_setup_blocked_without_opencv_but_not_fatal(tmp_path: Path, monkeypatch) -> None:
    """缺少 OpenCV 只标记 BLOCKED（任务照常编译），不能把整个任务判失败。"""
    monkeypatch.setattr(
        build_stage.opencv_dev,
        "locate",
        lambda settings: opencv_dev.OpenCvFiles(problems=["没装"]),
    )
    context = SimpleNamespace(build_options={"with_camera": True}, settings=SimpleNamespace())

    defines, status = build_stage._camera_setup(context)  # noqa: SLF001

    assert defines == {}
    assert status["status"] == "BLOCKED"
    assert status["problems"] == ["没装"]


def test_delivery_copies_only_needed_opencv_dlls(tmp_path: Path, monkeypatch) -> None:
    """只带程序真正用到的 OpenCV 模块（全带上会有几十个模块 + Qt6，太大）。

    依赖闭包由 dumpbin 解析，这里用假的依赖图替代（用例不依赖 VS 工具链）。
    """
    bin_dir = tmp_path / "opencv-bin"
    bin_dir.mkdir()
    for name in (
        "opencv_core500.dll",
        "opencv_imgproc500.dll",
        "opencv_videoio500.dll",
        "opencv_highgui500.dll",
        "opencv_bgsegm500.dll",  # 用不到，不该带
        "Qt6Core.dll",  # 只有 --display 才可能需要，不带
        "libopencv_core.so.500",  # Linux 产物
        "zlib1.dll",  # core 的依赖 → 必须带上
    ):
        (bin_dir / name).write_bytes(b"x")
    monkeypatch.setattr(
        delivery_stage,
        "_dumpbin_dependents",
        lambda path: ["zlib1.dll"] if "core" in path.name.lower() else [],
    )

    staging = tmp_path / "artifact"
    copied = delivery_stage._copy_opencv_runtime(staging, bin_dir)  # noqa: SLF001

    assert set(copied) == {
        "opencv_core500.dll",
        "opencv_imgproc500.dll",
        "opencv_videoio500.dll",
        "opencv_highgui500.dll",
        "libopencv_core.so.500",
        "zlib1.dll",
    }
    assert (staging / "bin" / "opencv_core500.dll").is_file()
    assert (staging / "bin" / "zlib1.dll").is_file(), "依赖 DLL 必须一起带上，否则解压后跑不起来"
    assert not (staging / "bin" / "Qt6Core.dll").exists()


def test_dumpbin_dependents_filters_system_dlls(tmp_path: Path, monkeypatch) -> None:
    """系统 DLL 与 api-ms-win-* 不能进交付包（Windows 自带），非 DLL 行要忽略。"""
    fake = tmp_path / "fake.dll"
    fake.write_bytes(b"x")

    class _Result:
        log_text = (
            "    KERNEL32.dll\n"
            "    api-ms-win-crt-runtime-l1-1-0.dll\n"
            "    zlib1.dll\n"
            "    VCRUNTIME140.dll\n"
            "    摘要\n"
        )

    from app.services import toolchain as toolchain_module

    monkeypatch.setattr(
        toolchain_module, "run_in_vs_env", lambda *a, **k: _Result(), raising=False
    )
    names = delivery_stage._dumpbin_dependents(fake)  # noqa: SLF001
    assert names == ["zlib1.dll"]


def test_delivery_opencv_runtime_is_noop_without_bin_dir(tmp_path: Path) -> None:
    assert delivery_stage._copy_opencv_runtime(tmp_path / "artifact", None) == []  # noqa: SLF001
