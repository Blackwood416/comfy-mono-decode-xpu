# SPDX-License-Identifier: Apache-2.0
import os
import sys
from pathlib import Path

_DLL_DIR_HANDLES = []
_DLL_DIR_PATHS = set()


def _add_win_dll_dir(path: Path) -> None:
    if not path.is_dir():
        return
    resolved = path.resolve()
    if resolved in _DLL_DIR_PATHS:
        return
    _DLL_DIR_HANDLES.append(os.add_dll_directory(str(resolved)))
    _DLL_DIR_PATHS.add(resolved)


def _configure_windows_dll_dirs() -> None:
    if sys.platform != "win32":
        return
    for root in {Path(sys.prefix), Path(sys.executable).resolve().parent}:
        _add_win_dll_dir(root / "Library" / "bin")
        _add_win_dll_dir(root / "DLLs")
    try:
        import torch
        _add_win_dll_dir(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        pass
    for program_root in (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
    ):
        oneapi = program_root / "Intel" / "oneAPI"
        for version in ("latest", "2026.2"):
            for rel in (
                ("compiler", version, "bin"),
                ("compiler", version, "bin", "compiler"),
                ("tbb", version, "bin"),
            ):
                _add_win_dll_dir(oneapi.joinpath(*rel))


_configure_windows_dll_dirs()
from . import _C  # noqa: F401
