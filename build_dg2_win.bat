@echo off
REM SPDX-License-Identifier: Apache-2.0
REM Single-invocation icpx build of comfy_mono_decode_xpu\_C*.pyd for Windows Intel XPU (DG2 / BMG).
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not defined MONO_DECODE_XPU_XPU_DEVICE set "MONO_DECODE_XPU_XPU_DEVICE=dg2"
if not defined PYTHON set "PYTHON=python"

where icpx >nul 2>&1
if errorlevel 1 (
    for %%V in (
        "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
    ) do (
        if exist %%V (
            call %%V -arch=amd64 -host_arch=amd64 >nul
            goto :vs_ok
        )
    )
    :vs_ok
    if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (
        call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force >nul 2>&1
    )
)

where icpx >nul 2>&1
if errorlevel 1 (
    echo icpx not found on PATH after VS/oneAPI initialization. 1>&2
    exit /b 1
)

if not defined TORCH_DIR (
    for /f "usebackq delims=" %%I in (`"%PYTHON%" -c "import torch, os; print(os.path.dirname(torch.__file__))"`) do set "TORCH_DIR=%%I"
)
if not defined EXT_SUFFIX (
    for /f "usebackq delims=" %%I in (`"%PYTHON%" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.pyd')"`) do set "EXT_SUFFIX=%%I"
)
if not defined PY_VER (
    for /f "usebackq delims=" %%I in (`"%PYTHON%" -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"`) do set "PY_VER=%%I"
)

if defined MONO_PY_INCLUDE (
    set "PY_INC=%MONO_PY_INCLUDE%"
) else (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%PY_VER%\include\Python.h" set "PY_INC=%LOCALAPPDATA%\Programs\Python\Python%PY_VER%\include"
    if not defined PY_INC if exist "C:\Program Files\Python%PY_VER%\include\Python.h" set "PY_INC=C:\Program Files\Python%PY_VER%\include"
)

if defined MONO_PY_LIB (
    set "PY_LIB=%MONO_PY_LIB%"
) else (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%PY_VER%\libs\python%PY_VER%.lib" set "PY_LIB=%LOCALAPPDATA%\Programs\Python\Python%PY_VER%\libs"
    if not defined PY_LIB if exist "C:\Program Files\Python%PY_VER%\libs\python%PY_VER%.lib" set "PY_LIB=C:\Program Files\Python%PY_VER%\libs"
)

if not exist "%PY_INC%\Python.h" (
    echo Python.h not found. Set MONO_PY_INCLUDE to a Python %PY_VER% include directory. 1>&2
    exit /b 1
)
if not exist "%PY_LIB%\python%PY_VER%.lib" (
    echo python%PY_VER%.lib not found. Set MONO_PY_LIB to a Python %PY_VER% libs directory. 1>&2
    exit /b 1
)

set "SRCS=csrc\bindings.cpp csrc\w4a8_decode.sycl csrc\act_prep.sycl csrc\mono_mlp.sycl csrc\mono_layer.sycl csrc\mono_gdn.sycl csrc\attn_port.sycl csrc\attn_dpas.sycl csrc\fast_topk.sycl csrc\rope_kv.sycl csrc\mtp_loop.sycl csrc\draft_lmhead.sycl csrc\topk_rows.sycl csrc\draft_fused.sycl"
set "EXTRA_DEFS="
if /I "%MONO_DECODE_XPU_XPU_DEVICE%"=="dg2" (
    set "EXTRA_DEFS=-DMONO_DECODE_XPU_DG2=1"
)

set "OUT=comfy_mono_decode_xpu\_C%EXT_SUFFIX%"
set "PATH=%TORCH_DIR%\lib;%PATH%"
set "OBJDIR=%CD%\build_obj"
if not exist "%OBJDIR%" mkdir "%OBJDIR%"

echo Building %OUT% for %MONO_DECODE_XPU_XPU_DEVICE% (parallel compile + single AOT link) ...
"%PYTHON%" -c "import concurrent.futures, os, shlex, subprocess, sys; srcs = sys.argv[1].split(); objdir = sys.argv[2]; base = ['icpx', '-fsycl', '-std=c++20', '-O3', '-DNDEBUG', '-Wno-ignored-attributes', '-fsycl-targets=spir64_gen', '-I' + os.path.join(os.environ['TORCH_DIR'], 'include'), '-I' + os.path.join(os.environ['TORCH_DIR'], 'include', 'torch', 'csrc', 'api', 'include'), '-I' + os.environ['PY_INC'], '-Icsrc', '-DTORCH_EXTENSION_NAME=_C', '-DTORCH_API_INCLUDE_EXTENSION_H'] + shlex.split(os.environ.get('EXTRA_DEFS', '')) + ['-c', '-x', 'c++']; jobs = min(8, len(srcs)); compile_one = lambda s: (s, subprocess.run(base + [s, '-o', os.path.join(objdir, os.path.basename(s) + '.obj')])); ex = concurrent.futures.ThreadPoolExecutor(max_workers=jobs); futs = [ex.submit(compile_one, s) for s in srcs]; failed = [s for s, r in (f.result() for f in futs) if r.returncode]; sys.exit(1 if failed else 0)" "%SRCS%" "%OBJDIR%"
if errorlevel 1 exit /b %errorlevel%

set "OBJS="
for %%S in (%SRCS%) do set "OBJS=!OBJS! %OBJDIR%\%%~nxS.obj"

icpx -fsycl -std=c++20 -O3 -DNDEBUG -shared -Wno-ignored-attributes ^
  -fsycl-targets=spir64_gen -Xsycl-target-backend "-device %MONO_DECODE_XPU_XPU_DEVICE%" ^
  %OBJS% -o "%OUT%" ^
  -L"%TORCH_DIR%\lib" -L"%PY_LIB%" ^
  -ltorch -ltorch_python -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -lpython%PY_VER%
if errorlevel 1 exit /b %errorlevel%

echo built %OUT%
exit /b 0
