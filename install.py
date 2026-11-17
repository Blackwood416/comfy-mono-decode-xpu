# SPDX-License-Identifier: Apache-2.0
"""Build the extension in place: comfy_mono_decode_xpu/_C*.so.

ComfyUI-Manager runs this after cloning the node, with ComfyUI's Python, in this directory. It can also
be run by hand (``python install.py``) or through ``install.sh``.

  python install.py [--force] [--target DEVICE]

- Skips the build when the extension is newer than every source (``--force`` rebuilds).
- Needs the oneAPI ``icpx`` compiler. If it is not on PATH, ``ONEAPI_ROOT/setvars.sh`` (default
  /opt/intel/oneapi) is sourced for the build.
- Never fails the node install: without a compiler or XPU it prints what to do and exits 0. ComfyUI
  then starts normally and decode runs the stock path.
"""
import argparse
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "comfy_mono_decode_xpu", "_C" + (sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if sys.platform == "win32" else ".so")))
PREBUILT_REPO = os.environ.get("MONO_DECODE_XPU_PREBUILT_REPO", "Blackwood416/comfy-mono-decode-xpu")
PREBUILT_TAG = os.environ.get("MONO_DECODE_XPU_PREBUILT_TAG", "v0.1.0+torch215.dg2.1")


def log(msg):
    print(f"[mono_decode_xpu] {msg}", flush=True)


def up_to_date():
    if not os.path.exists(SO):
        return False
    builder = os.path.join(HERE, "build_dg2_win.bat" if sys.platform == "win32" else "build_icpx.sh")
    srcs = glob.glob(os.path.join(HERE, "csrc", "*")) + [builder]
    return os.path.getmtime(SO) >= max(os.path.getmtime(p) for p in srcs)


def build_command():
    if sys.platform == "win32":
        script = os.path.join(HERE, "build_dg2_win.bat")
        return ["cmd", "/c", script]
    script = os.path.join(HERE, "build_icpx.sh")
    if shutil.which("icpx"):
        return ["bash", script]
    setvars = os.path.join(os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi"), "setvars.sh")
    if os.path.exists(setvars):
        return ["bash", "-c", f'source "{setvars}" >/dev/null 2>&1; exec bash "{script}"']
    return None


def _is_dg2_device(torch_mod):
    try:
        props = torch_mod.xpu.get_device_properties(0)
        # DG2/Alchemist (A770/A750/A580/A380): 64 KB SLM and SG8 support; BMG/Xe2 has 128+ KB SLM and no SG8.
        sg_sizes = list(getattr(props, "sub_group_sizes", []))
        if sg_sizes and 8 not in sg_sizes:
            return False
        local_mem = int(getattr(props, "local_mem_size", 0) or 0)
        if local_mem > 65536:
            return False
        return True
    except Exception:
        return True


def _try_download_prebuilt(torch_mod, target):
    if sys.platform != "win32" or target != "dg2":
        return False
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
    if sys.version_info[:2] != (3, 13) or ext_suffix != ".cp313-win_amd64.pyd":
        log(f"no prebuilt binary for Python {sys.version_info.major}.{sys.version_info.minor} ({ext_suffix}); trying source build.")
        return False
    torch_ver = str(getattr(torch_mod, "__version__", ""))
    if not torch_ver.startswith("2.15."):
        log(f"no prebuilt binary for torch {torch_ver} (prebuilt targets 2.15+xpu); trying source build.")
        return False

    asset_name = f"_C{ext_suffix}"
    base_url = f"https://github.com/{PREBUILT_REPO}/releases/download/{PREBUILT_TAG}"
    asset_url = f"{base_url}/{asset_name}"
    sums_url = f"{base_url}/SHA256SUMS.txt"
    tmp_path = SO + ".tmp"
    try:
        expected_sha = None
        try:
            req = urllib.request.Request(sums_url, headers={"User-Agent": "comfy-mono-decode-xpu-installer"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                for line in resp.read().decode("utf-8", "replace").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[1].lstrip("*") == asset_name:
                        expected_sha = parts[0].lower()
                        break
        except Exception:
            pass

        log(f"downloading prebuilt DG2 binary from {asset_url} ...")
        req = urllib.request.Request(asset_url, headers={"User-Agent": "comfy-mono-decode-xpu-installer"})
        os.makedirs(os.path.dirname(SO), exist_ok=True)
        h = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as dst:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
                h.update(chunk)
        got_sha = h.hexdigest().lower()
        if expected_sha and got_sha != expected_sha:
            raise ValueError(f"SHA-256 mismatch: expected {expected_sha}, got {got_sha}")
        os.replace(tmp_path, SO)
        log(f"installed prebuilt {os.path.relpath(SO, HERE)} (sha256={got_sha[:12]})")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"prebuilt download failed ({e}); falling back to local build.")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="rebuild even if the extension is up to date")
    ap.add_argument("--target", help="AOT device for icpx (default dg2 on Windows, bmg on Linux)")
    ap.add_argument("--prebuilt", action="store_true", help="download prebuilt Windows DG2 binary even if present")
    ap.add_argument("--no-prebuilt", action="store_true", help="skip prebuilt binary download and build from source")
    args = ap.parse_args()

    try:
        import torch
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            log("torch has no usable XPU device: skipping the build (the node stays inactive).")
            return 0
        log(f"torch {torch.__version__} | {torch.xpu.get_device_name(0)}")
    except Exception as e:  # noqa: BLE001
        log(f"torch not importable ({e}): skipping the build.")
        return 0

    default_target = "dg2" if (sys.platform == "win32" and _is_dg2_device(torch)) else "bmg"
    target = args.target or os.environ.get("MONO_DECODE_XPU_XPU_DEVICE") or default_target
    use_prebuilt = (
        sys.platform == "win32"
        and not args.no_prebuilt
        and os.environ.get("MONO_DECODE_XPU_BUILD_FROM_SOURCE", "0") != "1"
        and (args.prebuilt or not args.force)
    )

    if up_to_date() and not args.force and not args.prebuilt:
        log(f"extension up to date: {os.path.relpath(SO, HERE)}")
    elif use_prebuilt and _try_download_prebuilt(torch, target):
        pass
    else:
        cmd = build_command()
        if cmd is None:
            log("icpx not found and no oneAPI setvars.sh: install the oneAPI DPC++ compiler, then rerun "
                "`python install.py` here. Until then decode runs the stock path.")
            return 0
        env = dict(os.environ, PYTHON=sys.executable)
        env["MONO_DECODE_XPU_XPU_DEVICE"] = target
        if sys.platform == "win32":
            env["TORCH_DIR"] = os.path.dirname(torch.__file__)
            env["EXT_SUFFIX"] = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
            env["PY_VER"] = f"{sys.version_info.major}{sys.version_info.minor}"
            py_inc = sysconfig.get_paths()["include"]
            if not os.path.exists(os.path.join(py_inc, "Python.h")):
                for cand in (
                    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", f"Python{env['PY_VER']}", "include"),
                    os.path.join(r"C:\Program Files", f"Python{env['PY_VER']}", "include"),
                ):
                    if os.path.exists(os.path.join(cand, "Python.h")):
                        py_inc = cand
                        break
            py_lib = os.path.join(sysconfig.get_config_var("installed_base") or "", "libs")
            if not os.path.exists(os.path.join(py_lib, f"python{env['PY_VER']}.lib")):
                for cand in (
                    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", f"Python{env['PY_VER']}", "libs"),
                    os.path.join(r"C:\Program Files", f"Python{env['PY_VER']}", "libs"),
                ):
                    if os.path.exists(os.path.join(cand, f"python{env['PY_VER']}.lib")):
                        py_lib = cand
                        break
            env.setdefault("MONO_PY_INCLUDE", py_inc)
            env.setdefault("MONO_PY_LIB", py_lib)
        log(f"building for {target} (single icpx invocation)")
        if subprocess.call(cmd, cwd=HERE, env=env) != 0:
            log("build failed (see the compiler output above). Decode runs the stock path.")
            return 0

    sys.path.insert(0, HERE)
    try:
        from comfy_mono_decode_xpu import _C
        ops = [n for n in dir(_C) if not n.startswith("_")]
        log(f"extension ok ({len(ops)} ops). Restart ComfyUI; it is enabled by default (MONO_DECODE_XPU=0 disables it).")
    except Exception as e:  # noqa: BLE001
        log(f"built extension failed to import: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
