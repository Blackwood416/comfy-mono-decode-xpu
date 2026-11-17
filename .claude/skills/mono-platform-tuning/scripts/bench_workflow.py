"""Run a saved ComfyUI workflow on fresh servers, one per switch configuration, and collect decode numbers.

usage:
  python bench_workflow.py --comfyui DIR --workflow API.json --out DIR \
      --config stock:MONO_DECODE_XPU=0 --config default: [--config int4:MONO_DECODE_XPU_DRAFT_INT4=1] \
      [--warmup 2] [--runs 3] [--seed-base 100] [--pe-only] [--mtp tree4s1m8] [--max-length N] \
      [--launcher "CMD ..."] [--server-args "..."] [--extra-model-paths FILE] [--port 8288]

- API.json: the workflow in API format (frontend: Export (API)), or a saved /prompt request body
  with a "prompt" key. Keep the official template's models and settings.
- Each configuration starts a new server with only its MONO_DECODE_XPU* variables set (inherited
  ones are removed; a launcher script may still set its own defaults: check `env` in the records).
  Run i sets the prompt-enhance sampling seed to seed_base + i; nothing else changes, so prompt
  enhance and everything after it re-execute while the loaders stay cached.
- Run 0 is the first image after start; it is reported separately from the later runs, because the
  memory state differs (for example whether the tile-major weight copy fits).
- --pe-only keeps only the prompt-enhance node and what feeds it, plus a PreviewAny of its text.
- --mtp pins the MTP tree (for example tree4s1m8) so configurations do identical work per step;
  under `auto` the tree follows each configuration's measured speed.
Writes OUT/<config>/{server.log,gens.jsonl,run-<i>.json,text-<i>.txt} and OUT/summary.md.
"""
import argparse
import copy
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
PE_CLASSES = ("TextGenerate", "TextGenerateV2")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost: never via a proxy


def load_prompt(path):
    data = json.loads(Path(path).read_text())
    data = data.get("prompt", data)
    if not all(isinstance(v, dict) and "class_type" in v for v in data.values()):
        sys.exit(f"{path}: not an API-format workflow (export it with Export (API))")
    return data


def pe_nodes(prompt):
    return [nid for nid, n in prompt.items() if n["class_type"] in PE_CLASSES]


def closure(prompt, root):
    keep, todo = set(), [root]
    while todo:
        nid = todo.pop()
        if nid in keep:
            continue
        keep.add(nid)
        for v in prompt[nid]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in prompt:
                todo.append(v[0])
    return keep


def set_seed(node, seed):
    inputs = node["inputs"]
    for key in ("sampling_mode.seed", "seed"):
        if key in inputs:
            inputs[key] = seed
            return key
    return None


def api(base, path, data=None):
    req = urllib.request.Request(base + path, None if data is None else json.dumps(data).encode(),
                                 {"Content-Type": "application/json"})
    with OPENER.open(req, timeout=60) as r:
        return json.load(r)


def run_config(a, name, env_spec, prompt, pe_id):
    root = Path(a.out).resolve() / name  # the server runs in the ComfyUI directory
    for d in ("out", "user/__manager"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "user/__manager/config.ini").write_text("[default]\nnetwork_mode = offline\nfile_logging = false\n")
    paths = root / "observer-paths.yaml"
    paths.write_text(f"mono_tune:\n  base_path: {HERE}\n  custom_nodes: nodes\n")  # the observer only
    gens = root / "gens.jsonl"
    gens.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MONO_DECODE_XPU", "MONO_TUNE"))}
    env.update(MONO_TUNE_OUT=str(gens))
    env.update(env_spec)
    port = str(a.port)
    launcher = shlex.split(a.launcher) if a.launcher else [a.python, "main.py"]
    cmd = launcher + ["--listen", "127.0.0.1", "--port", port, "--output-directory", str(root / "out"),
                      "--user-directory", str(root / "user"), "--disable-auto-launch",
                      "--extra-model-paths-config", str(paths)]
    for extra in a.extra_model_paths:
        cmd += ["--extra-model-paths-config", extra]
    cmd += shlex.split(a.server_args)
    base = f"http://127.0.0.1:{port}"
    log = open(root / "server.log", "wb")
    srv = subprocess.Popen(cmd, cwd=a.comfyui, env=env, stdout=log, stderr=subprocess.STDOUT,
                           start_new_session=True)
    runs = []
    try:
        for _ in range(a.start_timeout):
            if srv.poll() is not None:
                sys.exit(f"[{name}] server exited before ready; see {root / 'server.log'}")
            try:
                api(base, "/system_stats")
                break
            except OSError:
                time.sleep(1)
        else:
            sys.exit(f"[{name}] server not ready after {a.start_timeout} s")
        for i in range(a.warmup + a.runs):
            g = copy.deepcopy(prompt)
            seed = a.seed_base + i
            set_seed(g[pe_id], seed)
            before = sum(1 for _ in open(gens)) if gens.exists() else 0
            t0 = time.time()
            pid = api(base, "/prompt", {"prompt": g, "client_id": str(uuid.uuid4())})["prompt_id"]
            hist = None
            while hist is None:
                h = api(base, f"/history/{pid}")
                if pid in h and h[pid].get("status", {}).get("status_str") in ("success", "error"):
                    hist = h[pid]
                elif srv.poll() is not None:
                    sys.exit(f"[{name}] server died; see {root / 'server.log'}")
                else:
                    time.sleep(0.5)
            wall = time.time() - t0
            recs = [json.loads(line) for line in list(open(gens))[before:]] if gens.exists() else []
            text = ""
            for out in (hist.get("outputs") or {}).values():
                if "text" in out:
                    text = "".join(str(x) for x in out["text"])
            (root / f"text-{i}.txt").write_text(text)
            (root / f"run-{i}.json").write_text(json.dumps(hist, indent=1))
            run = dict(config=name, run=i, kind="first" if i == 0 else ("warmup" if i < a.warmup else "measured"),
                       seed=seed, mtp=g[pe_id]["inputs"].get("mtp"), status=hist["status"]["status_str"],
                       wall_s=round(wall, 2),
                       text_chars=len(text), generates=recs)
            runs.append(run)
            r = recs[-1] if recs else {}
            print(json.dumps({k: run[k] for k in ("config", "run", "kind", "status", "wall_s", "text_chars")}
                             | {k: r.get(k) for k in ("tokens", "decode_tps", "tokens_per_step", "drafts_per_step",
                                                      "hit_max_length")}), flush=True)
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(timeout=90)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(srv.pid, signal.SIGKILL)
    return runs


def summary(all_runs):
    rows = ["| config | runs | decode tok/s | tokens | tokens/step | drafts/step | step ms | segments ms "
            "(draft/verify/tail) | weight layout | fallbacks | hit max |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]

    def fmt(vals, nd=1):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "-"
        if len(vals) == 1:
            return f"{vals[0]:.{nd}f}"
        return f"{statistics.mean(vals):.{nd}f} ({min(vals):.{nd}f}-{max(vals):.{nd}f})"

    for config in dict.fromkeys(r["config"] for r in all_runs):
        for kind in ("first", "measured"):
            recs = [g for r in all_runs if r["config"] == config and r["kind"] == kind for g in r["generates"]]
            if not recs:
                continue
            seg = [g["device_segments_ms"] for g in recs if g.get("device_segments_ms")]
            segs = "/".join(fmt([s[k] for s in seg], 2) for k in ("draft", "verify", "tail")) if seg else "-"
            layer = [g["stats"].get("layer", {}) for g in recs]
            layouts = sorted({str((x.get("weight_layout") or {}).get("layout")) for x in layer
                              if isinstance(x, dict) and isinstance(x.get("weight_layout"), dict)})
            fb = {}
            for x in layer:
                if isinstance(x, dict):
                    fb.update({k: v for k, v in (x.get("fallback") or {}).items() if v})
            rows.append(f"| {config} | {kind} x{len(recs)} | {fmt([g['decode_tps'] for g in recs])} | "
                        f"{fmt([g['tokens'] for g in recs], 0)} | {fmt([g['tokens_per_step'] for g in recs], 2)} | "
                        f"{fmt([g['drafts_per_step'] for g in recs], 2)} | {fmt([g['step_ms_mean'] for g in recs], 2)} | "
                        f"{segs} | {', '.join(layouts)} | {fb or '-'} | "
                        f"{sum(bool(g.get('hit_max_length')) for g in recs)} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--comfyui", required=True, help="ComfyUI root (server working directory)")
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", action="append", required=True, metavar="NAME:VAR=V,VAR=V")
    ap.add_argument("--warmup", type=int, default=2, help="runs after the first image that are not measured")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=100)
    ap.add_argument("--pe-only", action="store_true")
    ap.add_argument("--pe-node", help="node id of the prompt-enhance node (default: the only TextGenerate*)")
    ap.add_argument("--mtp", help="override the MTP mode (auto, off, 2-5, tree2s1, tree2s2, tree3s1, tree4s1m8)")
    ap.add_argument("--max-length", type=int)
    ap.add_argument("--launcher", help="server command instead of `python main.py` (ComfyUI arguments are appended)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--server-args", default="", help="extra ComfyUI arguments, e.g. --disable-dynamic-vram")
    ap.add_argument("--extra-model-paths", action="append", default=[])
    ap.add_argument("--port", type=int, default=8288)
    ap.add_argument("--start-timeout", type=int, default=600)
    a = ap.parse_args()
    a.warmup += 1  # run 0, the first image, is never part of the measured mean

    prompt = load_prompt(a.workflow)
    ids = [a.pe_node] if a.pe_node else pe_nodes(prompt)
    if len(ids) != 1 or ids[0] not in prompt:
        sys.exit(f"need exactly one prompt-enhance node, found {ids}; pass --pe-node")
    pe_id = ids[0]
    if set_seed(prompt[pe_id], 0) is None:
        sys.exit(f"node {pe_id} has no sampling seed input (sampling off?): seeds cannot vary the text")
    if a.mtp:
        prompt[pe_id]["inputs"]["mtp"] = a.mtp
    if a.max_length:
        prompt[pe_id]["inputs"]["max_length"] = a.max_length
    if a.pe_only:
        prompt = {k: v for k, v in prompt.items() if k in closure(prompt, pe_id)}
        prompt["mono_tune_preview"] = {"class_type": "PreviewAny", "inputs": {"source": [pe_id, 0]}}

    configs = []
    for spec in a.config:
        name, _, rest = spec.partition(":")
        env = dict(x.split("=", 1) for x in rest.split(",") if x)
        configs.append((name, env))
    if len({n for n, _ in configs}) != len(configs):
        sys.exit("configuration names must be unique (e.g. stock, default, stock2)")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    all_runs = []
    for name, env in configs:
        all_runs += run_config(a, name, env, prompt, pe_id)
        (Path(a.out) / "runs.json").write_text(json.dumps(all_runs, indent=1))
    table = summary(all_runs)
    (Path(a.out) / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
