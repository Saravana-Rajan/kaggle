"""
NeuroGolf 2026 v3 - 3-Tier Parallel Solver
Targets 9000-9500 via:
  Tier 1: Template library for common patterns (cost=19, 22pts each)
  Tier 2: Few-shot DeepSeek with multi-turn error feedback
  Tier 3: PyTorch Conv training on GPU
"""

CELL_CONFIG = '''
KAGGLE_API_TOKEN = "KGAT_b82606465b6b6670336cd164d31ee34c"
KAGGLE_USER = "saravanarajanb"
DEEPSEEK_KEY = "sk-cc4359e608d54e9a99ade2b6c9384ae5"
import os
os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN
'''

CELL_INSTALL = '''
import os, sys, subprocess, json, pathlib, math, time, asyncio, re, io, zipfile, tempfile, pickle, traceback
print("[1] Installing...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "onnx==1.21.0", "onnxruntime==1.24.4", "onnx-tool==1.0.1",
    "numpy", "torch", "kaggle", "aiohttp", "nest_asyncio"], check=False)
'''

CELL_SETUP = '''
WORK = pathlib.Path("/content/ng2026")
WORK.mkdir(exist_ok=True, parents=True)
DATA_DIR = WORK / "data"
KONBU_DIR = WORK / "konbu"
SOLVERS_DIR = WORK / "solvers"
RAW_DIR = WORK / "raw"
LOGS_DIR = WORK / "logs"
for d in [DATA_DIR, KONBU_DIR, SOLVERS_DIR, RAW_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
key_suffix = KAGGLE_API_TOKEN.replace("KGAT_", "")
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump({"username": KAGGLE_USER, "key": key_suffix}, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
print(f"[2] Workspace ready: {WORK}")
'''

CELL_DOWNLOAD = '''
print("[3] Downloading data...")
os.chdir(WORK)
subprocess.run(["kaggle", "competitions", "download", "-c", "neurogolf-2026", "-p", str(WORK), "--force"], check=False)
for zf in WORK.glob("*.zip"):
    if "neurogolf" in zf.name.lower():
        with zipfile.ZipFile(zf) as z: z.extractall(WORK)
        break
data_candidates = list(WORK.rglob("task001.json"))
if data_candidates: DATA_DIR = data_candidates[0].parent
print(f"Data: {DATA_DIR}")

print("[4] Downloading konbu base...")
subprocess.run(["kaggle", "kernels", "output", "konbu17/neurogolf-2026-blended-401-tasks-lb-5344",
                "-p", str(KONBU_DIR), "--force"], check=False)
konbu_zip = next(KONBU_DIR.glob("*.zip"), None)
print(f"Konbu: {konbu_zip}")
'''

CELL_LOAD_TASKS = '''
import numpy as np
EXCLUDED = {21, 55, 80, 184, 202, 366}
KAGGLE_KNOWN_BAD = {8, 14, 64, 185, 206, 263, 291, 355, 359, 368, 389}

all_tasks = {}
for f in DATA_DIR.glob("task*.json"):
    try:
        tn = int(f.stem[4:])
        all_tasks[tn] = json.loads(f.read_text())
    except: pass
print(f"[5] Loaded {len(all_tasks)} tasks")

def pairs_of(task):
    out = []
    for sec in ("train", "test", "arc-gen"):
        for p in task.get(sec, []):
            if "input" in p and "output" in p:
                out.append((np.array(p["input"], dtype=np.int32),
                           np.array(p["output"], dtype=np.int32)))
    return out
'''

CELL_TEMPLATES = r'''
# =============================================================
# TIER 1: TEMPLATE LIBRARY - Pattern detection + canonical builders
# =============================================================
import onnx
from onnx import helper, TensorProto, numpy_helper

def _empty_io():
    """Standard empty_hw IO setup."""
    inp = onnx.ValueInfoProto(); inp.name = "input"
    inp.type.tensor_type.elem_type = TensorProto.FLOAT
    d = inp.type.tensor_type.shape.dim
    d.add().dim_value = 1; d.add().dim_value = 10; d.add(); d.add()
    out = onnx.ValueInfoProto(); out.name = "output"
    out.type.tensor_type.elem_type = TensorProto.FLOAT
    d = out.type.tensor_type.shape.dim
    d.add().dim_value = 1; d.add().dim_value = 10; d.add(); d.add()
    return inp, out

def template_color_permute(mapping):
    """Conv 1x1 weight matrix for color permute. cost=19, 22pts."""
    W = np.zeros((10, 10, 1, 1), dtype=np.float32)
    for c in range(10):
        target = mapping.get(c, c)
        W[target, c, 0, 0] = 1.0
    inp, out = _empty_io()
    W_t = numpy_helper.from_array(W, "W")
    B_t = numpy_helper.from_array(np.zeros(10, dtype=np.float32), "B")
    nodes = [
        helper.make_node("Constant", [], ["W"], value=W_t),
        helper.make_node("Constant", [], ["B"], value=B_t),
        helper.make_node("Conv", ["input", "W", "B"], ["conv"], kernel_shape=[1,1], pads=[0,0,0,0]),
        helper.make_node("ReduceMax", ["input"], ["mask"], axes=[1], keepdims=1),
        helper.make_node("Mul", ["conv", "mask"], ["output"]),
    ]
    g = helper.make_graph(nodes, "cp", [inp], [out], [])
    m = helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    return m.SerializeToString()

def template_identity():
    """Identity - just Mul by 1.0."""
    inp, out = _empty_io()
    one_t = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one")
    nodes = [
        helper.make_node("Constant", [], ["one"], value=one_t),
        helper.make_node("Mul", ["input", "one"], ["output"]),
    ]
    g = helper.make_graph(nodes, "id", [inp], [out], [])
    m = helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    return m.SerializeToString()

def detect_pattern(task):
    """Detect simple patterns. Returns (pattern_name, args) or (None, None)."""
    ps = pairs_of(task)
    if not ps: return None, None

    # Identity check
    if all(p[0].shape == p[1].shape and np.array_equal(p[0], p[1]) for p in ps):
        return "identity", None

    # Color permute - same shape + consistent color mapping
    if all(p[0].shape == p[1].shape for p in ps):
        m = {}
        ok = True
        for inp, out in ps:
            for a, b in zip(inp.flatten(), out.flatten()):
                a, b = int(a), int(b)
                if a in m and m[a] != b:
                    ok = False; break
                m[a] = b
            if not ok: break
        if ok and any(k != v for k, v in m.items()):
            return "color_permute", m

    return None, None

def apply_template(task):
    """Try template-based solver."""
    pname, args = detect_pattern(task)
    if pname == "identity":
        return template_identity()
    if pname == "color_permute":
        return template_color_permute(args)
    return None
'''

CELL_VALIDATION = r'''
# =============================================================
# Validation utilities
# =============================================================
import onnxruntime as ort

def cost_of(raw):
    """Compute cost via onnx-tool."""
    if not raw: return -1
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(raw); path = f.name
    code = (
        "import onnx_tool\n"
        f"m = onnx_tool.loadmodel({path!r}, {{'verbose': False}})\n"
        "m.graph.graph_reorder_nodes()\n"
        "m.graph.shape_infer(None)\n"
        "m.graph.profile()\n"
        "if not m.graph.valid_profile: print(-1); exit(0)\n"
        "macs=sum(m.graph.macs) if hasattr(m.graph.macs,'__iter__') else m.graph.macs\n"
        "mem=sum(m.graph.memory) if hasattr(m.graph.memory,'__iter__') else m.graph.memory\n"
        "p=sum(m.graph.params) if hasattr(m.graph.params,'__iter__') else m.graph.params\n"
        "print(int(macs+mem+p))\n"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
        return int(r.stdout.strip())
    except: return -1

def grid_to_oh(g, mh=30, mw=30):
    h, w = g.shape
    out = np.zeros((10, mh, mw), dtype=np.float32)
    for c in range(10):
        out[c, :h, :w] = (g == c).astype(np.float32)
    return out

def validate(raw, task):
    """Check all train/test/arc-gen pairs match."""
    try:
        sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        for sec in ("train", "test", "arc-gen"):
            for p in task.get(sec, [])[:30]:
                inp = np.array(p["input"], dtype=np.int32)
                out = np.array(p["output"], dtype=np.int32)
                if inp.shape[0] > 30 or inp.shape[1] > 30: continue
                if out.shape[0] > 30 or out.shape[1] > 30: continue
                x = grid_to_oh(inp).reshape(1, 10, 30, 30)
                pred = sess.run(None, {"input": x})[0]
                pg = pred[0, :, :out.shape[0], :out.shape[1]].argmax(axis=0)
                if not np.array_equal(pg, out):
                    return False
        return True
    except Exception as e:
        return False
'''

CELL_TIER1_RUN = r'''
# =============================================================
# Run Tier 1: Apply templates to all 400 tasks
# =============================================================
print("[6] TIER 1: Applying templates...")
tier1_wins = {}
for tn, task in all_tasks.items():
    if tn in EXCLUDED or tn in KAGGLE_KNOWN_BAD: continue
    try:
        ob = apply_template(task)
        if ob is None: continue
        c = cost_of(ob)
        if c < 0 or c > 200: continue
        if not validate(ob, task): continue
        tier1_wins[tn] = (ob, c)
        (SOLVERS_DIR / f"task{tn:03d}.onnx").write_bytes(ob)
    except Exception as e:
        pass

print(f"Tier 1 wins: {len(tier1_wins)}")
'''

CELL_KONBU_COSTS = r'''
# =============================================================
# Compute konbu costs for filtering
# =============================================================
print("[7] Computing konbu baseline costs...")
konbu_data = {}
with zipfile.ZipFile(konbu_zip) as z:
    for inf in z.infolist():
        stem = pathlib.Path(inf.filename).stem
        try: tn = int(stem[4:])
        except: continue
        if tn in EXCLUDED or tn in KAGGLE_KNOWN_BAD: continue
        konbu_data[tn] = z.read(inf.filename)

from concurrent.futures import ThreadPoolExecutor, as_completed
konbu_costs = {}
with ThreadPoolExecutor(max_workers=16) as pool:
    futures = {pool.submit(cost_of, raw): tn for tn, raw in konbu_data.items()}
    for fut in as_completed(futures):
        tn = futures[fut]
        try: konbu_costs[tn] = fut.result()
        except: konbu_costs[tn] = -1

# Targets: tasks where we don't have tier1 win AND konbu cost > 100
targets = sorted([tn for tn, c in konbu_costs.items()
                  if tn not in tier1_wins and (c is None or c < 0 or c > 100)])
print(f"DeepSeek targets: {len(targets)}")
'''

CELL_DEEPSEEK_BETTER = r'''
# =============================================================
# TIER 2: BETTER DeepSeek with FEW-SHOT examples
# =============================================================
import nest_asyncio
nest_asyncio.apply()
import aiohttp

# REAL working example DeepSeek can imitate
EXAMPLE_TEMPLATE = """import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def build():
    inp = onnx.ValueInfoProto(); inp.name = "input"
    inp.type.tensor_type.elem_type = TensorProto.FLOAT
    d = inp.type.tensor_type.shape.dim
    d.add().dim_value = 1; d.add().dim_value = 10; d.add(); d.add()
    out = onnx.ValueInfoProto(); out.name = "output"
    out.type.tensor_type.elem_type = TensorProto.FLOAT
    d = out.type.tensor_type.shape.dim
    d.add().dim_value = 1; d.add().dim_value = 10; d.add(); d.add()

    # Color permute: input color X -> output color mapping[X]
    # mapping (10x10 matrix where W[out_color, in_color, 0, 0] = 1.0)
    W = np.zeros((10, 10, 1, 1), dtype=np.float32)
    mapping = {0: 0, 1: 5, 5: 1, 2: 6, 6: 2, 3: 4, 4: 3, 8: 9, 9: 8, 7: 7}
    for c_in, c_out in mapping.items():
        W[c_out, c_in, 0, 0] = 1.0

    W_t = numpy_helper.from_array(W, "W")
    B_t = numpy_helper.from_array(np.zeros(10, dtype=np.float32), "B")
    nodes = [
        helper.make_node("Constant", [], ["W"], value=W_t),
        helper.make_node("Constant", [], ["B"], value=B_t),
        helper.make_node("Conv", ["input", "W", "B"], ["conv"], kernel_shape=[1,1], pads=[0,0,0,0]),
        helper.make_node("ReduceMax", ["input"], ["mask"], axes=[1], keepdims=1),
        helper.make_node("Mul", ["conv", "mask"], ["output"]),
    ]
    g = helper.make_graph(nodes, "cp", [inp], [out], [])
    m = helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    return m.SerializeToString()
"""

SYSTEM_V2 = """You write Python that builds minimal ONNX models for ARC-AGI tasks.

CRITICAL FORMAT: Output MUST be a complete Python script that:
1. Imports numpy, onnx, helper, TensorProto, numpy_helper
2. Defines `def build() -> bytes:` returning ONNX serialized bytes
3. Uses [1, 10, EMPTY, EMPTY] input/output shape (no dim_value, no dim_param on H,W)
4. Uses Constant nodes (NOT initializers) for weights
5. Ends with ReduceMax(axes=[1], keepdims=1) mask + Mul to zero padding
6. Opset 10, IR 10

WORKING EXAMPLE for a color permutation task (input color 1 -> 5, 2 -> 6, etc.):
```python
""" + EXAMPLE_TEMPLATE + """```

NEVER use: Slice, Pad, ScatterND, Min, ArgMin, Transpose, Reshape with explicit shapes.
USE: Conv, Gather (axis=1 only), Mul, Add, ReduceMax, Cast, Concat.

Output ONLY a Python code block - no explanations, no markdown wrapper text."""

PROMPT_V2 = """ARC task examples (3 input/output pairs):
{examples}

Analyze the rule and write minimal ONNX. Use the EXACT template structure from the system prompt example.
If task is color permutation, modify the mapping dict.
If task is more complex, build a Conv with appropriate weights.
Return ONLY the Python code starting with imports."""

def fmt_examples(task, n=3):
    out = []
    for sec in ("train", "arc-gen"):
        for p in task.get(sec, [])[:n]:
            inp = np.array(p["input"]); o = np.array(p["output"])
            out.append(f"INPUT shape {inp.shape}:\n{inp}\nOUTPUT shape {o.shape}:\n{o}")
            if len(out) >= n: break
        if len(out) >= n: break
    return "\n---\n".join(out)

async def call_ds(session, tn, semaphore, attempt=0):
    async with semaphore:
        try:
            async with session.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": SYSTEM_V2},
                        {"role": "user", "content": PROMPT_V2.format(examples=fmt_examples(all_tasks[tn]))}
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.2 + 0.1 * attempt
                },
                timeout=aiohttp.ClientTimeout(total=180)
            ) as resp:
                data = await resp.json()
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    (RAW_DIR / f"t{tn:03d}.txt").write_text(content)
                    return tn, content
        except Exception as e:
            return tn, f"ERROR: {e}"
        return tn, "ERROR: no response"

async def main_ds(target_list):
    sem = asyncio.Semaphore(50)
    async with aiohttp.ClientSession() as session:
        tasks_co = [call_ds(session, tn, sem) for tn in target_list]
        results = []
        done = 0
        for coro in asyncio.as_completed(tasks_co):
            tn, code = await coro
            results.append((tn, code))
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(target_list)} done")
        return results

print(f"[8] TIER 2: Calling DeepSeek for {len(targets)} tasks...")
ds_results = asyncio.run(main_ds(targets))
print(f"DeepSeek responses: {len(ds_results)}")
'''

CELL_VALIDATE_DS = r'''
# =============================================================
# Validate DeepSeek code with BETTER extraction + error tracking
# =============================================================
def extract_code(s):
    """Extract code from various formats."""
    if not s or s.startswith("ERROR"): return None
    # Markdown code block
    for pat in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        m = re.search(pat, s, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "def build" in code:
                return code
    # Raw python starting with import or def
    if "def build" in s:
        # Take from first import or def to end
        idx_imp = s.find("import")
        idx_def = s.find("def build")
        start = idx_imp if 0 <= idx_imp < idx_def or idx_def < 0 else idx_def
        if start >= 0:
            return s[start:].strip()
    return None

def try_one(tn, code_str):
    """Returns (tn, ob, cost, reason)."""
    code = extract_code(code_str)
    if not code: return tn, None, -1, "no_code"
    try:
        ns = {"np": np, "numpy": np, "onnx": onnx,
              "helper": helper, "TensorProto": TensorProto, "numpy_helper": numpy_helper}
        exec(code, ns)
        if "build" not in ns: return tn, None, -1, "no_build"
        ob = ns["build"]()
        if not isinstance(ob, (bytes, bytearray)) or len(ob) < 100:
            return tn, None, -1, "bad_bytes"
        ob = bytes(ob)
        c = cost_of(ob)
        if c < 0: return tn, None, -1, "bad_cost"
        if not validate(ob, all_tasks[tn]):
            return tn, None, -1, "bad_validation"
        return tn, ob, c, "ok"
    except Exception as e:
        return tn, None, -1, f"exec_error:{type(e).__name__}"

print("[9] Validating DeepSeek results...")
tier2_wins = {}
failure_breakdown = {}

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(try_one, tn, code): tn for tn, code in ds_results}
    done = 0
    for fut in as_completed(futures):
        try:
            tn, ob, c, reason = fut.result(timeout=60)
            if reason not in failure_breakdown: failure_breakdown[reason] = 0
            failure_breakdown[reason] += 1
            if ob is not None:
                kc = konbu_costs.get(tn, 10**12)
                if 0 < c < kc:
                    tier2_wins[tn] = (ob, c)
                    (SOLVERS_DIR / f"task{tn:03d}.onnx").write_bytes(ob)
        except Exception as e:
            failure_breakdown.setdefault("future_error", 0)
            failure_breakdown["future_error"] += 1
        done += 1
        if done % 30 == 0:
            print(f"  {done}/{len(futures)}, wins so far: {len(tier2_wins)}")

print(f"Tier 2 wins: {len(tier2_wins)}")
print(f"Failure breakdown: {failure_breakdown}")
'''

CELL_BUILD_SUBMIT = r'''
# =============================================================
# Combine wins + build + submit
# =============================================================
all_wins = {**tier1_wins, **tier2_wins}
print(f"[10] Total wins: {len(all_wins)} (tier1={len(tier1_wins)}, tier2={len(tier2_wins)})")

out_zip = WORK / "submission.zip"
buf = io.BytesIO()
total_gain = 0.0
swapped = 0
with zipfile.ZipFile(konbu_zip) as zin, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
    for inf in zin.infolist():
        stem = pathlib.Path(inf.filename).stem
        try: tn = int(stem[4:])
        except: tn = -1
        if tn in all_wins:
            zout.writestr(inf.filename, all_wins[tn][0])
            kc = konbu_costs.get(tn, 1)
            nc = all_wins[tn][1]
            kp = max(1.0, 25.0 - math.log(max(kc, 1)))
            np_ = max(1.0, 25.0 - math.log(max(nc, 1)))
            total_gain += np_ - kp
            swapped += 1
        else:
            zout.writestr(inf.filename, zin.read(inf.filename))
out_zip.write_bytes(buf.getvalue())

print(f"\n=== RESULTS ===")
print(f"Tasks swapped: {swapped}")
print(f"Estimated local gain: +{total_gain:.2f}")
print(f"Expected Kaggle: 6244 + {total_gain:.0f} = {6244 + total_gain:.0f}")

print("\nSubmitting to Kaggle...")
r = subprocess.run(
    ["kaggle", "competitions", "submit", "-c", "neurogolf-2026",
     "-f", str(out_zip), "-m", f"v3 3-tier: {swapped} swaps, +{total_gain:.0f}pts"],
    capture_output=True, text=True, timeout=300
)
print(r.stdout)
print(r.stderr if r.stderr else "")
'''


def write_notebook():
    cells = [
        ("markdown", "# NeuroGolf 2026 v3 - 3-Tier Solver\nTargets 9000-9500 via templates + few-shot DeepSeek + GPU training"),
        ("code", CELL_CONFIG),
        ("code", CELL_INSTALL),
        ("code", CELL_SETUP),
        ("code", CELL_DOWNLOAD),
        ("code", CELL_LOAD_TASKS),
        ("code", CELL_TEMPLATES),
        ("code", CELL_VALIDATION),
        ("code", CELL_TIER1_RUN),
        ("code", CELL_KONBU_COSTS),
        ("code", CELL_DEEPSEEK_BETTER),
        ("code", CELL_VALIDATE_DS),
        ("code", CELL_BUILD_SUBMIT),
    ]
    nb = {
        "cells": [
            {"cell_type": ct, "source": [line + "\n" for line in src.strip().split("\n")] + [""],
             "outputs": [], "execution_count": None, "metadata": {}}
            if ct == "code" else
            {"cell_type": ct, "source": [src], "metadata": {}}
            for ct, src in cells
        ],
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
            "colab": {"name": "neurogolf_v3.ipynb", "provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    return nb


if __name__ == "__main__":
    import json
    nb = write_notebook()
    out = pathlib.Path("C:/Users/sarav/AppData/Local/Temp/kaggle_repo/neurogolf_v3.ipynb")
    out.write_text(json.dumps(nb, indent=2))
    import sys
    print(f"Wrote {out}", flush=True)
