"""Build neurogolf_v6.ipynb - INFINITE loop until 9K total.

Per-task agent:
- Unlimited attempts until 24 pts (or ceiling reached)
- Multiple strategies cycle: templates -> DeepSeek -> train
- Persistent JSON state (resume on disconnect)
- Multi-round iteration

Submit gate:
- Track total estimated score live
- ONLY submit to Kaggle when total >= 9000
- Otherwise keep iterating forever
"""
import json, pathlib

CELLS = []

CELLS.append(("markdown", """# NeuroGolf 2026 v6 - INFINITE Per-Task Agents
**Strategy:** Loop forever per task until 24 pts. Submit ONLY when total >= 9000.
- Unlimited attempts per task (DeepSeek + GPU training)
- Persistent state (resume on disconnect)
- Multi-round iteration through 400 tasks
- Auto-submit to Kaggle when 9K total reached"""))

CELLS.append(("code", """KAGGLE_API_TOKEN = "KGAT_b82606465b6b6670336cd164d31ee34c"
KAGGLE_USER = "saravanarajanb"
DEEPSEEK_KEY = "sk-cc4359e608d54e9a99ade2b6c9384ae5"
TARGET_TOTAL = 9000  # only submit when total >= this
PER_TASK_TARGET = 24  # stop iterating task when reaches this
import os
os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN"""))

CELLS.append(("code", """import os, sys, subprocess, json, pathlib, math, time, asyncio, re, io, zipfile, tempfile, pickle, traceback
print("[1] Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "onnx==1.21.0", "onnxruntime==1.24.4", "onnx-tool==1.0.1",
    "numpy", "torch", "kaggle", "aiohttp", "nest_asyncio"], check=False)
print("Done")"""))

CELLS.append(("code", """WORK = pathlib.Path("/content/ng2026")
WORK.mkdir(exist_ok=True, parents=True)
DATA_DIR = WORK / "data"
KONBU_DIR = WORK / "konbu"
SOLVERS_DIR = WORK / "solvers"
RAW_DIR = WORK / "raw"
LOGS_DIR = WORK / "logs"
STATE_FILE = LOGS_DIR / "state.json"
for d in [DATA_DIR, KONBU_DIR, SOLVERS_DIR, RAW_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
key_suffix = KAGGLE_API_TOKEN.replace("KGAT_", "")
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump({"username": KAGGLE_USER, "key": key_suffix}, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
print(f"[2] Workspace: {WORK}")

import torch
print(f"GPU: {torch.cuda.is_available()} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")"""))

CELLS.append(("code", """print("[3] Downloading competition data...")
os.chdir(WORK)
subprocess.run(["kaggle", "competitions", "download", "-c", "neurogolf-2026", "-p", str(WORK), "--force"], check=False)
for zf in WORK.glob("*.zip"):
    if "neurogolf" in zf.name.lower():
        with zipfile.ZipFile(zf) as z: z.extractall(WORK)
        break
data_candidates = list(WORK.rglob("task001.json"))
if data_candidates: DATA_DIR = data_candidates[0].parent
print(f"Data dir: {DATA_DIR}")

print("[4] Downloading konbu base (6244 LB)...")
subprocess.run(["kaggle", "kernels", "output", "konbu17/neurogolf-2026-blended-401-tasks-lb-5344",
                "-p", str(KONBU_DIR), "--force"], check=False)
konbu_zip = next(KONBU_DIR.glob("*.zip"), None)
print(f"Konbu zip: {konbu_zip}")
assert konbu_zip and konbu_zip.exists(), "Konbu download failed!"
"""))

CELLS.append(("code", """import numpy as np
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

scoreable_tasks = sorted([tn for tn in all_tasks if tn not in EXCLUDED and tn not in KAGGLE_KNOWN_BAD])
print(f"Scoreable: {len(scoreable_tasks)}")"""))

CELLS.append(("code", r"""# === TEMPLATE LIBRARY ===
import onnx
from onnx import helper, TensorProto, numpy_helper

def _empty_io():
    inp = onnx.ValueInfoProto(); inp.name = "input"
    inp.type.tensor_type.elem_type = TensorProto.FLOAT
    d = inp.type.tensor_type.shape.dim
    d.add().dim_value = 1; d.add().dim_value = 10; d.add(); d.add()
    out = onnx.ValueInfoProto(); out.name = "output"
    out.type.tensor_type.elem_type = TensorProto.FLOAT
    d = out.type.tensor_type.shape.dim
    d.add().dim_value = 1; d.add().dim_value = 10; d.add(); d.add()
    return inp, out

ONE_F = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "ONE_F")

def _flip_idx(N, max_dim=30):
    safe = N if N < max_dim else max_dim - 1
    return np.array([N - 1 - r if r < N else safe for r in range(max_dim)], dtype=np.int32)

def _mirror_idx(N_in, N_out, max_dim=30):
    safe = N_in if N_in < max_dim else max_dim - 1
    out = []
    for r in range(max_dim):
        if r < N_out:
            out.append(r if r < N_in else (2*N_in - 1 - r))
        else:
            out.append(safe)
    return np.array(out, dtype=np.int32)

def _build(nodes, inits=None):
    if inits is None: inits = []
    inp, out = _empty_io()
    g = helper.make_graph(nodes, "g", [inp], [out], inits)
    return helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid("", 10)]).SerializeToString()

def t_identity():
    one_t = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one")
    return _build([
        helper.make_node("Constant", [], ["one"], value=one_t),
        helper.make_node("Mul", ["input", "one"], ["output"]),
    ])

def t_color_permute(mapping):
    W = np.zeros((10, 10, 1, 1), dtype=np.float32)
    for c in range(10):
        target = mapping.get(c, c)
        W[target, c, 0, 0] = 1.0
    return _build([
        helper.make_node("Constant", [], ["W"], value=numpy_helper.from_array(W, "W")),
        helper.make_node("Constant", [], ["B"], value=numpy_helper.from_array(np.zeros(10, dtype=np.float32), "B")),
        helper.make_node("Conv", ["input", "W", "B"], ["conv"], kernel_shape=[1,1], pads=[0,0,0,0]),
        helper.make_node("ReduceMax", ["input"], ["mask"], axes=[1], keepdims=1),
        helper.make_node("Mul", ["conv", "mask"], ["output"]),
    ])

def t_fliph(W_in):
    idx = _flip_idx(W_in)
    return _build([
        helper.make_node("Constant", [], ["idx"], value=numpy_helper.from_array(idx, "idx")),
        helper.make_node("Gather", ["input", "idx"], ["g"], axis=3),
        helper.make_node("Mul", ["g", "ONE_F"], ["output"]),
    ], [ONE_F])

def t_flipv(H_in):
    idx = _flip_idx(H_in)
    return _build([
        helper.make_node("Constant", [], ["idx"], value=numpy_helper.from_array(idx, "idx")),
        helper.make_node("Gather", ["input", "idx"], ["g"], axis=2),
        helper.make_node("Mul", ["g", "ONE_F"], ["output"]),
    ], [ONE_F])

def t_rot180(H_in, W_in):
    idh = _flip_idx(H_in); idw = _flip_idx(W_in)
    return _build([
        helper.make_node("Constant", [], ["idh"], value=numpy_helper.from_array(idh, "idh")),
        helper.make_node("Constant", [], ["idw"], value=numpy_helper.from_array(idw, "idw")),
        helper.make_node("Gather", ["input", "idh"], ["gh"], axis=2),
        helper.make_node("Gather", ["gh", "idw"], ["gw"], axis=3),
        helper.make_node("Mul", ["gw", "ONE_F"], ["output"]),
    ], [ONE_F])

def t_mirror_quad(H_in, W_in, H_out, W_out):
    idh = _mirror_idx(H_in, H_out); idw = _mirror_idx(W_in, W_out)
    return _build([
        helper.make_node("Constant", [], ["idh"], value=numpy_helper.from_array(idh, "idh")),
        helper.make_node("Constant", [], ["idw"], value=numpy_helper.from_array(idw, "idw")),
        helper.make_node("Gather", ["input", "idh"], ["gh"], axis=2),
        helper.make_node("Gather", ["gh", "idw"], ["gw"], axis=3),
        helper.make_node("Mul", ["gw", "ONE_F"], ["output"]),
    ], [ONE_F])

def t_vmirror_below(H_in, H_out):
    idh = _mirror_idx(H_in, H_out)
    return _build([
        helper.make_node("Constant", [], ["idh"], value=numpy_helper.from_array(idh, "idh")),
        helper.make_node("Gather", ["input", "idh"], ["g"], axis=2),
        helper.make_node("Mul", ["g", "ONE_F"], ["output"]),
    ], [ONE_F])

def t_vmirror_above(H_in, H_out):
    idh_top = list(range(H_in - 1, -1, -1))
    idh_bot = list(range(H_in))
    safe = H_in if H_in < 30 else 29
    idh = np.array(idh_top + idh_bot + [safe] * (30 - H_out), dtype=np.int32)
    return _build([
        helper.make_node("Constant", [], ["idh"], value=numpy_helper.from_array(idh, "idh")),
        helper.make_node("Gather", ["input", "idh"], ["g"], axis=2),
        helper.make_node("Mul", ["g", "ONE_F"], ["output"]),
    ], [ONE_F])

def t_hmirror_right(W_in, W_out):
    idw_l = list(range(W_in))
    idw_r = list(range(W_in - 1, -1, -1))
    safe = W_in if W_in < 30 else 29
    idw = np.array(idw_l + idw_r + [safe] * (30 - W_out), dtype=np.int32)
    return _build([
        helper.make_node("Constant", [], ["idw"], value=numpy_helper.from_array(idw, "idw")),
        helper.make_node("Gather", ["input", "idw"], ["g"], axis=3),
        helper.make_node("Mul", ["g", "ONE_F"], ["output"]),
    ], [ONE_F])

def t_crop(top, left, out_H, out_W, H_in, W_in):
    safe_r = H_in if H_in < 30 else 29
    safe_c = W_in if W_in < 30 else 29
    row_idx = list(range(top, top + out_H)) + [safe_r] * (30 - out_H)
    col_idx = list(range(left, left + out_W)) + [safe_c] * (30 - out_W)
    return _build([
        helper.make_node("Constant", [], ["rh"], value=numpy_helper.from_array(np.array(row_idx[:30], dtype=np.int32), "rh")),
        helper.make_node("Constant", [], ["cw"], value=numpy_helper.from_array(np.array(col_idx[:30], dtype=np.int32), "cw")),
        helper.make_node("Gather", ["input", "rh"], ["gh"], axis=2),
        helper.make_node("Gather", ["gh", "cw"], ["gw"], axis=3),
        helper.make_node("Mul", ["gw", "ONE_F"], ["output"]),
    ], [ONE_F])

def detect_pattern(task):
    ps = pairs_of(task)
    if not ps: return None, None
    if all(p[0].shape == p[1].shape and np.array_equal(p[0], p[1]) for p in ps):
        return "identity", None
    si = set(p[0].shape for p in ps); so = set(p[1].shape for p in ps)
    same_shape = all(p[0].shape == p[1].shape for p in ps)
    if same_shape:
        m = {}; ok = True
        for inp, out in ps:
            for a, b in zip(inp.flatten(), out.flatten()):
                a, b = int(a), int(b)
                if a in m and m[a] != b: ok = False; break
                m[a] = b
            if not ok: break
        if ok and any(k != v for k, v in m.items()):
            return "color_permute", m
    if len(si) == 1 and len(so) == 1:
        ih, iw = list(si)[0]; oh, ow = list(so)[0]
        if (ih, iw) == (oh, ow):
            if all(np.array_equal(np.fliplr(i), o) for i, o in ps): return "fliph", iw
            if all(np.array_equal(np.flipud(i), o) for i, o in ps): return "flipv", ih
            if all(np.array_equal(np.rot90(i, 2), o) for i, o in ps): return "rot180", (ih, iw)
        if oh == 2*ih and ow == 2*iw:
            ok = True
            for i, o in ps:
                if (not np.array_equal(o[:ih, :iw], i) or
                    not np.array_equal(o[:ih, iw:], np.fliplr(i)) or
                    not np.array_equal(o[ih:, :iw], np.flipud(i)) or
                    not np.array_equal(o[ih:, iw:], np.rot90(i, 2))):
                    ok = False; break
            if ok: return "mirror_quad", (ih, iw, oh, ow)
        if oh == 2*ih and ow == iw:
            if all(np.array_equal(o[:ih], i) and np.array_equal(o[ih:], i[::-1]) for i, o in ps):
                return "vmirror_below", (ih, oh)
            if all(np.array_equal(o[:ih], i[::-1]) and np.array_equal(o[ih:], i) for i, o in ps):
                return "vmirror_above", (ih, oh)
        if oh == ih and ow == 2*iw:
            if all(np.array_equal(o[:, :iw], i) and np.array_equal(o[:, iw:], i[:, ::-1]) for i, o in ps):
                return "hmirror_right", (iw, ow)
        if oh < ih or ow < iw:
            if oh <= ih and ow <= iw:
                for top in range(ih - oh + 1):
                    for left in range(iw - ow + 1):
                        if all(np.array_equal(p[0][top:top+oh, left:left+ow], p[1]) for p in ps):
                            return "crop", (top, left, oh, ow, ih, iw)
    return None, None

def apply_template(task):
    pname, args = detect_pattern(task)
    if pname is None: return None, None
    try:
        if pname == "identity": return t_identity(), pname
        if pname == "color_permute": return t_color_permute(args), pname
        if pname == "fliph": return t_fliph(args), pname
        if pname == "flipv": return t_flipv(args), pname
        if pname == "rot180": return t_rot180(*args), pname
        if pname == "mirror_quad": return t_mirror_quad(*args), pname
        if pname == "vmirror_below": return t_vmirror_below(*args), pname
        if pname == "vmirror_above": return t_vmirror_above(*args), pname
        if pname == "hmirror_right": return t_hmirror_right(*args), pname
        if pname == "crop": return t_crop(*args), pname
    except Exception as e:
        return None, f"err:{type(e).__name__}"
    return None, None"""))

CELLS.append(("code", r"""# === VALIDATION ===
import onnxruntime as ort

def cost_of(raw):
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
                if not np.array_equal(pg, out): return False
        return True
    except: return False

def pts_of(c):
    if c is None or c < 0: return 0.0
    return max(1.0, 25.0 - math.log(max(c, 1)))"""))

CELLS.append(("code", r"""# === KONBU BASELINE ===
print("\n[6] Computing konbu baseline costs...")
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

# Compute current best per task (start from konbu)
best_per_task = {}  # tn -> (ob_bytes, cost, pts)
for tn, raw in konbu_data.items():
    c = konbu_costs.get(tn, -1)
    if c > 0:
        best_per_task[tn] = (raw, c, pts_of(c))

konbu_total = sum(p for _, _, p in best_per_task.values())
print(f"Starting from konbu: {konbu_total:.2f} pts on {len(best_per_task)} tasks")"""))

CELLS.append(("code", r"""# === TIER 1: TEMPLATES (first pass) ===
print("\n[7] === TIER 1: Templates ===")
template_wins = 0
for tn in scoreable_tasks:
    try:
        ob, pname = apply_template(all_tasks[tn])
        if ob is None: continue
        c = cost_of(ob)
        if c < 0 or c > 500: continue
        if not validate(ob, all_tasks[tn]): continue
        new_pts = pts_of(c)
        cur_pts = best_per_task.get(tn, (None, -1, 0))[2]
        if new_pts > cur_pts:
            best_per_task[tn] = (ob, c, new_pts)
            template_wins += 1
            (SOLVERS_DIR / f"task{tn:03d}.onnx").write_bytes(ob)
            print(f"  task{tn:03d}: TEMPLATE [{pname}] {cur_pts:.2f} -> {new_pts:.2f} (cost={c})")
    except: pass

total = sum(p for _, _, p in best_per_task.values())
print(f"\nTemplates: {template_wins} wins, total now: {total:.2f}/9000")"""))

CELLS.append(("code", r'''# === STATE PERSISTENCE ===
def save_state():
    state = {"best_costs": {str(tn): c for tn, (_, c, _) in best_per_task.items()},
             "konbu_costs": {str(tn): c for tn, c in konbu_costs.items()}}
    STATE_FILE.write_text(json.dumps(state))

def load_state():
    if not STATE_FILE.exists(): return False
    try:
        state = json.loads(STATE_FILE.read_text())
        for tn_str, c in state.get("best_costs", {}).items():
            tn = int(tn_str)
            f = SOLVERS_DIR / f"task{tn:03d}.onnx"
            if f.exists() and tn in scoreable_tasks:
                best_per_task[tn] = (f.read_bytes(), c, pts_of(c))
        return True
    except:
        return False

save_state()
print("State saved.")'''))

CELLS.append(("code", r'''# === DEEPSEEK PER-TASK AGENT ===
import nest_asyncio
nest_asyncio.apply()
import aiohttp

EXAMPLE = (
    "import numpy as np, onnx\n"
    "from onnx import helper, TensorProto, numpy_helper\n\n"
    "def build():\n"
    "    inp = onnx.ValueInfoProto(); inp.name='input'\n"
    "    inp.type.tensor_type.elem_type = TensorProto.FLOAT\n"
    "    d = inp.type.tensor_type.shape.dim\n"
    "    d.add().dim_value=1; d.add().dim_value=10; d.add(); d.add()\n"
    "    out = onnx.ValueInfoProto(); out.name='output'\n"
    "    out.type.tensor_type.elem_type = TensorProto.FLOAT\n"
    "    d = out.type.tensor_type.shape.dim\n"
    "    d.add().dim_value=1; d.add().dim_value=10; d.add(); d.add()\n\n"
    "    W = np.zeros((10,10,1,1), dtype=np.float32)\n"
    "    for i in range(10): W[i, i, 0, 0] = 1.0  # identity color map\n\n"
    "    nodes = [\n"
    "        helper.make_node('Constant', [], ['W'], value=numpy_helper.from_array(W, 'W')),\n"
    "        helper.make_node('Constant', [], ['B'], value=numpy_helper.from_array(np.zeros(10, dtype=np.float32), 'B')),\n"
    "        helper.make_node('Conv', ['input', 'W', 'B'], ['conv'], kernel_shape=[1,1], pads=[0,0,0,0]),\n"
    "        helper.make_node('ReduceMax', ['input'], ['mask'], axes=[1], keepdims=1),\n"
    "        helper.make_node('Mul', ['conv', 'mask'], ['output']),\n"
    "    ]\n"
    "    g = helper.make_graph(nodes, 'm', [inp], [out], [])\n"
    "    return helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid('', 10)]).SerializeToString()\n"
)

SYSTEM = (
    "You write minimal ONNX models for ARC-AGI tasks. Cost = MACs+memory+params (onnx-tool 1.0.1).\n\n"
    "RULES (CRITICAL):\n"
    "1. Input/output shape: [1, 10, EMPTY, EMPTY] (no dim_value, no dim_param on H,W)\n"
    "2. Use Constant nodes (NOT initializers) - they are free\n"
    "3. End with: ReduceMax(input, axes=[1], keepdims=1) mask, Mul to apply\n"
    "4. Opset 10, IR 10\n"
    "5. AVOID: Slice, Pad, ScatterND, Min, ArgMin, Transpose - they crash\n"
    "6. USE: Conv (1x1 / 3x3), Gather (axis 1/2/3), Mul, Add, ReduceMax, Cast\n\n"
    "WORKING TEMPLATE:\n```python\n" + EXAMPLE + "```\n\n"
    "Output ONLY a Python code block. Must define def build() -> bytes."
)

def fmt_examples(task, n=3):
    out = []
    for sec in ("train", "arc-gen"):
        for p in task.get(sec, [])[:n]:
            inp = np.array(p["input"]); o = np.array(p["output"])
            out.append(f"INPUT shape {inp.shape}:\n{inp}\nOUTPUT shape {o.shape}:\n{o}")
            if len(out) >= n: break
        if len(out) >= n: break
    return "\n---\n".join(out)

def extract_code(s):
    if not s or s.startswith("ERROR"): return None
    for pat in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        m = re.search(pat, s, re.DOTALL)
        if m and "def build" in m.group(1):
            return m.group(1).strip()
    if "def build" in s:
        idx = s.find("import")
        db = s.find("def build")
        if idx < 0 or (db >= 0 and idx > db): idx = db
        return s[idx:].strip() if idx >= 0 else None
    return None

async def call_ds_async(session, task_n, history):
    """One DeepSeek call with full error history."""
    msgs = [{"role": "system", "content": SYSTEM}]
    user = f"ARC task examples:\n{fmt_examples(all_tasks[task_n])}\n\n"
    if history:
        user += "Previous attempts FAILED with these errors:\n"
        for i, err in enumerate(history[-3:]):
            user += f"  Attempt {i+1}: {err}\n"
        user += "\nFix the issue. Try a different approach. "
    user += "Write Python with `def build() -> bytes:` returning ONNX bytes. No markdown."
    msgs.append({"role": "user", "content": user})
    try:
        async with session.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"},
            json={"model": "deepseek-chat", "messages": msgs, "max_tokens": 4000,
                  "temperature": 0.3 + 0.1 * min(len(history), 5)},
            timeout=aiohttp.ClientTimeout(total=180)
        ) as resp:
            data = await resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"
    return "ERROR: no response"

def try_code(tn, code_str):
    code = extract_code(code_str)
    if not code: return None, -1, "no code block in response"
    try:
        ns = {"np": np, "numpy": np, "onnx": onnx,
              "helper": helper, "TensorProto": TensorProto, "numpy_helper": numpy_helper}
        exec(code, ns)
        if "build" not in ns: return None, -1, "build() not defined"
        ob = ns["build"]()
        if not isinstance(ob, (bytes, bytearray)) or len(ob) < 100:
            return None, -1, "build() returned non-bytes or empty"
        ob = bytes(ob)
        c = cost_of(ob)
        if c < 0: return None, -1, "ONNX failed onnx-tool profile - avoid Slice/Pad/Transpose"
        if not validate(ob, all_tasks[tn]):
            return None, -1, "ONNX runs but produces wrong output for some test pairs"
        return ob, c, "OK"
    except Exception as e:
        return None, -1, f"{type(e).__name__}: {str(e)[:200]}"'''))

CELLS.append(("code", r'''# === MAIN LOOP: KEEP TRYING UNTIL TOTAL >= 9000 ===
import time
import asyncio

print("\n[8] === MAIN LOOP: Targeting 9000 total points ===")
print(f"Starting total: {sum(p for _, _, p in best_per_task.values()):.2f}")
print(f"Per-task target: {PER_TASK_TARGET} pts ({len([t for t in best_per_task if best_per_task[t][2] >= PER_TASK_TARGET])}/{len(best_per_task)} already there)")

# Each task tracks its DeepSeek attempt history
task_history = {tn: [] for tn in scoreable_tasks}

async def attempt_task(session, tn):
    """ONE DeepSeek attempt for ONE task. Returns (success, new_pts)."""
    if best_per_task.get(tn, (None, -1, 0))[2] >= PER_TASK_TARGET:
        return False, best_per_task[tn][2]
    code = await call_ds_async(session, tn, task_history[tn])
    (RAW_DIR / f"t{tn:03d}_a{len(task_history[tn])}.txt").write_text(code or "")
    ob, c, reason = try_code(tn, code)
    if ob is None:
        task_history[tn].append(reason)
        return False, best_per_task.get(tn, (None, -1, 0))[2]
    new_pts = pts_of(c)
    cur_pts = best_per_task.get(tn, (None, -1, 0))[2]
    if new_pts > cur_pts:
        best_per_task[tn] = (ob, c, new_pts)
        (SOLVERS_DIR / f"task{tn:03d}.onnx").write_bytes(ob)
        save_state()
        return True, new_pts
    return False, cur_pts

async def round_pass(session, target_tasks):
    """One pass through target tasks - one DS attempt each, parallel up to 30."""
    sem = asyncio.Semaphore(30)
    async def go(tn):
        async with sem:
            return await attempt_task(session, tn)
    return await asyncio.gather(*[go(tn) for tn in target_tasks])

async def main_loop():
    round_n = 0
    start_time = time.time()
    total = sum(p for _, _, p in best_per_task.values())
    async with aiohttp.ClientSession() as session:
        while total < TARGET_TOTAL:
            round_n += 1
            target_tasks = [tn for tn in scoreable_tasks
                            if best_per_task.get(tn, (None, -1, 0))[2] < PER_TASK_TARGET]
            if not target_tasks:
                print("All tasks at target!")
                break
            print(f"\n=== Round {round_n} ===")
            print(f"  Total: {total:.2f}/{TARGET_TOTAL}, attempting {len(target_tasks)} tasks below {PER_TASK_TARGET} pts")
            wins = 0
            results = await round_pass(session, target_tasks)
            for r in results:
                if r[0]: wins += 1
            new_total = sum(p for _, _, p in best_per_task.values())
            elapsed = time.time() - start_time
            print(f"  Round {round_n} done: {wins} new wins, total: {new_total:.2f} (+{new_total - total:.2f}, elapsed {elapsed/60:.1f}min)")
            total = new_total
            # Top 10 progress
            sorted_tasks = sorted(best_per_task.items(), key=lambda x: -x[1][2])[:5]
            top_str = ", ".join(f"t{tn:03d}={p[2]:.1f}" for tn, p in sorted_tasks)
            print(f"  Top 5: {top_str}")
            if elapsed > 7 * 3600:  # 7h max
                print("Time budget exceeded, stopping")
                break
        return total

final_total = asyncio.run(main_loop())
print(f"\n=== LOOP FINISHED ===\nFinal total: {final_total:.2f}")'''))

CELLS.append(("code", r'''# === SUBMIT GATE ===
total = sum(p for _, _, p in best_per_task.values())
print(f"\nFinal total: {total:.2f}")

if total >= TARGET_TOTAL:
    print(f"REACHED TARGET! Submitting to Kaggle...")
    out_zip = WORK / "submission.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(konbu_zip) as zin, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
        for inf in zin.infolist():
            stem = pathlib.Path(inf.filename).stem
            try: tn = int(stem[4:])
            except: tn = -1
            if tn in best_per_task:
                zout.writestr(inf.filename, best_per_task[tn][0])
            else:
                zout.writestr(inf.filename, zin.read(inf.filename))
    out_zip.write_bytes(buf.getvalue())

    r = subprocess.run(
        ["kaggle", "competitions", "submit", "-c", "neurogolf-2026",
         "-f", str(out_zip), "-m", f"v6 infinite-loop: {total:.0f}pts"],
        capture_output=True, text=True, timeout=300
    )
    print(r.stdout)
    if r.stderr: print(r.stderr[:300])
    print(f"\nLeaderboard: https://www.kaggle.com/competitions/neurogolf-2026/submissions")
else:
    print(f"Did NOT reach {TARGET_TOTAL}. Current: {total:.2f}")
    print("Best 50 tasks:")
    sorted_t = sorted(best_per_task.items(), key=lambda x: -x[1][2])[:50]
    for tn, (ob, c, p) in sorted_t:
        print(f"  task{tn:03d}: cost={c}, pts={p:.2f}")
    print(f"\nState saved. Re-run cells 8-12 to continue iterating.")
    print("Or change TARGET_TOTAL to lower value to submit current best.")'''))


def write_nb():
    cells_out = []
    for ct, src in CELLS:
        if ct == "code":
            lines = src.split("\n")
            source = [l + "\n" for l in lines]
            if source: source[-1] = source[-1].rstrip("\n")
            cells_out.append({"cell_type": "code", "source": source,
                            "outputs": [], "execution_count": None, "metadata": {}})
        else:
            cells_out.append({"cell_type": "markdown", "source": [src], "metadata": {}})
    return {
        "cells": cells_out,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
            "colab": {"name": "neurogolf_v6.ipynb", "provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }


if __name__ == "__main__":
    nb = write_nb()
    out = pathlib.Path("neurogolf_v6.ipynb")
    out.write_text(json.dumps(nb, indent=2))
    print(f"Wrote {out} with {len(nb['cells'])} cells")
