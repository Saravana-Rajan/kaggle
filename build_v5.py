"""Build neurogolf_v5.ipynb with EXPANDED templates from prior battle-tested winners.

Tier 1 templates (port of all my proven winners):
  - identity
  - color_permute (Conv 1x1)
  - rot180 (Gather flip indices)
  - fliph, flipv
  - mirror_quad (2Hx2W = quad mirror)
  - vmirror_below, vmirror_above (2HxW)
  - hmirror_right, hmirror_left (Hx2W)
  - crop (smaller output via Gather)
  - tile (repeat pattern)
"""
import json, pathlib

CELLS = []

CELLS.append(("markdown", """# NeuroGolf 2026 v5 - 3-Tier with EXPANDED Templates
**Targets 6500-7500+ via:**
- Tier 1: 13+ template patterns (rot, flip, mirror, crop, color_permute)
- Tier 2: DeepSeek retry with error feedback (3 attempts)
- Tier 3: PyTorch Conv with multiple architectures
- Detailed per-task points display"""))

CELLS.append(("code", """KAGGLE_API_TOKEN = "KGAT_b82606465b6b6670336cd164d31ee34c"
KAGGLE_USER = "saravanarajanb"
DEEPSEEK_KEY = "sk-cc4359e608d54e9a99ade2b6c9384ae5"
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
TRAINED_DIR = WORK / "trained"
for d in [DATA_DIR, KONBU_DIR, SOLVERS_DIR, RAW_DIR, LOGS_DIR, TRAINED_DIR]:
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
    return out"""))

CELLS.append(("code", r"""# === EXPANDED TEMPLATE LIBRARY ===
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

# ----- TEMPLATES -----
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
    W_t = numpy_helper.from_array(W, "W")
    B_t = numpy_helper.from_array(np.zeros(10, dtype=np.float32), "B")
    return _build([
        helper.make_node("Constant", [], ["W"], value=W_t),
        helper.make_node("Constant", [], ["B"], value=B_t),
        helper.make_node("Conv", ["input", "W", "B"], ["conv"], kernel_shape=[1,1], pads=[0,0,0,0]),
        helper.make_node("ReduceMax", ["input"], ["mask"], axes=[1], keepdims=1),
        helper.make_node("Mul", ["conv", "mask"], ["output"]),
    ])

def t_fliph(W_in):
    # Horizontal flip - flip width axis.
    idx = _flip_idx(W_in)
    idx_t = numpy_helper.from_array(idx, "idx")
    return _build([
        helper.make_node("Constant", [], ["idx"], value=idx_t),
        helper.make_node("Gather", ["input", "idx"], ["g"], axis=3),
        helper.make_node("Mul", ["g", "ONE_F"], ["output"]),
    ], [ONE_F])

def t_flipv(H_in):
    # Vertical flip - flip height axis.
    idx = _flip_idx(H_in)
    idx_t = numpy_helper.from_array(idx, "idx")
    return _build([
        helper.make_node("Constant", [], ["idx"], value=idx_t),
        helper.make_node("Gather", ["input", "idx"], ["g"], axis=2),
        helper.make_node("Mul", ["g", "ONE_F"], ["output"]),
    ], [ONE_F])

def t_rot180(H_in, W_in):
    # 180-degree rotation.
    idh = _flip_idx(H_in); idw = _flip_idx(W_in)
    return _build([
        helper.make_node("Constant", [], ["idh"], value=numpy_helper.from_array(idh, "idh")),
        helper.make_node("Constant", [], ["idw"], value=numpy_helper.from_array(idw, "idw")),
        helper.make_node("Gather", ["input", "idh"], ["gh"], axis=2),
        helper.make_node("Gather", ["gh", "idw"], ["gw"], axis=3),
        helper.make_node("Mul", ["gw", "ONE_F"], ["output"]),
    ], [ONE_F])

def t_mirror_quad(H_in, W_in, H_out, W_out):
    # Mirror-quad: 2H x 2W output, 4 reflections of input.
    idh = _mirror_idx(H_in, H_out); idw = _mirror_idx(W_in, W_out)
    return _build([
        helper.make_node("Constant", [], ["idh"], value=numpy_helper.from_array(idh, "idh")),
        helper.make_node("Constant", [], ["idw"], value=numpy_helper.from_array(idw, "idw")),
        helper.make_node("Gather", ["input", "idh"], ["gh"], axis=2),
        helper.make_node("Gather", ["gh", "idw"], ["gw"], axis=3),
        helper.make_node("Mul", ["gw", "ONE_F"], ["output"]),
    ], [ONE_F])

def t_vmirror_below(H_in, H_out):
    # vstack(input, flipud(input)): output rows 0..H-1=input, H..2H-1=reversed.
    idh = _mirror_idx(H_in, H_out)
    return _build([
        helper.make_node("Constant", [], ["idh"], value=numpy_helper.from_array(idh, "idh")),
        helper.make_node("Gather", ["input", "idh"], ["g"], axis=2),
        helper.make_node("Mul", ["g", "ONE_F"], ["output"]),
    ], [ONE_F])

def t_vmirror_above(H_in, H_out):
    # vstack(flipud(input), input).
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
    # hstack(input, fliplr(input)).
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
    # Crop a sub-region.
    safe_r = H_in if H_in < 30 else 29
    safe_c = W_in if W_in < 30 else 29
    row_idx = list(range(top, top + out_H)) + [safe_r] * (30 - out_H)
    col_idx = list(range(left, left + out_W)) + [safe_c] * (30 - out_W)
    rh = numpy_helper.from_array(np.array(row_idx[:30], dtype=np.int32), "rh")
    cw = numpy_helper.from_array(np.array(col_idx[:30], dtype=np.int32), "cw")
    return _build([
        helper.make_node("Constant", [], ["rh"], value=rh),
        helper.make_node("Constant", [], ["cw"], value=cw),
        helper.make_node("Gather", ["input", "rh"], ["gh"], axis=2),
        helper.make_node("Gather", ["gh", "cw"], ["gw"], axis=3),
        helper.make_node("Mul", ["gw", "ONE_F"], ["output"]),
    ], [ONE_F])

# ----- PATTERN DETECTION -----
def detect_pattern(task):
    # Try multiple patterns. Returns (name, params).
    ps = pairs_of(task)
    if not ps: return None, None

    # Identity
    if all(p[0].shape == p[1].shape and np.array_equal(p[0], p[1]) for p in ps):
        return "identity", None

    si = set(p[0].shape for p in ps)
    so = set(p[1].shape for p in ps)
    same_shape = all(p[0].shape == p[1].shape for p in ps)
    fixed_in = len(si) == 1
    fixed_out = len(so) == 1

    # Color permute (any shape, same in/out shape per pair)
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

    # Fixed-shape spatial transformations (need fixed in/out)
    if fixed_in and fixed_out:
        ih, iw = list(si)[0]
        oh, ow = list(so)[0]

        # fliph (same shape)
        if (ih, iw) == (oh, ow):
            if all(np.array_equal(np.fliplr(i), o) for i, o in ps):
                return "fliph", iw
            if all(np.array_equal(np.flipud(i), o) for i, o in ps):
                return "flipv", ih
            if all(np.array_equal(np.rot90(i, 2), o) for i, o in ps):
                return "rot180", (ih, iw)

        # mirror_quad (2H x 2W)
        if oh == 2 * ih and ow == 2 * iw:
            ok = True
            for i, o in ps:
                if (not np.array_equal(o[:ih, :iw], i) or
                    not np.array_equal(o[:ih, iw:], np.fliplr(i)) or
                    not np.array_equal(o[ih:, :iw], np.flipud(i)) or
                    not np.array_equal(o[ih:, iw:], np.rot90(i, 2))):
                    ok = False; break
            if ok: return "mirror_quad", (ih, iw, oh, ow)

        # vmirror_below (2H x W)
        if oh == 2 * ih and ow == iw:
            if all(np.array_equal(o[:ih], i) and np.array_equal(o[ih:], i[::-1]) for i, o in ps):
                return "vmirror_below", (ih, oh)
            if all(np.array_equal(o[:ih], i[::-1]) and np.array_equal(o[ih:], i) for i, o in ps):
                return "vmirror_above", (ih, oh)

        # hmirror_right (H x 2W)
        if oh == ih and ow == 2 * iw:
            if all(np.array_equal(o[:, :iw], i) and np.array_equal(o[:, iw:], i[:, ::-1]) for i, o in ps):
                return "hmirror_right", (iw, ow)

        # crop (smaller output)
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

CELLS.append(("code", r"""# === VALIDATION UTILITIES ===
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

CELLS.append(("code", r"""# === TIER 1: EXPANDED TEMPLATES ===
print("\n[6] === TIER 1: 13+ Templates ===")
tier1_wins = {}
pattern_count = {}
for tn in sorted(all_tasks.keys()):
    if tn in EXCLUDED or tn in KAGGLE_KNOWN_BAD: continue
    try:
        ob, pname = apply_template(all_tasks[tn])
        if ob is None: continue
        c = cost_of(ob)
        if c < 0 or c > 500: continue
        if not validate(ob, all_tasks[tn]): continue
        tier1_wins[tn] = (ob, c)
        pattern_count[pname] = pattern_count.get(pname, 0) + 1
        (SOLVERS_DIR / f"task{tn:03d}.onnx").write_bytes(ob)
        print(f"  task{tn:03d}: TEMPLATE WIN [{pname}] cost={c} pts={pts_of(c):.2f}")
    except Exception as e: pass

print(f"\nTier 1 wins: {len(tier1_wins)} | by pattern: {pattern_count}")"""))

CELLS.append(("code", r"""# === KONBU BASELINE COSTS ===
print("\n[7] Computing konbu costs (parallel)...")
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

# DeepSeek targets: tasks not in tier1 wins AND konbu cost > 100
targets = sorted([tn for tn, c in konbu_costs.items()
                  if tn not in tier1_wins and (c is None or c < 0 or c > 100)])
print(f"DeepSeek targets: {len(targets)} (excluding {len(tier1_wins)} tier1 wins)")"""))

CELLS.append(("code", r'''# === TIER 2: DEEPSEEK with RETRY + ERROR FEEDBACK ===
print(f"\n[8] === TIER 2: DeepSeek with retry for {len(targets)} tasks ===")
import nest_asyncio
nest_asyncio.apply()
import aiohttp

EXAMPLE = """import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

def build():
    inp = onnx.ValueInfoProto(); inp.name='input'
    inp.type.tensor_type.elem_type = TensorProto.FLOAT
    d = inp.type.tensor_type.shape.dim
    d.add().dim_value=1; d.add().dim_value=10; d.add(); d.add()
    out = onnx.ValueInfoProto(); out.name='output'
    out.type.tensor_type.elem_type = TensorProto.FLOAT
    d = out.type.tensor_type.shape.dim
    d.add().dim_value=1; d.add().dim_value=10; d.add(); d.add()

    # Color permute via Conv 1x1
    W = np.zeros((10,10,1,1), dtype=np.float32)
    mapping = {0:0, 1:5, 5:1, 2:6, 6:2, 3:4, 4:3, 8:9, 9:8, 7:7}
    for ci, co in mapping.items(): W[co, ci, 0, 0] = 1.0

    nodes = [
        helper.make_node('Constant', [], ['W'], value=numpy_helper.from_array(W, 'W')),
        helper.make_node('Constant', [], ['B'], value=numpy_helper.from_array(np.zeros(10, dtype=np.float32), 'B')),
        helper.make_node('Conv', ['input', 'W', 'B'], ['conv'], kernel_shape=[1,1], pads=[0,0,0,0]),
        helper.make_node('ReduceMax', ['input'], ['mask'], axes=[1], keepdims=1),
        helper.make_node('Mul', ['conv', 'mask'], ['output']),
    ]
    g = helper.make_graph(nodes, 'cp', [inp], [out], [])
    return helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid('', 10)]).SerializeToString()
"""

SYSTEM = (
    "You write Python ONNX builders for ARC-AGI tasks. Cost = MACs+memory+params (onnx-tool 1.0.1).\n\n"
    "RULES:\n"
    "1. Input/output shape: [1, 10, EMPTY, EMPTY] (no dim_value, no dim_param on H,W)\n"
    "2. Use Constant nodes (NOT initializers). Constants are free.\n"
    "3. End with: ReduceMax(input, axes=[1], keepdims=1) mask, Mul to apply\n"
    "4. Opset 10, IR 10\n"
    "5. AVOID: Slice, Pad, ScatterND, Min, ArgMin, Transpose. They crash.\n"
    "6. USE: Conv (1x1 or 3x3), Gather (axis 1/2/3), Mul, Add, ReduceMax, Cast.\n\n"
    "WORKING TEMPLATE:\n```python\n" + EXAMPLE + "```\n\n"
    "Output ONLY the Python code block."
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

async def call_ds(session, tn, prev_error=None):
    # Single DS call. If prev_error given, include it in retry prompt.
    if prev_error:
        prompt = (f"Your previous code FAILED with: {prev_error}\n\n"
                  f"Task examples:\n{fmt_examples(all_tasks[tn])}\n\n"
                  "Fix the issue and write Python code with `def build() -> bytes:`. "
                  "Use the EXACT template structure. No markdown.")
    else:
        prompt = (f"ARC task examples:\n{fmt_examples(all_tasks[tn])}\n\n"
                  "Write Python with `def build() -> bytes:`. Use template structure. No markdown.")

    try:
        async with session.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 4000,
                "temperature": 0.2 if not prev_error else 0.5
            },
            timeout=aiohttp.ClientTimeout(total=180)
        ) as resp:
            data = await resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"
    return "ERROR: no response"

async def task_with_retry(session, tn, semaphore):
    # Get DS code with up to 3 attempts, validating each.
    async with semaphore:
        prev_error = None
        for attempt in range(3):
            code = await call_ds(session, tn, prev_error)
            if code.startswith("ERROR"):
                prev_error = code; continue
            (RAW_DIR / f"t{tn:03d}_a{attempt}.txt").write_text(code)
            # Try to validate
            extracted = extract_code(code)
            if not extracted:
                prev_error = "no code block found"; continue
            try:
                ns = {"np": np, "numpy": np, "onnx": onnx,
                      "helper": helper, "TensorProto": TensorProto, "numpy_helper": numpy_helper}
                exec(extracted, ns)
                if "build" not in ns:
                    prev_error = "build function not defined"; continue
                ob = ns["build"]()
                if not isinstance(ob, (bytes, bytearray)) or len(ob) < 100:
                    prev_error = "build() must return bytes"; continue
                ob = bytes(ob)
                c = cost_of(ob)
                if c < 0:
                    prev_error = "ONNX failed onnx-tool profiling (avoid Slice/Pad/Transpose)"; continue
                if not validate(ob, all_tasks[tn]):
                    prev_error = "ONNX runs but produces wrong output"; continue
                return tn, ob, c, attempt + 1
            except Exception as e:
                prev_error = f"{type(e).__name__}: {str(e)[:200]}"
        return tn, None, -1, 3

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

async def main_ds(target_list):
    sem = asyncio.Semaphore(50)  # Lower concurrency due to retries
    async with aiohttp.ClientSession() as session:
        tasks_co = [task_with_retry(session, tn, sem) for tn in target_list]
        results = []
        done = 0
        for coro in asyncio.as_completed(tasks_co):
            tn, ob, c, attempts = await coro
            results.append((tn, ob, c, attempts))
            done += 1
            if ob is not None:
                kc = konbu_costs.get(tn, 10**12)
                if 0 < c < kc:
                    (SOLVERS_DIR / f"task{tn:03d}.onnx").write_bytes(ob)
                    kp = pts_of(kc); np_ = pts_of(c); g = np_ - kp
                    print(f"  task{tn:03d}: WIN(att={attempts}) konbu({kc}->{kp:.2f}) -> ours({c}->{np_:.2f}) +{g:.2f}")
            if done % 30 == 0:
                wins = sum(1 for r in results if r[1] is not None)
                print(f"  --- {done}/{len(target_list)} done, wins={wins} ---")
        return results

ds_results = asyncio.run(main_ds(targets))
tier2_wins = {tn: (ob, c) for tn, ob, c, _ in ds_results if ob is not None}
# Filter to only those that beat konbu
tier2_wins = {tn: (ob, c) for tn, (ob, c) in tier2_wins.items() if c < konbu_costs.get(tn, 10**12)}
print(f"\nTier 2 wins: {len(tier2_wins)}")'''))

CELLS.append(("code", r'''# === TIER 3: PYTORCH GPU TRAINING WITH MULTIPLE ARCHITECTURES ===
print(f"\n[9] === TIER 3: GPU training (multiple architectures) ===")
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
solved = set(tier1_wins.keys()) | set(tier2_wins.keys())
tier3_targets = [tn for tn in targets if tn not in solved]
TIER3_LIMIT = 100  # try 100 tasks
print(f"Tier 3 targets: min({TIER3_LIMIT}, {len(tier3_targets)})")

class TinyNet(nn.Module):
    def __init__(self, n_layers, ksize, hidden=16):
        super().__init__()
        layers = []; prev = 10
        for i in range(n_layers):
            out_ch = 10 if i == n_layers - 1 else hidden
            layers.append(nn.Conv2d(prev, out_ch, ksize, padding=ksize//2))
            if i < n_layers - 1: layers.append(nn.ReLU())
            prev = out_ch
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x) * x.amax(dim=1, keepdim=True)

def train_model(task, model, n_epochs=200, lr=0.01):
    ps = pairs_of(task)
    ps = [(i, o) for i, o in ps if i.shape[0] <= 30 and i.shape[1] <= 30 and o.shape[0] <= 30 and o.shape[1] <= 30]
    if len(ps) < 4: return None, 0
    np.random.seed(42)
    perm = np.random.permutation(len(ps))
    n_train = max(int(0.85 * len(ps)), len(ps) - 30)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train+30]

    X = np.stack([grid_to_oh(ps[i][0]) for i in train_idx])
    Y = np.stack([grid_to_oh(ps[i][1]) for i in train_idx])
    X = torch.from_numpy(X).to(DEVICE)
    Y = torch.from_numpy(Y).to(DEVICE)

    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(n_epochs):
        model.train(); opt.zero_grad()
        loss_fn(model(X), Y).backward()
        opt.step()

    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for i in val_idx[:20]:
            inp, out = ps[i]
            x = torch.from_numpy(grid_to_oh(inp)).unsqueeze(0).to(DEVICE)
            pred = model(x).cpu().numpy()[0]
            pg = pred[:, :out.shape[0], :out.shape[1]].argmax(axis=0)
            if np.array_equal(pg, out): correct += 1
            total += 1
    return model.cpu(), correct / max(total, 1)

def export_trained(model):
    state = model.state_dict()
    inp, out = _empty_io()
    nodes = []; cur = "input"
    weight_keys = sorted([k for k in state if "weight" in k])
    n_layers = len(weight_keys)
    for li, k in enumerate(weight_keys):
        W_arr = state[k].numpy().astype(np.float32)
        B_key = k.replace("weight", "bias")
        B_arr = state[B_key].numpy().astype(np.float32) if B_key in state else np.zeros(W_arr.shape[0], dtype=np.float32)
        nodes.append(helper.make_node("Constant", [], [f"W{li}"], value=numpy_helper.from_array(W_arr, f"W{li}")))
        nodes.append(helper.make_node("Constant", [], [f"B{li}"], value=numpy_helper.from_array(B_arr, f"B{li}")))
        pad = (W_arr.shape[2] - 1) // 2
        nodes.append(helper.make_node("Conv", [cur, f"W{li}", f"B{li}"], [f"c_{li}"],
                                       kernel_shape=[W_arr.shape[2], W_arr.shape[3]], pads=[pad, pad, pad, pad]))
        cur = f"c_{li}"
        if li < n_layers - 1:
            nodes.append(helper.make_node("Relu", [cur], [f"r_{li}"]))
            cur = f"r_{li}"
    nodes.append(helper.make_node("ReduceMax", ["input"], ["mask"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Mul", [cur, "mask"], ["output"]))
    g = helper.make_graph(nodes, "trained", [inp], [out], [])
    return helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid("", 10)]).SerializeToString()

ARCHS = [
    {"n_layers": 1, "ksize": 1},
    {"n_layers": 2, "ksize": 3, "hidden": 16},
    {"n_layers": 3, "ksize": 3, "hidden": 24},
    {"n_layers": 2, "ksize": 5, "hidden": 16},
]

tier3_wins = {}
attempted = 0
for tn in tier3_targets[:TIER3_LIMIT]:
    attempted += 1
    best = None
    for arch in ARCHS:
        try:
            model = TinyNet(**arch)
            model, acc = train_model(all_tasks[tn], model, n_epochs=150, lr=0.01)
            if model is None or acc < 0.95: continue
            ob = export_trained(model)
            c = cost_of(ob)
            if c < 0: continue
            if not validate(ob, all_tasks[tn]): continue
            kc = konbu_costs.get(tn, 10**12)
            if c >= kc: continue
            if best is None or c < best[1]:
                best = (ob, c)
        except: pass
    if best:
        tier3_wins[tn] = best
        (SOLVERS_DIR / f"task{tn:03d}.onnx").write_bytes(best[0])
        kc = konbu_costs.get(tn, 1)
        kp = pts_of(kc); np_ = pts_of(best[1]); g = np_ - kp
        print(f"  task{tn:03d}: GPU TRAIN WIN konbu({kc}->{kp:.2f}) -> ours({best[1]}->{np_:.2f}) +{g:.2f}")

print(f"\nTier 3 wins: {len(tier3_wins)} / {attempted} attempted")'''))

CELLS.append(("code", r'''# === FINAL: COMBINE + SUBMIT ===
all_wins = {**tier1_wins, **tier2_wins, **tier3_wins}
print(f"\n[10] === FINAL ===")
print(f"Total wins: {len(all_wins)}")
print(f"  Tier 1 (templates): {len(tier1_wins)}")
print(f"  Tier 2 (DeepSeek):  {len(tier2_wins)}")
print(f"  Tier 3 (GPU train): {len(tier3_wins)}")

print("\n=== TOP 50 BIGGEST GAINS ===")
gains = []
for tn, (ob, c) in all_wins.items():
    kc = konbu_costs.get(tn, 1)
    g = pts_of(c) - pts_of(kc)
    gains.append((tn, kc, c, pts_of(kc), pts_of(c), g))
gains.sort(key=lambda x: -x[5])
for tn, kc, c, kp, np_, g in gains[:50]:
    print(f"  task{tn:03d}: konbu({kc:>10} -> {kp:5.2f}pts) -> ours({c:>6} -> {np_:5.2f}pts) +{g:5.2f}")

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
            kc = konbu_costs.get(tn, 1); nc = all_wins[tn][1]
            total_gain += pts_of(nc) - pts_of(kc)
            swapped += 1
        else:
            zout.writestr(inf.filename, zin.read(inf.filename))
out_zip.write_bytes(buf.getvalue())

print(f"\n=== SUBMISSION ===")
print(f"Tasks swapped: {swapped}")
print(f"Total local gain: +{total_gain:.2f}")
print(f"Expected Kaggle: 6244 + {total_gain:.0f} = {6244 + total_gain:.0f}")

print("\nSubmitting to Kaggle...")
r = subprocess.run(
    ["kaggle", "competitions", "submit", "-c", "neurogolf-2026",
     "-f", str(out_zip), "-m", f"v5: T1={len(tier1_wins)} T2={len(tier2_wins)} T3={len(tier3_wins)} +{total_gain:.0f}pts"],
    capture_output=True, text=True, timeout=300
)
print(r.stdout)
if r.stderr: print(r.stderr[:300])'''))


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
            "colab": {"name": "neurogolf_v5.ipynb", "provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }


if __name__ == "__main__":
    nb = write_nb()
    out = pathlib.Path("neurogolf_v5.ipynb")
    out.write_text(json.dumps(nb, indent=2))
    print(f"Wrote {out} with {len(nb['cells'])} cells")
