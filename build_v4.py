"""Build neurogolf_v4.ipynb with Tier 1+2+3 and detailed per-task prints."""
import json, pathlib

CELLS = []

CELLS.append(("markdown", """# NeuroGolf 2026 v4 - 3-Tier Solver with GPU
**Targets 8000-9500 via:**
- Tier 1: Template library (color permute, identity)
- Tier 2: Few-shot DeepSeek (50 concurrent)
- Tier 3: PyTorch Conv training on GPU
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

# Check GPU
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

def template_color_permute(mapping):
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
    return helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid("", 10)]).SerializeToString()

def template_identity():
    inp, out = _empty_io()
    one_t = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one")
    nodes = [
        helper.make_node("Constant", [], ["one"], value=one_t),
        helper.make_node("Mul", ["input", "one"], ["output"]),
    ]
    g = helper.make_graph(nodes, "id", [inp], [out], [])
    return helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid("", 10)]).SerializeToString()

def detect_pattern(task):
    ps = pairs_of(task)
    if not ps: return None, None
    if all(p[0].shape == p[1].shape and np.array_equal(p[0], p[1]) for p in ps):
        return "identity", None
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
    pname, args = detect_pattern(task)
    if pname == "identity": return template_identity()
    if pname == "color_permute": return template_color_permute(args)
    return None"""))

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

CELLS.append(("code", r"""# === TIER 1: TEMPLATES (with detailed prints) ===
print("\n[6] === TIER 1: Templates ===")
tier1_wins = {}
checked = 0
for tn in sorted(all_tasks.keys()):
    if tn in EXCLUDED or tn in KAGGLE_KNOWN_BAD: continue
    checked += 1
    try:
        ob = apply_template(all_tasks[tn])
        if ob is None: continue
        c = cost_of(ob)
        if c < 0 or c > 200: continue
        if not validate(ob, all_tasks[tn]): continue
        tier1_wins[tn] = (ob, c)
        (SOLVERS_DIR / f"task{tn:03d}.onnx").write_bytes(ob)
        print(f"  task{tn:03d}: TEMPLATE WIN cost={c} pts={pts_of(c):.2f}")
    except Exception as e: pass

print(f"\nTier 1 summary: {len(tier1_wins)} wins / {checked} checked")"""))

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

# DeepSeek targets: high-cost konbu tasks not yet won by tier1
targets = sorted([tn for tn, c in konbu_costs.items()
                  if tn not in tier1_wins and (c is None or c < 0 or c > 100)])
print(f"DeepSeek targets: {len(targets)} (excluding {len(tier1_wins)} tier1 wins)")"""))

CELLS.append(("code", r'''# === TIER 2: FEW-SHOT DEEPSEEK ===
print(f"\n[8] === TIER 2: DeepSeek for {len(targets)} tasks (50 concurrent) ===")
import nest_asyncio
nest_asyncio.apply()
import aiohttp

EXAMPLE_TEMPLATE = """import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def build():
    inp = onnx.ValueInfoProto(); inp.name = 'input'
    inp.type.tensor_type.elem_type = TensorProto.FLOAT
    d = inp.type.tensor_type.shape.dim
    d.add().dim_value = 1; d.add().dim_value = 10; d.add(); d.add()
    out = onnx.ValueInfoProto(); out.name = 'output'
    out.type.tensor_type.elem_type = TensorProto.FLOAT
    d = out.type.tensor_type.shape.dim
    d.add().dim_value = 1; d.add().dim_value = 10; d.add(); d.add()

    W = np.zeros((10, 10, 1, 1), dtype=np.float32)
    mapping = {0: 0, 1: 5, 5: 1, 2: 6, 6: 2, 3: 4, 4: 3, 8: 9, 9: 8, 7: 7}
    for c_in, c_out in mapping.items():
        W[c_out, c_in, 0, 0] = 1.0

    W_t = numpy_helper.from_array(W, 'W')
    B_t = numpy_helper.from_array(np.zeros(10, dtype=np.float32), 'B')
    nodes = [
        helper.make_node('Constant', [], ['W'], value=W_t),
        helper.make_node('Constant', [], ['B'], value=B_t),
        helper.make_node('Conv', ['input', 'W', 'B'], ['conv'], kernel_shape=[1,1], pads=[0,0,0,0]),
        helper.make_node('ReduceMax', ['input'], ['mask'], axes=[1], keepdims=1),
        helper.make_node('Mul', ['conv', 'mask'], ['output']),
    ]
    g = helper.make_graph(nodes, 'cp', [inp], [out], [])
    m = helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid('', 10)])
    return m.SerializeToString()
"""

SYSTEM = (
    "You write Python that builds minimal ONNX models for ARC-AGI tasks.\n\n"
    "CRITICAL: Output a complete Python script with `def build() -> bytes:`\n"
    "1. Use [1, 10, EMPTY, EMPTY] input/output shape (no dim_value, no dim_param on H,W)\n"
    "2. Use Constant nodes (NOT initializers) for weights\n"
    "3. End with ReduceMax(axes=[1], keepdims=1) mask + Mul to zero padding\n"
    "4. Opset 10, IR 10\n\n"
    "WORKING EXAMPLE for color permute task:\n```python\n" + EXAMPLE_TEMPLATE + "```\n\n"
    "AVOID: Slice, Pad, ScatterND, Min, ArgMin, Transpose, Reshape with explicit shapes.\n"
    "USE: Conv, Gather (axis=1 only), Mul, Add, ReduceMax, Cast, Concat.\n\n"
    "Output ONLY a Python code block - no explanations."
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

PROMPT = ("ARC task examples:\n{examples}\n\nWrite Python with `def build() -> bytes:` "
          "using the example template structure. Modify it for this task.")

async def call_ds(session, tn, semaphore):
    async with semaphore:
        try:
            async with session.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": PROMPT.format(examples=fmt_examples(all_tasks[tn]))}
                    ],
                    "max_tokens": 4000, "temperature": 0.2
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
            if done % 30 == 0:
                print(f"  {done}/{len(target_list)} DeepSeek calls done")
        return results

ds_results = asyncio.run(main_ds(targets))
print(f"DeepSeek responses: {len(ds_results)}")'''))

CELLS.append(("code", r'''# === VALIDATE DEEPSEEK with detailed per-task prints ===
print(f"\n[9] Validating {len(ds_results)} DeepSeek responses...")
def extract_code(s):
    if not s or s.startswith("ERROR"): return None
    for pat in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        m = re.search(pat, s, re.DOTALL)
        if m and "def build" in m.group(1):
            return m.group(1).strip()
    if "def build" in s:
        idx = s.find("import")
        if idx < 0 or (s.find("def build") >= 0 and idx > s.find("def build")):
            idx = s.find("def build")
        return s[idx:].strip() if idx >= 0 else None
    return None

def try_one(tn, code_str):
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
        if not validate(ob, all_tasks[tn]): return tn, None, -1, "bad_validation"
        return tn, ob, c, "ok"
    except Exception as e:
        return tn, None, -1, f"exec_error:{type(e).__name__}"

tier2_wins = {}
failure_breakdown = {}
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(try_one, tn, code): tn for tn, code in ds_results}
    done = 0
    for fut in as_completed(futures):
        try:
            tn, ob, c, reason = fut.result(timeout=60)
            failure_breakdown[reason] = failure_breakdown.get(reason, 0) + 1
            if ob is not None:
                kc = konbu_costs.get(tn, 10**12)
                if 0 < c < kc:
                    tier2_wins[tn] = (ob, c)
                    (SOLVERS_DIR / f"task{tn:03d}.onnx").write_bytes(ob)
                    kp = pts_of(kc); np_ = pts_of(c); g = np_ - kp
                    print(f"  task{tn:03d}: DEEPSEEK WIN konbu({kc}->{kp:.2f}) -> ours({c}->{np_:.2f}) +{g:.2f}")
        except Exception as e:
            failure_breakdown["future_err"] = failure_breakdown.get("future_err", 0) + 1
        done += 1
        if done % 50 == 0:
            print(f"  --- {done}/{len(futures)} done, wins={len(tier2_wins)} ---")

print(f"\nTier 2 wins: {len(tier2_wins)}")
print(f"Failure breakdown: {failure_breakdown}")'''))

CELLS.append(("code", r'''# === TIER 3: PYTORCH GPU TRAINING (color permute fallback) ===
print(f"\n[10] === TIER 3: PyTorch GPU training for unsolved tasks ===")
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training device: {DEVICE}")

# Targets for tier 3: tasks NOT yet solved by tier 1 or tier 2
solved_already = set(tier1_wins.keys()) | set(tier2_wins.keys())
tier3_targets = [tn for tn in targets if tn not in solved_already]
print(f"Tier 3 targets: {len(tier3_targets)}")

class TinyConv(nn.Module):
    def __init__(self, n_layers=2, ksize=3, hidden=16):
        super().__init__()
        layers = []
        prev = 10
        for i in range(n_layers):
            out_ch = 10 if i == n_layers - 1 else hidden
            layers.append(nn.Conv2d(prev, out_ch, ksize, padding=ksize//2))
            if i < n_layers - 1: layers.append(nn.ReLU())
            prev = out_ch
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        out = self.net(x)
        mask = x.amax(dim=1, keepdim=True)
        return out * mask

def train_solver(task, n_epochs=200, lr=0.01):
    ps = pairs_of(task)
    ps = [(i, o) for i, o in ps if i.shape[0] <= 30 and i.shape[1] <= 30 and o.shape[0] <= 30 and o.shape[1] <= 30]
    if len(ps) < 4: return None, 0.0
    np.random.seed(42)
    perm = np.random.permutation(len(ps))
    n_train = max(int(0.85 * len(ps)), len(ps) - 30)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train+30]

    X = np.stack([grid_to_oh(ps[i][0]) for i in train_idx])
    Y = np.stack([grid_to_oh(ps[i][1]) for i in train_idx])
    X = torch.from_numpy(X).to(DEVICE)
    Y = torch.from_numpy(Y).to(DEVICE)

    model = TinyConv(n_layers=2, ksize=3, hidden=16).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()

    # Validate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in val_idx[:20]:
            inp, out = ps[i]
            x = torch.from_numpy(grid_to_oh(inp)).unsqueeze(0).to(DEVICE)
            pred = model(x).cpu().numpy()[0]
            pg = pred[:, :out.shape[0], :out.shape[1]].argmax(axis=0)
            if np.array_equal(pg, out): correct += 1
            total += 1
    return model.cpu(), correct/max(total,1)

def export_trained(model):
    state = model.state_dict()
    inp, out = _empty_io()
    nodes = []
    cur = "input"
    weight_keys = sorted([k for k in state if "weight" in k])
    n_layers = len(weight_keys)
    for li, k in enumerate(weight_keys):
        W_arr = state[k].numpy().astype(np.float32)
        B_key = k.replace("weight", "bias")
        B_arr = state[B_key].numpy().astype(np.float32) if B_key in state else np.zeros(W_arr.shape[0], dtype=np.float32)
        W_t = numpy_helper.from_array(W_arr, f"W{li}")
        B_t = numpy_helper.from_array(B_arr, f"B{li}")
        nodes.append(helper.make_node("Constant", [], [f"W{li}"], value=W_t))
        nodes.append(helper.make_node("Constant", [], [f"B{li}"], value=B_t))
        pad = (W_arr.shape[2]-1)//2
        nodes.append(helper.make_node("Conv", [cur, f"W{li}", f"B{li}"], [f"conv_{li}"],
                                       kernel_shape=[W_arr.shape[2], W_arr.shape[3]], pads=[pad,pad,pad,pad]))
        cur = f"conv_{li}"
        if li < n_layers - 1:
            nodes.append(helper.make_node("Relu", [cur], [f"r_{li}"]))
            cur = f"r_{li}"
    nodes.append(helper.make_node("ReduceMax", ["input"], ["mask"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Mul", [cur, "mask"], ["output"]))
    g = helper.make_graph(nodes, "trained", [inp], [out], [])
    return helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid("", 10)]).SerializeToString()

# Train tier 3 solvers (limit to 50 fastest tasks for time budget)
tier3_wins = {}
attempted = 0
TIER3_LIMIT = 50  # train at most this many
for tn in tier3_targets[:TIER3_LIMIT]:
    attempted += 1
    try:
        model, val_acc = train_solver(all_tasks[tn], n_epochs=200, lr=0.01)
        if model is None or val_acc < 0.95: continue
        ob = export_trained(model)
        c = cost_of(ob)
        if c < 0: continue
        if not validate(ob, all_tasks[tn]): continue
        kc = konbu_costs.get(tn, 10**12)
        if c < kc:
            tier3_wins[tn] = (ob, c)
            (SOLVERS_DIR / f"task{tn:03d}.onnx").write_bytes(ob)
            kp = pts_of(kc); np_ = pts_of(c); g = np_ - kp
            print(f"  task{tn:03d}: GPU TRAIN WIN konbu({kc}->{kp:.2f}) -> ours({c}->{np_:.2f}) +{g:.2f} val_acc={val_acc:.0%}")
    except Exception as e:
        pass

print(f"\nTier 3 wins: {len(tier3_wins)} / {attempted} attempted")'''))

CELLS.append(("code", r'''# === FINAL: COMBINE + DETAILED REPORT + SUBMIT ===
all_wins = {**tier1_wins, **tier2_wins, **tier3_wins}
print(f"\n[11] === FINAL ===")
print(f"Total wins: {len(all_wins)}")
print(f"  Tier 1 (templates): {len(tier1_wins)}")
print(f"  Tier 2 (DeepSeek):  {len(tier2_wins)}")
print(f"  Tier 3 (GPU train): {len(tier3_wins)}")

# Detailed gain table
print("\n=== TOP 30 BIGGEST GAINS ===")
gains = []
for tn, (ob, c) in all_wins.items():
    kc = konbu_costs.get(tn, 1)
    g = pts_of(c) - pts_of(kc)
    gains.append((tn, kc, c, pts_of(kc), pts_of(c), g))
gains.sort(key=lambda x: -x[5])
for tn, kc, c, kp, np_, g in gains[:30]:
    print(f"  task{tn:03d}: konbu({kc:>10} cost->{kp:5.2f}pts) -> ours({c:>6} cost->{np_:5.2f}pts) +{g:5.2f}")

# Build submission.zip
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
print(f"Total gain: +{total_gain:.2f} pts")
print(f"Expected Kaggle: 6244 + {total_gain:.0f} = {6244 + total_gain:.0f}")

print("\nSubmitting to Kaggle...")
r = subprocess.run(
    ["kaggle", "competitions", "submit", "-c", "neurogolf-2026",
     "-f", str(out_zip), "-m", f"v4 3tier: {swapped} swaps T1={len(tier1_wins)} T2={len(tier2_wins)} T3={len(tier3_wins)} +{total_gain:.0f}pts"],
    capture_output=True, text=True, timeout=300
)
print(r.stdout)
if r.stderr: print(r.stderr[:300])
print("\nLive scoreboard: https://www.kaggle.com/competitions/neurogolf-2026/submissions")'''))


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

    nb = {
        "cells": cells_out,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
            "colab": {"name": "neurogolf_v4.ipynb", "provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    return nb


if __name__ == "__main__":
    nb = write_nb()
    out = pathlib.Path("neurogolf_v4.ipynb")
    out.write_text(json.dumps(nb, indent=2))
    print(f"Wrote {out} with {len(nb['cells'])} cells")
