"""Generate Kaggle notebook for the GPU training pipeline."""
import json, pathlib

cells = []

def code(src):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': src.strip().split('\n')})
def md(src):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': src.strip().split('\n')})

md("""# NeuroGolf 2026 — GPU ML Training Pipeline v1
For each unsolved task, train tiny Conv networks (3 architectures, 3 seeds).
Filter for: val_acc=100% AND cost < thisray's. Export valid ONNX.

**Inputs needed:** competitions/neurogolf-2026 dataset, thisray's submission.zip uploaded as private dataset.

**Estimated runtime:** 8-12 hrs on T4.

**Expected output:** /kaggle/working/improvements.zip with 50-100 ONNX files cheaper than thisray.""")

code("""
!pip install -q onnx==1.21.0 onnxruntime==1.24.4 onnx-tool==1.0.1 numpy==2.4.4 2>&1 | tail -3
""")

# Read the training script
with open('C:/Users/sarav/AppData/Local/Temp/kaggle_repo/train_pipeline_v1.py') as f:
    train_src = f.read()
code(train_src)

code("""
# Zip up improvements
import zipfile, pathlib
out = pathlib.Path('/kaggle/working')
imp = out / 'improvements'
with zipfile.ZipFile(out / 'improvements.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in imp.glob('task*.onnx'):
        z.write(f, f.name)
print(f"Zipped {len(list(imp.glob('task*.onnx')))} ONNX files")
""")

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10'},
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

out = pathlib.Path('C:/Users/sarav/AppData/Local/Temp/kaggle_repo/neurogolf_train_v1.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
