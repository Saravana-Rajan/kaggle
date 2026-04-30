"""Generate Kaggle notebook for v100 group submitter."""
import json, pathlib

cells = []

def md(src):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': src.strip().split('\n')})

def code(src):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': src.strip().split('\n')})

md("""# NeuroGolf v100 Group Submitter

Change `GROUP_NUM` below (1-16) for each version. Save & submit each version separately.

**Inputs needed:**
- thisray's submission.zip (named `neurogolf-4808-21-post-apr-2...` or similar)

**Output:** `/kaggle/working/submission.zip` - upload via the Submit button when notebook completes.
""")

code("!pip install -q onnx==1.21.0 2>&1 | tail -1")

with open('C:/Users/sarav/AppData/Local/Temp/kaggle_repo/v100_group_submit.py') as f:
    src = f.read()

code(src)

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10'},
    },
    'nbformat': 4, 'nbformat_minor': 5
}

out = pathlib.Path('C:/Users/sarav/AppData/Local/Temp/kaggle_repo/neurogolf_v100_submit.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
