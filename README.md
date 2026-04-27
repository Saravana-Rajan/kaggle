# NeuroGolf 2026 - Parallel Solver

Push from 6244 (top public) to 7500-9000+ via parallel DeepSeek + GPU.

## Quick Start (Colab - 3 clicks)

1. **Open in Colab**: [Click here to open neurogolf_v2.ipynb in Colab](https://colab.research.google.com/github/Saravana-Rajan/kaggle/blob/main/neurogolf_v2.ipynb)
2. **Add Kaggle key** in Cell 1 (get from https://www.kaggle.com/settings → Create New Token)
3. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**

The notebook will:
- Download ARC-AGI competition data
- Download konbu's 6244-pt base submission
- Call DeepSeek API in parallel (50 concurrent) for 400 tasks
- Validate generated ONNX (cost + correctness)
- Build optimized submission.zip
- Auto-submit to Kaggle

## Files

- `neurogolf_v2.ipynb` — Colab notebook (13 cells, run all)
- `colab_bootstrap_v2.py` — Standalone Python script version

## Expected Score

| Scenario | Score |
|---|---|
| Konbu base | 6244 |
| Realistic | 7000-7800 |
| Good run | 7800-8500 |
| Stretch | 8500-9000+ |
