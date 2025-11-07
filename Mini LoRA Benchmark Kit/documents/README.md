Mini LoRA Benchmark Kit
=======================

This project provides a minimal local LoRA fine-tuning and benchmarking setup using Hugging Face Transformers + PEFT.

Files:
- `notebooks/mini_lora_benchmark.ipynb` — Jupyter notebook with the entire pipeline.
- `data/mini_dataset.csv` — Tiny 12-sample dataset used for the demo.
- `requirements.txt` — Python dependencies.
- `REPORT.md` — Short summary and interpretation guidance.
- `results/` — Notebook will write `results.json` and `results.csv` here after running.

Quick start (Windows PowerShell):
1. Create a new venv and activate it:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install deps:

```powershell
pip install -r requirements.txt
```

3. Start Jupyter and run the notebook:

```powershell
jupyter notebook notebooks\mini_lora_benchmark.ipynb
```

4. Google Collab

```collab
upload the notebook in google collab and run each cell from top to bottom
```

Notes:
- The notebook is designed to run on CPU or GPU; if you have a GPU and PyTorch CUDA installed, it will use it automatically.
- If you want a generative benchmark (per-token generation speed), switch to a causal model like `distilgpt2` or a small Mistral / phi variant.

