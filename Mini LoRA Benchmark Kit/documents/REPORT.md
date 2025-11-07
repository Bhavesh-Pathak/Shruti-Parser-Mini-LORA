# Mini LoRA Benchmark Kit — Report

## Summary
This small project demonstrates a local LoRA fine-tuning workflow using `distilbert-base-uncased` on a tiny 12-sample binary classification dataset. The goal is to show how to apply lightweight LoRA adapters with PEFT, run a short fine-tuning job, and compare inference latency and quality to the unmodified base model.

## What I implemented
- A Jupyter notebook at `notebooks/mini_lora_benchmark.ipynb` that: 
  - Loads a tiny dataset (`data/mini_dataset.csv`).
  - Tokenizes and prepares the data.
  - Evaluates the base DistilBERT classification model on the test split.
  - Applies LoRA (PEFT) to a fresh model copy and fine-tunes it for a few epochs.
  - Benchmarks inference latency and tokens/sec for base vs LoRA models.
  - Saves results to `results/results.json` and `results/results.csv`.

## Key results (example / placeholder)
Run the notebook to produce concrete numbers in `results/results.json` and `results/results.csv`. Expected outputs include:
- `eval_accuracy` and `eval_f1` for both base and LoRA models.
- `base_latency_s`, `lora_latency_s` (average inference latency in seconds for a small batch),
- `base_tokens_per_sec`, `lora_tokens_per_sec`.

## Interpretation guidance
- Because the dataset is extremely small, metric differences are noisy. The notebook is meant to show the pipeline, not to produce production-grade evaluation.
- For generative token-level benchmarks (per-token generation latency), switch to a small causal model (e.g., `distilgpt2`) and measure per-token generation time.

## Next steps (suggestions)
1. Replace the dataset with a larger, task-appropriate dataset (50–1k samples) for better quality comparisons.
2. Switch to a causal LM for per-token generation benchmarks if desired.
3. Try different LoRA configs (r, alpha, target_modules) and measure trade-offs.
4. Optionally enable 8-bit training (bitsandbytes + prepare_model_for_kbit_training) for larger base models.

