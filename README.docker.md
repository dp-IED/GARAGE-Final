# Docker setup (LM Studio + granite-4.0-h-micro)

This project can run fully in Docker with:
- an LM Studio headless server (`lms`/`llmster`) exposing OpenAI-compatible `/v1` endpoints,
- automatic model download for **only** `ibm-granite/granite-4.0-h-micro-GGUF@q4_k_m`,
- strict JSON schema mode for evals and diagnostics.

## 1) Build images

```bash
docker compose build
```

## 2) Launch dashboard (with LM Studio server)

```bash
docker compose up dashboard
```

Open: <http://localhost:8501>

The first startup downloads the model and can take several minutes.

## 3) Run evaluations with existing checkpoints

Run both pipelines:

```bash
docker compose run --rm garage eval-all
```

Run individually:

```bash
docker compose run --rm garage eval-baseline
docker compose run --rm garage eval-kag
```

Outputs are written to:
- `results-new/results_llm_baseline_docker.json`
- `results-new/results_gdn_kg_llm_docker.json`

### Compare overall metrics across methods

Use the utility below to summarise and compare overall metrics side-by-side:

```bash
python llm/evaluation/summarise_method_metrics.py \
  results-new/results_llm_baseline.json \
  results-new/results_gdn_kg_llm.json \
  results-new/results_gdn_only.json
```

Or auto-discover result JSONs in `results-new/`:

```bash
python llm/evaluation/summarise_method_metrics.py
```

Inside Docker:

```bash
docker compose run --rm garage shell -lc "python llm/evaluation/summarise_method_metrics.py"
```

## 4) Retrain models

Stage 1:

```bash
docker compose run --rm garage train-stage1 --epochs 75
```

Stage 2:

```bash
docker compose run --rm garage train-stage2 --epochs 40
```

Defaults inside containers:
- training data dir: `llm/evaluation/shared_dataset`
- stage1 checkpoint: `checkpoints/stage1_best_forecast_20260302_223838.pt`
- stage2 checkpoint: `checkpoints/stage2_clean_20260302_223838.pt`

## 5) Optional: rebuild demo cache

```bash
docker compose run --rm garage build-cache
```

## Notes

- LM Studio API is exposed at `http://localhost:1234/v1` on the host.
- The stack is pinned to Granite 4.0 H Micro; changing to another model is blocked by the LM Studio entrypoint.
