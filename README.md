# Time-series foundation models vs. the classics — restaurant demand

Reading-group notebook. It simulates daily store × item restaurant demand under five sets of
assumptions, then backtests twelve forecasting methods on it:

| Family | Methods |
|---|---|
| Baseline | Seasonal naive |
| Classical (local) | AutoETS, AutoARIMA, MSTL, Croston-SBA (`statsforecast`) |
| ML (global) | LightGBM direct multi-horizon quantile regression |
| Bayesian | Hierarchical negative-binomial model with a random-walk store level (PyMC + nutpie) |
| Foundation (zero-shot) | Chronos-Bolt, TimesFM 2.5, Chronos-2 (univariate / + covariates / + cross-learning) |
| Reference | Oracle (true simulated mean and noise) and an ensemble |

Scenarios: `baseline`, `drivers` (promotions, weather, holiday closures), `regime_shift`,
`intermittent`, and `cold_start`.

## Run it

Requires [uv](https://docs.astral.sh/uv/). Everything is pinned in `uv.lock`, and the dependency
resolution is frozen in time with `exclude-newer` in `pyproject.toml`.

```bash
cd reading-group/time-series-foundation
uv sync                      # creates .venv with the exact locked versions (Python 3.12)
uv run jupyter lab           # open restaurant_demand_forecasting.ipynb
```

Or execute it headless:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 restaurant_demand_forecasting.ipynb
```

* **Runtime.** A full run from scratch takes about 45 minutes on a 12-core Apple-silicon laptop
  (≈9 min per scenario), roughly 60% of it on the Bayesian model (15 NUTS fits). On a laptop,
  prefix the command with `caffeinate -i` so macOS doesn't sleep and pause the run.
* **Cache.** The repo ships with the forecasts from the full run in `results/full/`, so executing
  the notebook only re-scores and re-plots (~2 minutes, mostly loading the foundation models).
  Delete that folder, or set `RECOMPUTE = True`, to recompute.
* **Smoke test.** `FAST=1 uv run jupyter lab` (or prefix the nbconvert command) uses one
  origin and shorter MCMC, and writes to `results/fast/`. It takes about 12 minutes.
* **Model weights** (Chronos-2, Chronos-Bolt-base, TimesFM 2.5; ~1 GB total) download from
  Hugging Face on first run. TimesFM 3.0 is deliberately not used: its weights are licensed for
  non-commercial use only.

## Files

| File | What |
|---|---|
| `restaurant_demand_forecasting.ipynb` | the notebook |
| `restaurant_demand_forecasting.py` | the same notebook in jupytext percent format (diff-friendly; edit either, then `uv run jupytext --sync restaurant_demand_forecasting.ipynb`) |
| `foundation_models.py` | Chronos / TimesFM wrappers, run in a separate worker process |
| `results/full/` | cached forecasts and Bayesian sampler diagnostics |

## Platform notes

* PyTorch and LightGBM each bundle an OpenMP runtime that deadlocks or segfaults on macOS when
  both run in one process. The notebook therefore runs the foundation models in a spawned
  worker process.
* PyTensor is set to its Numba backend (`PYTENSOR_FLAGS=mode=NUMBA,cxx=`), so no C++ compiler
  is needed. Some macOS toolchains fail to link PyTensor's C extensions.
