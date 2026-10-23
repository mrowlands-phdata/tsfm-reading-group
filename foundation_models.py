"""Zero-shot time-series foundation-model forecasters.

These run in their own worker process (see the notebook): PyTorch and LightGBM each bundle an
OpenMP runtime, and on macOS the two deadlock or segfault when both are used in one process.

Every forecaster takes `train` (history up to the origin) and `future` (covariates for the next
`h` days) as long data frames keyed by `unique_id` and `ds`, and returns a frame with
`unique_id, ds, point, q0.1 … q0.9`.
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
for _name in ["transformers", "chronos", "huggingface_hub", "httpx"]:
    logging.getLogger(_name).setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(os.cpu_count())

DEVICE = "cpu"
_MODELS = {}


def load(name: str):
    """Load (once per process) and return a pretrained model."""
    if name not in _MODELS:
        if name == "chronos2":
            from chronos import Chronos2Pipeline

            _MODELS[name] = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)
        elif name == "bolt":
            from chronos import BaseChronosPipeline

            _MODELS[name] = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map=DEVICE)
        elif name == "timesfm":
            import timesfm

            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            model.compile(timesfm.ForecastConfig(
                max_context=1024, max_horizon=64, normalize_inputs=True, use_continuous_quantile_head=True,
                fix_quantile_crossing=True, infer_is_positive=True, per_core_batch_size=32,
            ))
            _MODELS[name] = model
    return _MODELS[name]


def warmup():
    """Process initialiser: load every model so download/initialisation stays out of the timings."""
    for name in ["chronos2", "bolt", "timesfm"]:
        load(name)


def _q_cols(quantiles):
    return [f"q{q:.1f}" for q in quantiles]


def _contexts(train):
    ids = sorted(train.unique_id.unique())
    return ids, [train.loc[train.unique_id == u].sort_values("ds").y.to_numpy(float) for u in ids]


def _frame(future, q, quantiles):
    """Attach a (series × horizon × quantile) array to the sorted future index."""
    out = future[["unique_id", "ds"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    out[_q_cols(quantiles)] = q.reshape(-1, len(quantiles))
    out["point"] = out["q0.5"]
    return out


def chronos_bolt(train, future, h, quantiles):
    _, ctx = _contexts(train)
    ctx = [torch.tensor(c[-2048:], dtype=torch.float32) for c in ctx]
    q, _ = load("bolt").predict_quantiles(ctx, prediction_length=h, quantile_levels=list(quantiles))
    return _frame(future, q.numpy(), quantiles)


def timesfm_25(train, future, h, quantiles):
    _, ctx = _contexts(train)
    _, q = load("timesfm").forecast(horizon=h, inputs=ctx)  # q[..., 0] is the mean, then deciles 0.1..0.9
    return _frame(future, q[:, :h, 1:10], quantiles)


def chronos2(train, future, h, quantiles, covariates=(), cross_learning=False):
    cols = ["unique_id", "ds", *covariates]
    pred = load("chronos2").predict_df(
        train[[*cols, "y"]].sort_values(["unique_id", "ds"]),
        future_df=future[cols].sort_values(["unique_id", "ds"]) if covariates else None,
        prediction_length=h, quantile_levels=list(quantiles), id_column="unique_id", timestamp_column="ds",
        target="y", cross_learning=cross_learning, batch_size=256,
    )
    rename = {str(q): c for q, c in zip(quantiles, _q_cols(quantiles))} | {"predictions": "point"}
    return pred.rename(columns=rename)[["unique_id", "ds", "point", *_q_cols(quantiles)]]


FORECASTERS = {"chronos_bolt": chronos_bolt, "timesfm": timesfm_25, "chronos2": chronos2}


def forecast(name: str, train: pd.DataFrame, future: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return FORECASTERS[name](train, future, **kwargs)
