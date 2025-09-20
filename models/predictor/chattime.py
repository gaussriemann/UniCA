from typing import List
import numpy as np
import pandas as pd
from gluonts.itertools import batcher
from gluonts.model import Forecast, QuantileForecast
from tqdm import tqdm

from models.predictor.ChatTime.model.model import ChatTime


class ChatTimePredictor:
    def __init__(
            self,
            model_path,
            prediction_length: int,
            context_length: int,
            ds_freq: str,
            *args,
            **kwargs,
    ):
        self.prediction_length = prediction_length
        self.ds_freq = ds_freq
        self.quantile_levels = kwargs.get('quantile_levels', (0.1, 0.5, 0.9))
        self.model = ChatTime(pred_len=prediction_length, hist_len=context_length, model_path=model_path)

    def predict(self, test_data_input, batch_size: int = 1024) -> List[Forecast]:
        forecasts: List[Forecast] = []
        # Iterate in mini-batches for memory efficiency, but run model per-series
        for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
            for ts in batch:
                # --- Prepare inputs ---
                hist = np.asarray(ts["target"], dtype=float)
                # Respect the model's expected history length if provided
                if self.model.hist_len is not None and len(hist) > self.model.hist_len:
                    hist = hist[-self.model.hist_len:]

                ctx_text = ts.get("context", None)
                if isinstance(ctx_text, (list, tuple, np.ndarray)) and len(ctx_text) == 1:
                    ctx_text = ctx_text[0]

                # --- Draw samples and convert to quantiles ---
                # Use the model's sampling interface; fallback to deterministic prediction if unavailable
                if hasattr(self.model, "predict_samples"):
                    samples = self.model.predict_samples(hist_data=hist, context=ctx_text)
                else:
                    # Fall back to repeating the point prediction
                    point_pred = self.model.predict(hist_data=hist, context=ctx_text)
                    samples = np.repeat(point_pred[None, :], repeats=getattr(self.model, 'num_samples', 8), axis=0)

                # Ensure shape (num_samples, prediction_length)
                samples = np.asarray(samples, dtype=float)
                if samples.ndim != 2:
                    raise ValueError(f"Expected samples with shape (num_samples, prediction_length), got {samples.shape}")
                if samples.shape[1] != self.prediction_length:
                    # Truncate or pad to match requested prediction length
                    if samples.shape[1] > self.prediction_length:
                        samples = samples[:, : self.prediction_length]
                    else:
                        pad = np.full((samples.shape[0], self.prediction_length - samples.shape[1]), np.nan)
                        samples = np.concatenate([samples, pad], axis=1)

                q_arr = np.quantile(samples, q=self.quantile_levels, axis=0)

                # --- Compute forecast start date robustly ---
                start_field = ts["start"]
                tgt_len = int(len(ts["target"]))
                if isinstance(start_field, pd.Period):
                    forecast_start_date = start_field + tgt_len
                else:
                    # Assume pandas.Timestamp; convert to Period with provided frequency
                    if not hasattr(self, 'ds_freq') or self.ds_freq is None:
                        raise ValueError("ds_freq must be provided to compute forecast start date from Timestamp start.")
                    start_period = pd.Period(start_field, freq=self.ds_freq)
                    forecast_start_date = start_period + tgt_len

                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=q_arr,
                        forecast_keys=[str(q) for q in self.quantile_levels],
                        start_date=forecast_start_date,
                    )
                )

        return forecasts
