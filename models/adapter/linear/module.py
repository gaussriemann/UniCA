import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import Output
from gluonts.torch.distributions.quantile_output import QuantileOutput

from models.wrapper.fm.base import FMWrapperBase

logger = logging.getLogger(__name__)


class LinearCovariateAdapter(nn.Module):
    r"""
    Zero-shot covariate adapter that mimics the regression-based strategy used
    in the official TimesFM covariate notebook.

    Steps:

        1. Fit a per-series linear regression between the contextual targets and
           the provided covariates (dynamic + static).
        2. Use the regression to obtain contextual predictions and compute the
           residuals.
        3. Feed the residual series into the frozen TSFM to forecast future
           residuals (``y1``).
        4. Apply the same linear regression to future covariates (``y2``).
        5. Combine both pieces (``y1 + y2``) to obtain the final forecasts.

    If some dynamic covariates are only available historically, the adapter
    first leverages the TSFM itself to forecast the missing future values so
    that all regressors become future-known.
    """

    @validated()
    def __init__(
            self,
            context_length: int,
            prediction_length: int,
            model_wrapper: Optional[FMWrapperBase] = None,
            d_feat_static_real: Optional[List[int]] = None,
            c_feat_static_cat: Optional[List[int]] = None,
            d_feat_dynamic_real: Optional[List[int]] = None,
            c_feat_dynamic_cat: Optional[List[int]] = None,
            d_past_feat_dynamic_real: Optional[List[int]] = None,
            dropout_rate: float = 0.0,
            distr_output: Optional[Output] = None,
            ridge: float = 1e-4,
            *args,
            **kwargs,
    ):
        super().__init__()
        if model_wrapper is None:
            raise ValueError("LinearCovariateAdapter requires a FM wrapper.")

        self.model_wrapper = model_wrapper
        self.model_wrapper.freeze_model()

        self.context_length = context_length
        self.prediction_length = prediction_length

        self.d_feat_static_real = d_feat_static_real or []
        self.c_feat_static_cat = c_feat_static_cat or []
        self.d_feat_dynamic_real = d_feat_dynamic_real or []
        self.c_feat_dynamic_cat = c_feat_dynamic_cat or []
        self.d_past_feat_dynamic_real = d_past_feat_dynamic_real or []

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.ridge = ridge

        if distr_output is None:
            distr_output = QuantileOutput(self.model_wrapper.quantiles)
        self.distr_output = distr_output

        # Register a dummy buffer so that distributed training backends do not
        # complain about parameter-less modules while still avoiding any trainable
        # weights that would trigger optimizer steps.
        self.register_buffer("_ddp_dummy", torch.zeros(1), persistent=False)

    # --------------------------------------------------------------------- #
    #  Lightning / GluonTS plumbing
    # --------------------------------------------------------------------- #
    def describe_inputs(self, batch_size=1) -> InputSpec:
        total_dyn_real = sum(self.d_feat_dynamic_real)
        total_past_dyn_real = sum(self.d_past_feat_dynamic_real)
        total_dyn_cat = len(self.c_feat_dynamic_cat)
        total_static_real = sum(self.d_feat_static_real)
        total_static_cat = len(self.c_feat_static_cat)

        spec: Dict[str, Input] = {
            "past_target": Input(
                shape=(batch_size, self.context_length),
                dtype=torch.float,
            ),
            "past_observed_values": Input(
                shape=(batch_size, self.context_length),
                dtype=torch.float,
            ),
        }

        if total_static_real > 0:
            spec["feat_static_real"] = Input(
                shape=(batch_size, total_static_real),
                dtype=torch.float,
                required=False,
            )
        if total_static_cat > 0:
            spec["feat_static_cat"] = Input(
                shape=(batch_size, total_static_cat),
                dtype=torch.long,
                required=False,
            )

        if total_dyn_real > 0:
            spec["feat_dynamic_real"] = Input(
                shape=(batch_size, self.context_length + self.prediction_length, total_dyn_real),
                dtype=torch.float,
                required=False,
            )
            spec["observed_feat_dynamic_real"] = Input(
                shape=(batch_size, self.context_length + self.prediction_length, total_dyn_real),
                dtype=torch.float,
                required=False,
            )
        if total_dyn_cat > 0:
            spec["feat_dynamic_cat"] = Input(
                shape=(batch_size, self.context_length + self.prediction_length, total_dyn_cat),
                dtype=torch.long,
                required=False,
            )
            spec["observed_feat_dynamic_cat"] = Input(
                shape=(batch_size, self.context_length + self.prediction_length, total_dyn_cat),
                dtype=torch.float,
                required=False,
            )

        if total_past_dyn_real > 0:
            spec["past_feat_dynamic_real"] = Input(
                shape=(batch_size, self.context_length, total_past_dyn_real),
                dtype=torch.float,
                required=False,
            )
            spec["past_observed_feat_dynamic_real"] = Input(
                shape=(batch_size, self.context_length, total_past_dyn_real),
                dtype=torch.float,
                required=False,
            )

        return InputSpec(spec, torch.zeros)

    def input_types(self) -> Dict[str, torch.dtype]:
        return {
            "past_target": torch.float,
            "past_observed_values": torch.float,
            "feat_static_real": torch.float,
            "feat_static_cat": torch.long,
            "feat_dynamic_real": torch.float,
            "feat_dynamic_cat": torch.long,
            "observed_feat_dynamic_real": torch.float,
            "observed_feat_dynamic_cat": torch.float,
            "past_feat_dynamic_real": torch.float,
            "past_observed_feat_dynamic_real": torch.float,
        }

    # --------------------------------------------------------------------- #
    #  Core logic
    # --------------------------------------------------------------------- #
    def forward(
            self,
            past_target: torch.Tensor,
            past_observed_values: torch.Tensor,
            feat_static_real: Optional[torch.Tensor] = None,
            feat_static_cat: Optional[torch.Tensor] = None,
            feat_dynamic_real: Optional[torch.Tensor] = None,
            feat_dynamic_cat: Optional[torch.Tensor] = None,
            observed_feat_dynamic_real: Optional[torch.Tensor] = None,
            observed_feat_dynamic_cat: Optional[torch.Tensor] = None,
            past_feat_dynamic_real: Optional[torch.Tensor] = None,
            past_observed_feat_dynamic_real: Optional[torch.Tensor] = None,
            is_train: bool = False,
            **kwargs,
    ):
        device = past_target.device

        dynamic_real = self._prepare_dynamic_real(
            feat_dynamic_real,
            observed_feat_dynamic_real,
            past_feat_dynamic_real,
            past_observed_feat_dynamic_real,
        )
        dynamic_cat = feat_dynamic_cat

        ctx_features = self._build_design_matrix(
            dynamic_real,
            dynamic_cat,
            feat_static_real,
            feat_static_cat,
            portion="context",
        )

        if ctx_features is None:
            raise ValueError("At least one covariate is required for linear regression.")

        betas = self._solve_regression(
            ctx_features,
            past_target,
            past_observed_values,
        )
        ctx_pred = self._apply_regression(ctx_features, betas)
        residual_context = past_target - ctx_pred

        residual_inputs = {
            "context": residual_context,
            "observed_values": past_observed_values,
        }
        tokenized = self.model_wrapper.tokenize(residual_inputs)
        encoded = self.model_wrapper.encode(tokenized)
        decode_output = self.model_wrapper.predict(encoded)

        future_features = self._build_design_matrix(
            dynamic_real,
            dynamic_cat,
            feat_static_real,
            feat_static_cat,
            portion="future",
        )
        if future_features is None:
            raise ValueError("Future covariates are required for the regression adapter.")
        linear_future = self._apply_regression(future_features, betas)

        scaling_stats = decode_output.get("loc_scale") or decode_output.get("stats")
        normalized_linear = self._normalize_component(linear_future, scaling_stats, device)
        decode_output["prediction"] = self._add_linear_component(
            decode_output["prediction"],
            normalized_linear,
        )

        decode_output["linear_component"] = linear_future

        if is_train:
            return decode_output
        return self.model_wrapper.process_forecast(decode_output)

    def loss(
            self,
            future_target: torch.Tensor,
            future_observed_values: torch.Tensor,
            **input_kwargs,
    ) -> torch.Tensor:
        output = self(is_train=True, **input_kwargs)
        inputs = {
            "future_target": future_target,
            "future_observed_values": future_observed_values,
            **output,
        }
        return self.model_wrapper.loss(inputs)

    # --------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------- #
    def _prepare_dynamic_real(
            self,
            feat_dynamic_real: Optional[torch.Tensor],
            observed_feat_dynamic_real: Optional[torch.Tensor],
            past_feat_dynamic_real: Optional[torch.Tensor],
            past_observed_feat_dynamic_real: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Ensure all dynamic real features contain both context and future values.
        Missing future steps are imputed via TSFM forecasts.
        """
        if feat_dynamic_real is None and past_feat_dynamic_real is None:
            return None

        pieces = []
        if past_feat_dynamic_real is not None:
            future_from_past = self._forecast_series(
                past_feat_dynamic_real,
                past_observed_feat_dynamic_real,
            )
            pieces.append(torch.cat([past_feat_dynamic_real, future_from_past], dim=1))

        if feat_dynamic_real is not None:
            context = feat_dynamic_real[:, :self.context_length, :]
            if observed_feat_dynamic_real is not None:
                future_obs = observed_feat_dynamic_real[:, self.context_length:, :]
            else:
                future_obs = None
            imputed_future = self._impute_future_dynamic(
                feat_dynamic_real[:, self.context_length:, :],
                future_obs,
                context,
                observed_feat_dynamic_real[:, :self.context_length, :] if observed_feat_dynamic_real is not None else None,
            )
            pieces.append(torch.cat([context, imputed_future], dim=1))

        return torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]

    def _impute_future_dynamic(
            self,
            provided_future: torch.Tensor,
            future_mask: Optional[torch.Tensor],
            context: torch.Tensor,
            context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if provided_future.size(1) >= self.prediction_length:
            future = provided_future[:, :self.prediction_length, :]
        else:
            future = torch.zeros(
                provided_future.size(0),
                self.prediction_length,
                provided_future.size(2),
                device=provided_future.device,
                dtype=provided_future.dtype,
            )
            future[:, :provided_future.size(1), :] = provided_future

        if future_mask is None:
            return future

        needs_impute = (future_mask[:, :self.prediction_length, :] < 0.5).any()
        if not needs_impute:
            return future

        forecast = self._forecast_series(context, context_mask)
        mask = future_mask[:, :self.prediction_length, :]
        mask_bool = mask >= 0.5
        return torch.where(mask_bool, future, forecast)

    def _build_design_matrix(
            self,
            dynamic_real: Optional[torch.Tensor],
            dynamic_cat: Optional[torch.Tensor],
            static_real: Optional[torch.Tensor],
            static_cat: Optional[torch.Tensor],
            portion: str,
    ) -> Optional[torch.Tensor]:
        if portion not in {"context", "future"}:
            raise ValueError("portion must be either 'context' or 'future'.")

        length = self.context_length if portion == "context" else self.prediction_length

        features: List[torch.Tensor] = []
        if dynamic_real is not None:
            start = 0 if portion == "context" else self.context_length
            dyn_slice = dynamic_real[:, start:start + length, :]
            features.append(dyn_slice)

        if dynamic_cat is not None and len(self.c_feat_dynamic_cat) > 0:
            start = 0 if portion == "context" else self.context_length
            cat_slice = dynamic_cat[:, start:start + length, :]
            one_hot = []
            for idx, cardinality in enumerate(self.c_feat_dynamic_cat):
                vals = cat_slice[..., idx].long().clamp(min=0)
                one_hot.append(F.one_hot(vals, num_classes=cardinality).float())
            features.append(torch.cat(one_hot, dim=-1))

        if static_real is not None and static_real.shape[-1] > 0:
            repeated = static_real.unsqueeze(1).repeat(1, length, 1)
            features.append(repeated)

        if static_cat is not None and len(self.c_feat_static_cat) > 0:
            static_hot = []
            for idx, cardinality in enumerate(self.c_feat_static_cat):
                vals = static_cat[:, idx].long().clamp(min=0)
                hot = F.one_hot(vals, num_classes=cardinality).float()
                static_hot.append(hot.unsqueeze(1).repeat(1, length, 1))
            features.append(torch.cat(static_hot, dim=-1))

        if not features:
            return None

        design = torch.cat(features, dim=-1)
        if self.dropout is not None:
            design = self.dropout(design)

        intercept = torch.ones(
            design.size(0),
            length,
            1,
            device=design.device,
            dtype=design.dtype,
        )
        return torch.cat([design, intercept], dim=-1)

    def _solve_regression(
            self,
            design_matrix: torch.Tensor,
            past_target: torch.Tensor,
            past_observed_values: torch.Tensor,
    ) -> torch.Tensor:
        mask = past_observed_values.bool()
        mask = mask.unsqueeze(-1)
        X = design_matrix * mask
        y = past_target * past_observed_values

        Xt = X.transpose(1, 2)
        XtX = Xt @ X
        eye = torch.eye(
            XtX.size(-1),
            device=X.device,
            dtype=X.dtype,
        ).unsqueeze(0).expand_as(XtX)
        XtX = XtX + self.ridge * eye
        Xty = Xt @ y.unsqueeze(-1)

        try:
            beta = torch.linalg.solve(XtX, Xty).squeeze(-1)
        except RuntimeError:
            beta = torch.linalg.pinv(XtX) @ Xty
            beta = beta.squeeze(-1)
        return beta

    def _apply_regression(
            self,
            design_matrix: torch.Tensor,
            beta: torch.Tensor,
    ) -> torch.Tensor:
        return torch.bmm(design_matrix, beta.unsqueeze(-1)).squeeze(-1)

    def _forecast_series(
            self,
            context: torch.Tensor,
            observed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, ctx_len, dims = context.shape
        series = context.reshape(bsz * dims, ctx_len)
        if observed is not None:
            obs = observed.reshape(bsz * dims, ctx_len)
        else:
            obs = torch.ones_like(series)

        inputs = {"context": series, "observed_values": obs}
        with torch.no_grad():
            tokenized = self.model_wrapper.tokenize(inputs)
            encoded = self.model_wrapper.encode(tokenized)
            decoded = self.model_wrapper.predict(encoded)
            point_forecast = self._extract_point_forecast(decoded["prediction"])
        point_forecast = point_forecast.reshape(bsz, dims, self.prediction_length)
        return point_forecast.permute(0, 2, 1)

    def _extract_point_forecast(self, prediction: torch.Tensor) -> torch.Tensor:
        if prediction.dim() != 3:
            raise ValueError(f"Unsupported prediction shape {prediction.shape}")

        if prediction.shape[1] == len(self.model_wrapper.quantiles):
            quantiles = torch.tensor(
                self.model_wrapper.quantiles,
                device=prediction.device,
                dtype=prediction.dtype,
            )
            idx = torch.argmin(torch.abs(quantiles - 0.5))
            return prediction[:, idx, :]
        if prediction.shape[2] == len(self.model_wrapper.quantiles) + 1:
            # TimesFM style: [B, H, mean + quantiles]
            return prediction[:, :, 0]
        if prediction.shape[2] == len(self.model_wrapper.quantiles):
            quantiles = torch.tensor(
                self.model_wrapper.quantiles,
                device=prediction.device,
                dtype=prediction.dtype,
            )
            idx = torch.argmin(torch.abs(quantiles - 0.5))
            return prediction[:, :, idx]
        raise ValueError(f"Unable to infer point forecast from shape {prediction.shape}")

    def _normalize_component(
            self,
            component: torch.Tensor,
            scaling_stats,
            device: torch.device,
    ) -> torch.Tensor:
        if scaling_stats is None:
            return component

        if isinstance(scaling_stats, (tuple, list)):
            loc, scale = scaling_stats
        else:
            raise ValueError("Unexpected scaling statistics format.")

        if isinstance(loc, torch.Tensor):
            loc = loc.to(device=device, dtype=component.dtype)
        else:
            loc = torch.tensor(loc, device=device, dtype=component.dtype)

        if isinstance(scale, torch.Tensor):
            scale = scale.to(device=device, dtype=component.dtype)
        else:
            scale = torch.tensor(scale, device=device, dtype=component.dtype)

        while loc.dim() < component.dim():
            loc = loc.unsqueeze(-1)
        while scale.dim() < component.dim():
            scale = scale.unsqueeze(-1)

        scale = torch.clamp(scale, min=1e-6)
        return (component - loc) / scale

    def _add_linear_component(
            self,
            prediction: torch.Tensor,
            component: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.dim() != 3:
            raise ValueError(f"Unsupported prediction shape {prediction.shape}")

        if prediction.shape[-1] == self.prediction_length:
            # Shape: [B, num_outputs, H]
            return prediction + component.unsqueeze(1)
        if prediction.shape[1] == self.prediction_length:
            # Shape: [B, H, num_outputs]
            return prediction + component.unsqueeze(-1)
        raise ValueError(f"Unexpected prediction layout: {prediction.shape}")
