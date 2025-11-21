import logging
from typing import Optional, Sequence, Tuple

import torch
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.torch.distributions import QuantileOutput

from models.wrapper.fm.base import FMWrapperBase

logger = logging.getLogger(__name__)

try:
    from momentfm import MOMENTPipeline
    from momentfm.utils.masking import Masking
except ImportError as err:
    raise ImportError(
        "momentfm is required for the MOMENT adapter. Please install it via "
        "`pip install momentfm`."
    ) from err


class MomentWrapper(FMWrapperBase):
    """
    Wrapper that exposes MOMENT-1 models through the ``FMWrapperBase`` interface so
    UniCA can treat MOMENT like any other TSFM backbone.
    """

    def __init__(
            self,
            model_id_or_path: str,
            prediction_length: int,
            device: Optional[str] = None,
            model_kwargs: Optional[dict] = None,
            moment_task: str = "forecasting",
            anomaly_criterion: str = "mse",
    ):
        super().__init__()
        self.prediction_length = prediction_length
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        valid_tasks = {"forecasting", "anomaly_detection", "imputation"
        if moment_task not in valid_tasks:
            raise ValueError(f"moment_task must be one of {valid_tasks}")
        self.moment_task = moment_task
        self.anomaly_criterion = anomaly_criterion

        model_kwargs = model_kwargs or {}
        if self.moment_task == "anomaly_detection":
            model_kwargs.setdefault("task_name", "reconstruction")
        elif self.moment_task == "imputation":
            model_kwargs.setdefault("task_name", "forecasting")
            model_kwargs.setdefault("forecast_horizon", prediction_length)

        logger.info("Loading MOMENT pipeline from %s", model_id_or_path)
        self.model = MOMENTPipeline.from_pretrained(
            model_id_or_path,
            model_kwargs=model_kwargs,
        )
        self.model.init()
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self._quantiles = [0.5]
        self.quantile_output = QuantileOutput(self._quantiles)

    # ------------------------------------------------------------------ #
    # Required properties
    # ------------------------------------------------------------------ #
    @property
    def context_length(self) -> int:
        return int(self.model.config.seq_len)

    @property
    def model_dim(self) -> int:
        return int(self.model.config.d_model)

    @property
    def quantiles(self) -> Sequence[float]:
        return self._quantiles

    # ------------------------------------------------------------------ #
    # Tokenize / Encode / Predict
    # ------------------------------------------------------------------ #
    def tokenize(self, inputs):
        if self.moment_task not in {"forecasting", "imputation"}:
            raise RuntimeError(
                "Tokenization is only available in forecasting or imputation mode. "
                "Use `detect_anomalies` for anomaly detection."
            )

        context = inputs["context"]
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if context.dim() == 2:
            context = context.unsqueeze(1)
        elif context.dim() != 3:
            raise ValueError(f"Unsupported MOMENT input shape: {context.shape}")

        observed = inputs.get("observed_values")
        if observed is None:
            observed = torch.ones(context.size(0), context.size(-1), device=context.device)
        if observed.dim() == 1:
            observed = observed.unsqueeze(0)
        observed = observed.to(context.device)

        context = context.to(self.device).float()
        observed = observed.to(self.device).float()

        x_enc = self.model.normalizer(x=context, mask=observed, mode="norm")
        mean = self.model.normalizer.mean.detach().clone()
        stdev = self.model.normalizer.stdev.detach().clone()
        x_enc = torch.nan_to_num(x_enc, nan=0.0, posinf=0.0, neginf=0.0)

        x_enc = self.model.tokenizer(x=x_enc)
        mask_for_embedding = observed
        enc_in = self.model.patch_embedding(x_enc, mask=mask_for_embedding)

        batch_size, n_channels = context.shape[0], context.shape[1]
        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.model.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(observed, self.model.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)

        return {
            "token_embeddings": enc_in,
            "attention_mask": attention_mask,
            "input_mask": observed,
            "norm_stats": (mean, stdev),
        }

    def encode(self, inputs):
        if self.moment_task not in {"forecasting", "imputation"}:
            raise RuntimeError("Encoding is only available in forecasting or imputation mode.")

        token_embeddings = inputs["token_embeddings"]
        attention_mask = inputs["attention_mask"]

        encoder_outputs = self.model.encoder(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_outputs.last_hidden_state

        batch_size = inputs["input_mask"].shape[0]
        seq_batches = token_embeddings.shape[0]
        n_channels = seq_batches // batch_size
        n_patches = hidden_states.shape[1]
        hidden_states = hidden_states.reshape(
            batch_size, n_channels, n_patches, self.model.config.d_model
        )

        return {
            "encoded_embeddings": hidden_states,
            "input_mask": inputs["input_mask"],
            "norm_stats": inputs["norm_stats"],
            "num_channels": n_channels,
        }

    def predict(self, inputs):
        if self.moment_task != "forecasting":
            raise RuntimeError("Predict is only available in forecasting mode. "
                               "Use `detect_anomalies` for anomaly detection.")

        hidden_states = inputs["encoded_embeddings"]
        num_channels = inputs["num_channels"]

        dec_out = self.model.head(hidden_states)

        mean, stdev = inputs["norm_stats"]
        forecast = self._denormalize(dec_out, mean, stdev)
        # MOMENT forecasting head returns [batch, channels, horizon]
        if num_channels > 1:
            forecast = forecast.mean(dim=1, keepdim=True)
        else:
            forecast = forecast[:, :1, :]
        forecast = forecast[:, :, : self.prediction_length]

        prediction = forecast.squeeze(1).unsqueeze(-1)  # [B, H, Q]

        return {"prediction": prediction}

    def reconstruct_from_encoded(self, inputs):
        if self.moment_task != "imputation":
            raise RuntimeError("Reconstruction can only be used when `moment_task='imputation'`.")

        hidden_states = inputs["encoded_embeddings"]
        dec_out = self.model.head(hidden_states)
        mean, stdev = inputs["norm_stats"]
        reconstruction = self._denormalize(dec_out, mean, stdev)

        if reconstruction.dim() == 3 and reconstruction.size(1) == 1:
            reconstruction = reconstruction.squeeze(1)

        return {"reconstruction": reconstruction}

    def impute(
            self,
            context: torch.Tensor,
            observed_values: Optional[torch.Tensor] = None,
            input_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.moment_task != "imputation":
            raise RuntimeError("Instantiate MomentWrapper with `moment_task='imputation'` to run imputation.")
        context, mask = self._prepare_series(context, observed_values)
        if input_mask is None:
            input_mask = torch.ones_like(mask)
        outputs = self.model.reconstruct(
            x_enc=context,
            input_mask=input_mask,
            mask=mask,
        )
        reconstruction = outputs.reconstruction
        if reconstruction.dim() == 3 and reconstruction.size(1) == 1:
            reconstruction = reconstruction.squeeze(1)
        return reconstruction

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #
    def _prepare_series(
            self,
            context: torch.Tensor,
            observed: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if context.dim() == 2:
            context = context.unsqueeze(1)
        if context.dim() != 3:
            raise ValueError(f"Expected 3D tensor for MOMENT inputs, got {context.shape}")
        if observed is None:
            observed = torch.ones(context.shape[0], context.shape[-1], device=context.device)
        if observed.dim() == 1:
            observed = observed.unsqueeze(0)
        return context.to(self.device).float(), observed.to(self.device).float()

    def detect_anomalies(
            self,
            context: torch.Tensor,
            observed_values: Optional[torch.Tensor] = None,
            anomaly_criterion: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs MOMENT in reconstruction mode and returns anomaly scores and reconstructions.
        """
        if self.moment_task != "anomaly_detection":
            raise RuntimeError("Instantiate MomentWrapper with `moment_task='anomaly_detection'` "
                               "to use anomaly detection utilities.")
        x_enc, mask = self._prepare_series(context, observed_values)
        criterion = anomaly_criterion or self.anomaly_criterion
        outputs = self.model.detect_anomalies(
            x_enc=x_enc,
            input_mask=mask,
            anomaly_criterion=criterion,
        )
        return outputs.anomaly_scores.detach().cpu(), outputs.reconstruction.detach().cpu()

    @staticmethod
    def _denormalize(values: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor) -> torch.Tensor:
        while mean.dim() < values.dim():
            mean = mean.unsqueeze(-1)
        while stdev.dim() < values.dim():
            stdev = stdev.unsqueeze(-1)
        return values * stdev + mean

    def loss(self, inputs):
        predictions = inputs["prediction"]
        target = inputs["future_target"]
        mask = inputs.get("future_observed_values")

        loss = self.quantile_output.loss(target, (predictions,))
        if mask is not None:
            mask = mask.unsqueeze(-1).to(loss.device)
            loss = loss * mask
            denom = torch.clamp(mask.sum(), min=1.0)
            return loss.sum() / denom
        return loss.mean()

    def forecast_generator(self):
        return QuantileForecastGenerator(self.quantiles)

    def process_forecast(self, outputs):
        prediction = outputs["prediction"]  # [B, H, Q]
        return (prediction, ), None, None
