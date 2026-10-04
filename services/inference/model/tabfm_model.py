"""TabFM: a locally-hosted tabular foundation model used as a zero-shot regressor."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from .base import (
    MAX_CONTEXT_ROWS,
    MAX_NUM_FEATURES,
    TARGET_COLUMN,
    BaseInferenceModel,
    ModelUnavailableError,
    register,
    returns_volatility,
    take_recent,
    validate_context,
)

logger = logging.getLogger("mft.inference.model.tabfm")


WEIGHTS_LICENCE = "tabfm-non-commercial-v1.0"


NON_COMMERCIAL = True


ACK_ENV_VAR = "MFT_ACK_NON_COMMERCIAL_TABFM"


CHECKPOINT_ENV_VAR = "TABFM_CHECKPOINT_DIR"


HF_REPO_ID = "google/tabfm-1.0.0-pytorch"


WEIGHTS_VERSION = "1.0.0"


WEIGHTS_FILE = "regression/model.safetensors"


WEIGHTS_FOOTPRINT_BYTES = 7 * 1024**3


WEIGHTS_HEADROOM_BYTES = 1 * 1024**3


DEFAULT_N_ESTIMATORS = 8


DEFAULT_RANDOM_STATE = 42

LICENCE_WARNING = (
    "TabFM weights are licensed under %s: NON-COMMERCIAL, no production "
    "trading use. This instance is research-only. Plan.md section 5 requires "
    "the model to sit behind the InferenceModel protocol so it can be "
    "replaced with a commercially licensed model. Set %s=1 to record that "
    "this restriction has been read."
)


def weights_available(checkpoint_dir: str | None = None) -> bool:
    """Whether the TabFM backend and its weights can be loaded."""
    if checkpoint_dir is None:
        checkpoint_dir = os.environ.get(CHECKPOINT_ENV_VAR)
    if checkpoint_dir:
        return os.path.isdir(checkpoint_dir)
    from importlib.util import find_spec

    for package in ("tabfm", "huggingface_hub"):
        try:
            if find_spec(package) is None:
                return False
        except (ImportError, ValueError):
            return False
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    try:
        hit = try_to_load_from_cache(
            repo_id=HF_REPO_ID,
            filename=WEIGHTS_FILE,
        )
    except Exception:  # noqa: BLE001
        return False
    return isinstance(hit, str)


def _free_memory_bytes() -> int | None:
    """Bytes currently available for allocation, or None if unknowable."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if not line.startswith("MemAvailable:"):
                    continue
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _weights_version() -> str:
    """Version of the weights this tabfm release ships."""
    return WEIGHTS_VERSION


def _make_backend(device: str, dtype: Any) -> Any:
    """Fetch the pretrained TabFM module for `device`."""
    checkpoint_dir = os.environ.get(CHECKPOINT_ENV_VAR) or None


    if checkpoint_dir is not None and not os.path.isdir(checkpoint_dir):
        raise ModelUnavailableError(
            f"{CHECKPOINT_ENV_VAR} points at {checkpoint_dir!r}, which is not a "
            "directory. Fix the path or unset it to load "
            f"{HF_REPO_ID} from the Hugging Face cache."
        )

    required = WEIGHTS_FOOTPRINT_BYTES + WEIGHTS_HEADROOM_BYTES
    available = _free_memory_bytes()
    if available is not None and available < required:
        raise ModelUnavailableError(
            f"loading TabFM needs about {required / 1024**3:.1f} GB of free "
            f"memory and this machine reports {available / 1024**3:.1f} GB "
            "available. The OOM killer would take the process rather than "
            "raise. Free memory, lower inference.n_estimators, or set "
            "inference.model to 'heuristic'."
        )

    try:
        import torch
    except ImportError as err:
        raise ModelUnavailableError(
            "torch is not installed; run services/inference/venv.sh, or switch "
            "inference.model to 'heuristic'. See Plan.md section 5 for the "
            "JAX backend as an alternative."
        ) from err

    try:
        from tabfm.src.pytorch import tabfm_v1_0_0
    except ImportError as err:
        raise ModelUnavailableError(
            "the tabfm package is installed but its PyTorch backend is not "
            "importable; this needs a torch build matching the interpreter"
        ) from err

    backend_repo = getattr(tabfm_v1_0_0, "HF_REPO_ID", HF_REPO_ID)
    if backend_repo != HF_REPO_ID:
        raise ModelUnavailableError(
            f"tabfm backend targets {backend_repo} but this module is written "
            f"against {HF_REPO_ID}; a backend bump needs a review of the "
            "prompt layout in _predict_raw before it is trusted"
        )

    try:
        model = tabfm_v1_0_0.load(
            model_type="regression",
            checkpoint_path=checkpoint_dir,
            device=device,
            dtype=dtype,
            use_cache=True,
        )
    except Exception as err:  # noqa: BLE001
        raise ModelUnavailableError(
            f"could not load TabFM v{WEIGHTS_VERSION} regression weights "
            f"({WEIGHTS_LICENCE}) for {checkpoint_dir or HF_REPO_ID}: {err}. "
            "First load needs network access to huggingface.co and ~6.6 GB of "
            "free space in the Hugging Face cache; afterwards it works offline. "
            "To skip it entirely, set inference.model to 'heuristic'."
        ) from err
    return model


@register("tabfm")
class TabFMModel(BaseInferenceModel):
    """Zero-shot next-bar return prediction with TabFM, on CPU."""

    def __init__(
        self,
        device: str = "cpu",
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        random_state: int = DEFAULT_RANDOM_STATE,
        max_context_rows: int = MAX_CONTEXT_ROWS,
        dtype: Any = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
        if max_context_rows < 1:
            raise ValueError(f"max_context_rows must be >= 1, got {max_context_rows}")
        self._device = device
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._max_context_rows = max_context_rows

        self._dtype = dtype
        self._backend: Any = None
        self._loaded = False

    @property
    def _name(self) -> str:
        return f"tabfm-{_weights_version()}"

    @property
    def weights_licence(self) -> str:
        """Licence of the checkpoint backing this instance."""
        return WEIGHTS_LICENCE

    @property
    def is_commercially_licensed(self) -> bool:
        """Always False."""
        return not NON_COMMERCIAL

    def is_licence_acknowledged(self) -> bool:
        """Whether `ACK_ENV_VAR` records that the restriction has been read."""
        return os.environ.get(ACK_ENV_VAR) == "1"

    async def _load(self) -> None:
        """Acquire the pretrained weights."""
        logger.warning(LICENCE_WARNING, WEIGHTS_LICENCE, ACK_ENV_VAR)
        self._backend = _make_backend(self._device, dtype=self._dtype)
        logger.info(
            "tabfm backend ready: weights=%s device=%s n_estimators=%d "
            "max_context_rows=%d prompt_budget_rows=%d",
            self._name,
            self._device,
            self._n_estimators,
            self._max_context_rows,
            MAX_CONTEXT_ROWS,
        )

    def _regressor(self) -> Any:
        """Build a regressor around the loaded backend."""
        from tabfm import TabFMRegressor

        return TabFMRegressor(
            self._backend,
            n_estimators=self._n_estimators,
            max_num_rows=self._max_context_rows,
            max_num_features=MAX_NUM_FEATURES,
            random_state=self._random_state,
            use_amp=False,
            verbose=False,
        )

    def _predict_raw(self, context: pd.DataFrame, horizon: int) -> float:
        """Run in-context regression and return the prediction in return units."""
        table = validate_context(context, horizon=horizon)
        features = table.drop(columns=[TARGET_COLUMN])
        total = len(table)
        prompt = take_recent(features.iloc[: total - horizon], self._max_context_rows)
        future_returns = np.convolve(
            table[TARGET_COLUMN].to_numpy(dtype=float)[1:], np.ones(horizon), mode="valid"
        )
        targets = pd.Series(future_returns[-len(prompt):])
        volatility = returns_volatility(targets.to_numpy(dtype=float))
        prompt = prompt.reset_index(drop=True)

        if len(prompt) < 2 or len(prompt) != len(targets):
            raise ModelUnavailableError(
                f"prompt/target misalignment after truncation: "
                f"{len(prompt)} rows vs {len(targets)} targets"
            )

        query = features.iloc[[total - 1]]

        regressor = self._regressor()
        regressor.fit(prompt, targets)
        prediction = np.asarray(regressor.predict(query), dtype=float).ravel()
        if prediction.size == 0 or not np.isfinite(prediction[0]):
            raise ModelUnavailableError(
                f"tabfm returned a non-finite prediction: {prediction!r}"
            )


        return float(prediction[0] / volatility)


__all__ = [
    "ACK_ENV_VAR",
    "CHECKPOINT_ENV_VAR",
    "DEFAULT_N_ESTIMATORS",
    "DEFAULT_RANDOM_STATE",
    "HF_REPO_ID",
    "LICENCE_WARNING",
    "NON_COMMERCIAL",
    "TabFMModel",
    "WEIGHTS_FOOTPRINT_BYTES",
    "WEIGHTS_LICENCE",
    "WEIGHTS_VERSION",
    "weights_available",
]
