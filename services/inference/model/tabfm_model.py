"""TabFM: a locally-hosted tabular foundation model used as a zero-shot regressor.

    TabFM (Tabular Foundation Model), by Google Research.
    Source:  https://github.com/google-research/tabfm
    Source licence:  Apache-2.0

    ***  WEIGHTS LICENCE:  tabfm-non-commercial-v1.0  ***

    The pretrained checkpoint is released under `tabfm-non-commercial-v1.0`.
    It is non-commercial and explicitly excludes production trading use. This
    is a live trading platform, so loading these weights is a research
    activity only. See `Plan.md` §5 for the full decision.

    The restriction is not a comment. `WEIGHTS_LICENCE` and `NON_COMMERCIAL`
    below are read by this module's own tests, and `load()` logs a WARNING
    every time the weights are acquired, so a deployment cannot pick the
    model up silently. `TabFMModel` lives behind `InferenceModel` precisely so
    that a commercially licensed model can replace it without the service
    noticing.

How it works
------------
TabFM does no training. The historical rows *are* the prompt: `fit` runs
in-context regression over the table it is given, and `predict` queries it.
Concretely, for a context of N rows and a horizon of h bars:

    train rows  = rows[0 : N-h]        features of the earlier bars
    train target = rows[h : N][ret_1]  the return each of them went on to make
    query row   = rows[N-1]            the newest bar, whose future we want

CPU is the target. Attention cost grows roughly linearly with the prompt, so
`MAX_CONTEXT_ROWS` truncates the prompt to its most recent rows rather than
feeding the whole history. That is a deliberate choice, not a limit being hit:
a stale regime at the head of the table costs signal, not just time.
"""

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

#: Licence of the pretrained checkpoint. See the module docstring.
WEIGHTS_LICENCE = "tabfm-non-commercial-v1.0"

#: True. There is no configuration of this model that makes it commercially
#: usable, and no flag in the service that should be read as doing so.
NON_COMMERCIAL = True

#: Environment variable a deployment can set to record that a human has seen
#: and accepted the restriction. It changes nothing about the licence; it only
#: makes the acknowledgement auditable in the logs.
ACK_ENV_VAR = "MFT_ACK_NON_COMMERCIAL_TABFM"

#: Set to a local directory to load weights from disk instead of Hugging Face.
CHECKPOINT_ENV_VAR = "TABFM_CHECKPOINT_DIR"

#: The Hugging Face repository holding the pretrained checkpoint. Duplicated
#: from the backend's own constant on purpose: reading it must not import the
#: backend, because `tabfm.src.pytorch.tabfm_v1_0_0` imports torch at module
#: scope and `name` is read by every signal, by logs, and by the test suite.
HF_REPO_ID = "google/tabfm-1.0.0-pytorch"

#: Version of the checkpoint this module targets. The backend ships a single
#: checkpoint; `_load` re-checks it against the backend's `HF_REPO_ID` so a
#: silent backend bump is caught at load time rather than at signal time.
WEIGHTS_VERSION = "1.0.0"

#: Weight file the cache probe looks for. The regression checkpoint is the only
#: one this module ever loads.
WEIGHTS_FILE = "regression/model.safetensors"

#: Peak resident cost of materialising the checkpoint, used by the free-memory
#: preflight below. The float32 regression weights are ~6.6 GB; the preflight
#: asks for this much plus headroom so a short-memory machine gets a typed
#: error instead of the OOM killer taking the process.
WEIGHTS_FOOTPRINT_BYTES = 7 * 1024**3

#: Headroom kept free so the loader is not racing the page cache and the rest
#: of the service while it maps the checkpoint.
WEIGHTS_HEADROOM_BYTES = 1 * 1024**3

#: Ensemble members. TabFM defaults to 32; each member is a full pass over the
#: prompt, and on CPU that is the dominant cost. 8 is a reasonable
#: quality/latency point for a 100x18 table.
DEFAULT_N_ESTIMATORS = 8

#: Deterministic by default so a backtest is reproducible.
DEFAULT_RANDOM_STATE = 42

LICENCE_WARNING = (
    "TabFM weights are licensed under %s: NON-COMMERCIAL, no production "
    "trading use. This instance is research-only. Plan.md section 5 requires "
    "the model to sit behind the InferenceModel protocol so it can be "
    "replaced with a commercially licensed model. Set %s=1 to record that "
    "this restriction has been read."
)


def weights_available(checkpoint_dir: str | None = None) -> bool:
    """Whether the TabFM backend and its weights can be loaded.

    Presence only: the backend is importable and the checkpoint is in the
    local Hugging Face cache. It never touches the network and never reads
    the checkpoint, so it is safe to call on a memory-constrained machine. It
    can be wrong in the optimistic direction: weights present in the cache may
    still be corrupt. It is not a licence to load them — the free-memory
    preflight in `_make_backend` is the gate that matters.

    The backend check is a `find_spec` rather than an import on purpose, and it
    deliberately asks only about the top-level `tabfm` package. `find_spec` on
    a dotted name imports every parent package on the way, and
    `tabfm.src.pytorch` imports torch at module scope — so probing the
    submodule would cost a full torch import. `try_to_load_from_cache` is a
    stat call, so the cache check stays honest. Whether the PyTorch backend
    specifically imports is settled by `_make_backend`, which is the only
    thing that imports it.
    """
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
    except Exception:  # noqa: BLE001 - cache probing must never raise
        return False
    return isinstance(hit, str)


def _free_memory_bytes() -> int | None:
    """Bytes currently available for allocation, or None if unknowable.

    Reads `MemAvailable` from procfs, which accounts for reclaimable page cache
    rather than reporting free RAM. Non-Linux hosts get None, and callers must
    treat that as "cannot check", not as "no memory".
    """
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
    """Version of the weights this tabfm release ships.

    Returns the module constant rather than reading the backend's repo id,
    because reading it would mean importing torch. `_load` reconciles the two
    once the backend is being imported anyway.
    """
    return WEIGHTS_VERSION


def _make_backend(device: str, dtype: Any) -> Any:
    """Fetch the pretrained TabFM module for `device`.

    This is the only function in the component that reads the Hugging Face
    cache or downloads from it, and the only one that imports torch. Every
    refusal happens before the checkpoint is touched so that a misconfigured
    or under-resourced machine gets a `ModelUnavailableError` rather than a
    6.6 GB download or an OOM kill.

    Args:
        device: Torch device string, e.g. ``"cpu"``.
        dtype: Compute dtype. TabFM is designed for bfloat16, but torch's
            CPU bfloat16 matmuls are emulated and slower than float32, so CPU
            inference wants the float32 weights. Pass ``None`` to keep them,
            which is the default.

    Raises:
        ModelUnavailableError: If the checkpoint path, the free memory, torch,
            or the weights are unavailable.
    """
    checkpoint_dir = os.environ.get(CHECKPOINT_ENV_VAR) or None

    # An operator who points at a directory means it. Falling through to a
    # 6.6 GB download because the path was mistyped is the failure mode this
    # guards, and it is why the check precedes the torch import below.
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
    except Exception as err:  # noqa: BLE001 - backend raises many types
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
    """Zero-shot next-bar return prediction with TabFM, on CPU.

    Args:
        device: Torch device. Defaults to CPU; there is no GPU on the target
            machine and the table sizes do not justify one.
        n_estimators: Ensemble members per prediction. Latency scales with it.
        random_state: Seed for the ensemble, fixed so runs are reproducible.
        max_context_rows: Prompt truncation. The frozen schema is 18 columns
            and attention is roughly linear in rows, so 100 is the knee.
        dtype: Compute dtype for the backend. ``None`` keeps the float32
            checkpoint, which is faster on CPU than the bfloat16 default.
    """

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
        # None keeps the float32 checkpoint, which is the right call on CPU.
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
        """Always False. The weights are non-commercial; see the docstring."""
        return not NON_COMMERCIAL

    def is_licence_acknowledged(self) -> bool:
        """Whether `ACK_ENV_VAR` records that the restriction has been read."""
        return os.environ.get(ACK_ENV_VAR) == "1"

    async def _load(self) -> None:
        """Acquire the pretrained weights. Idempotent via the base class."""
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
        """Run in-context regression and return the prediction in return units.

        `fit` on TabFM is not training: it is the forward pass that reads the
        prompt table. It is called per request because the prompt is the
        rolling context, which is what makes the model zero-shot.
        """
        table = validate_context(context, horizon=horizon)
        volatility = returns_volatility(table[TARGET_COLUMN].to_numpy(dtype=float))

        features = table.drop(columns=[TARGET_COLUMN])
        total = len(table)
        prompt = take_recent(features.iloc[: total - horizon], self._max_context_rows)
        targets = take_recent(
            table[TARGET_COLUMN].iloc[horizon:], self._max_context_rows
        ).reset_index(drop=True)
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

        # Reported in units of the context's own return volatility so the
        # score is comparable across instruments. The base class applies the
        # final [-1, 1] mapping.
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
