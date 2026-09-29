#!/usr/bin/env bash
#
# Bootstrap the inference virtualenv.
#
# This machine has Python 3.14.7, no pip on the system interpreter and no uv.
# `ensurepip` is present, so `python3 -m venv` is enough to get going.
#
#   ./venv.sh                 create .venv and install requirements.txt
#   ./venv.sh --dev           also install requirements-dev.txt (pytest)
#   ./venv.sh --recreate      throw the venv away and start over
#   ./venv.sh --skip-model    install everything except the torch/tabfm
#                             backend, for machines where no wheel exists
#                             and the service will run the heuristic model
#
# requirements.txt is FROZEN and is never edited here. weights are downloaded
# by the model on first load and cached outside the repository. See Plan.md
# section 5 for the licence.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
REQUIREMENTS="$HERE/requirements.txt"
DEV_REQUIREMENTS="$HERE/requirements-dev.txt"
PYTHON="${PYTHON:-python3}"

# PyTorch's PyPI wheel for linux is the CUDA build: it drags in ~2.5 GB of
# nvidia-* wheels for hardware this machine does not have, and the venv lands
# around 6 GB. The CPU index publishes the same versions for cp314 and the
# venv lands under 2 GB. requirements.txt says "torch is the CPU build", so
# this is what it means. Set MFT_TORCH_CPU_ONLY=0 to take the stock wheel.
TORCH_CPU_ONLY="${MFT_TORCH_CPU_ONLY:-1}"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

MIN_PY_MAJOR=3
MIN_PY_MINOR=11

log()  { printf '%s\n' "$*" >&2; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

RECREATE=0
SKIP_MODEL=0
WITH_DEV=0
for arg in "$@"; do
  case "$arg" in
    --recreate)   RECREATE=1 ;;
    --skip-model) SKIP_MODEL=1 ;;
    --dev)        WITH_DEV=1 ;;
    -h|--help)    awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' \
                    "${BASH_SOURCE[0]}"; exit 0 ;;
    *)            fail "unknown argument: $arg (try --help)" ;;
  esac
done

[ -f "$REQUIREMENTS" ] || fail "requirements not found at $REQUIREMENTS"

command -v "$PYTHON" >/dev/null 2>&1 \
  || fail "'$PYTHON' not found. Install Python >= ${MIN_PY_MAJOR}.${MIN_PY_MINOR} or set PYTHON=/path/to/python3."

"$PYTHON" - <<'PY' || exit 1
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 11):
    sys.exit(
        f"error: Python {major}.{minor} is too old; tabfm needs >= 3.11 "
        f"(this machine has {sys.version.split()[0]})."
    )
PY

if [ "$RECREATE" = 1 ] && [ -d "$VENV" ]; then
  log "==> removing $VENV (--recreate)"
  rm -rf "$VENV"
fi

# --- venv ------------------------------------------------------------------

if [ ! -x "$VENV/bin/python" ]; then
  log "==> creating $VENV with $PYTHON -m venv"
  # Some distro builds ship python3-venv without ensurepip bundled, so fall
  # back to bootstrapping pip from the system interpreter's ensurepip.
  if ! "$PYTHON" -m venv "$VENV" 2>/tmp/venv-bootstrap.$$.log; then
    log "    'python3 -m venv' failed, retrying with --without-pip + ensurepip"
    "$PYTHON" -m venv --without-pip "$VENV" \
      || fail "could not create the venv. See /tmp/venv-bootstrap.$$.log and install the venv module (Debian/Ubuntu: apt install python3-venv)."
    "$VENV/bin/python" -m ensurepip --upgrade \
      || fail "ensurepip failed inside the venv. Install pip: apt install python3-pip, or bootstrap manually with get-pip.py."
  fi
fi

PIP="$VENV/bin/python -m pip"
VENV_PY="$VENV/bin/python"

if ! $PIP --version >/dev/null 2>&1; then
  log "==> bootstrapping pip via ensurepip"
  "$VENV_PY" -m ensurepip --upgrade \
    || fail "could not bootstrap pip inside $VENV"
fi

log "==> upgrading pip"
$PIP install --upgrade --quiet pip \
  || log "    pip upgrade failed; continuing with the bundled version"

# --- torch -----------------------------------------------------------------

if [ "$TORCH_CPU_ONLY" = 1 ] && [ "$SKIP_MODEL" = 0 ]; then
  log "==> installing torch from the CPU index ($TORCH_CPU_INDEX)"
  if ! $PIP install --index-url "$TORCH_CPU_INDEX" torch; then
    fail "could not install the CPU build of torch for $("$VENV_PY" -c 'import sys; print("python%d.%d" % sys.version_info[:2])').

Torch CPU wheels for this interpreter are missing. Two options:

  1. Use the JAX backend instead (Plan.md section 5). jaxlib publishes
     cp314 wheels, so this works on 3.14:
         .venv/bin/python -m pip uninstall -y torch
         .venv/bin/python -m pip install 'tabfm[jax]'
     tabfm_model.py already imports the PyTorch backend only; a JAX backend
     is a one-line change and is listed in Plan.md as the supported
     alternative.

  2. Re-run with the stock PyPI wheel, which bundles CUDA:
         MFT_TORCH_CPU_ONLY=0 ./venv.sh
     That is ~2.5 GB of nvidia-* wheels plus ~0.9 GB of triton on a machine
     with no GPU.

Or skip the model backend entirely and run the service on the heuristic
model, which needs neither torch nor weights:
    inference.model: heuristic   # in configs/config.yaml"
  fi
fi

# --- requirements ----------------------------------------------------------

# The frozen file marks the model backend with a "# --- model ---" section
# header. --skip-model strips everything from there down so the rest of the
# service still installs. requirements.txt itself is never modified.
INSTALL_FROM="$REQUIREMENTS"
STRIPPED=""
if [ "$SKIP_MODEL" = 1 ]; then
  STRIPPED="$(mktemp -t mft-requirements-nomodel.XXXXXX.txt)"
  sed '/^# --- model ---/,$d' "$REQUIREMENTS" > "$STRIPPED"
  INSTALL_FROM="$STRIPPED"
  log "==> --skip-model: installing without torch, scikit-learn, huggingface-hub, tabfm"
  log "    set inference.model: heuristic in configs/config.yaml"
fi

log "==> installing $INSTALL_FROM"
REQ_LOG="$(mktemp -t mft-requirements.XXXXXX.log)"
# The CPU index has to stay reachable during this install. requirements.txt
# says `torch>=2.12.1` and is frozen, so it resolves against PyPI, where the
# same version number is the CUDA build. Leaving the CPU index off the
# resolution order here would silently undo the install above the moment any
# transitive dependency forces a torch reinstall.
TORCH_INDEX_ARGS=()
if [ "$TORCH_CPU_ONLY" = 1 ]; then
  TORCH_INDEX_ARGS=(--extra-index-url "$TORCH_CPU_INDEX")
fi
if ! $PIP install --progress-bar off "${TORCH_INDEX_ARGS[@]+"${TORCH_INDEX_ARGS[@]}"}" -r "$INSTALL_FROM" 2>&1 | tee "$REQ_LOG"; then
  MISSING="$(grep -oE 'No matching distribution found for [A-Za-z0-9._-]+' "$REQ_LOG" \
    | sed 's/No matching distribution found for //' | sort -u | tr '\n' ' ')"
  {
    printf '\n'
    printf 'error: could not install %s\n' "$REQUIREMENTS"
    if [ -n "$MISSING" ]; then
      printf 'no wheel or sdist for: %s\n' "$MISSING"
    fi
    printf 'interpreter: %s\n' "$("$VENV_PY" -c 'import sys; print(sys.version)')"
    printf '\n'
    printf 'This machine is Python 3.14, which is new enough that some\n'
    printf 'wheels are not published for it yet. Options, cheapest first:\n'
    printf '\n'
    printf '  1. Drop the model backend and use the heuristic model. It needs\n'
    printf '     no torch and no weights:\n'
    printf '         inference.model: heuristic   # configs/config.yaml\n'
    printf '     Then re-run with: ./venv.sh --skip-model\n'
    printf '\n'
    printf '  2. Use the JAX backend instead of torch (Plan.md section 5).\n'
    printf '     jaxlib has cp314 wheels:\n'
    printf '         %s -m pip uninstall -y torch\n' "$VENV_PY"
    printf "         %s -m pip install 'tabfm[jax]'\n" "$VENV_PY"
    printf '\n'
    printf '  3. Build the venv on an older interpreter. TabFM needs >= 3.11:\n'
    printf '         PYTHON=python3.12 ./venv.sh\n'
    printf '         PYTHON=python3.11 ./venv.sh\n'
    printf '\n'
    printf 'Full pip output: %s\n' "$REQ_LOG"
  } >&2
  exit 1
fi

# --- torch build enforcement ------------------------------------------------

# requirements.txt says "torch is the CPU build" but pins only `torch>=2.12.1`,
# and that version exists on PyPI as the CUDA wheel. The resolver can pick it
# during the install above, so the promise is checked afterwards rather than
# assumed. Read from the wheel filename via importlib.metadata: importing torch
# to ask it is both slow and the thing being verified.
assert_cpu_torch() {
  [ "$TORCH_CPU_ONLY" = 1 ] || return 0
  [ "$SKIP_MODEL" = 0 ] || return 0

  local verdict
  verdict="$("$VENV_PY" - <<'PY'
from importlib.metadata import PackageNotFoundError, distribution

try:
    dist = distribution("torch")
except PackageNotFoundError:
    print("absent")
    raise SystemExit(0)

version = dist.version
local = version.partition("+")[2]
cuda_wheels = [n for n in (dist.requires or []) if n.lower().startswith("nvidia-")]
print(f"{version}|{local}|{len(cuda_wheels)}")
PY
)"

  local version local cuda_deps
  IFS='|' read -r version local cuda_deps <<<"$verdict"

  if [ "$version" = "absent" ]; then
    log "==> torch is not installed; the heuristic model will be used"
    return 0
  fi

  if [ -n "$local" ] || [ "$cuda_deps" = "0" ]; then
    log "==> torch $version is the CPU build"
    return 0
  fi

  log "==> WARNING: torch $version is the CUDA build"
  log "    Reinstalling the CPU build; the CUDA wheel is ~2.5 GB of"
  log "    nvidia-* packages plus ~0.9 GB of triton for a machine with no GPU."
  $PIP uninstall -y torch >/dev/null 2>&1 || true
  if $PIP install --index-url "$TORCH_CPU_INDEX" torch; then
    log "==> torch $("$VENV_PY" -c 'import importlib.metadata as m; print(m.version("torch"))') is the CPU build"
    return 0
  fi

  fail "torch $version is the CUDA build and the CPU build could not be installed.
Remove the CUDA wheels by hand and re-run with MFT_TORCH_CPU_ONLY=0 only if
you really want them:
    $VENV/bin/python -m pip uninstall -y torch
    MFT_TORCH_CPU_ONLY=0 ./venv.sh"
}

assert_cpu_torch

# --- verify ----------------------------------------------------------------

if [ "$WITH_DEV" = 1 ] && [ -f "$DEV_REQUIREMENTS" ]; then
  log "==> installing $DEV_REQUIREMENTS"
  $PIP install --progress-bar off "${TORCH_INDEX_ARGS[@]+"${TORCH_INDEX_ARGS[@]}"}" -r "$DEV_REQUIREMENTS" \
    || fail "could not install the dev requirements from $DEV_REQUIREMENTS"
  assert_cpu_torch
fi

log "==> verifying imports"
MFT_SKIP_MODEL="$SKIP_MODEL" "$VENV_PY" - <<'PY'
import importlib
import os
import sys

skip_model = os.environ.get("MFT_SKIP_MODEL") == "1"

required = ["numpy", "pandas", "fastapi", "uvicorn", "httpx"]
optional = ["sklearn", "tabfm"]

failures = []
for module in required:
    try:
        importlib.import_module(module)
    except ImportError as err:
        failures.append(f"{module}: {err}")

if not skip_model:
    for module in optional:
        try:
            importlib.import_module(module)
        except ImportError as err:
            failures.append(f"{module}: {err}")

if failures:
    for line in failures:
        print(f"error: cannot import {line}", file=sys.stderr)
    sys.exit(1)

import numpy
import pandas

print(f"  python {sys.version.split()[0]}")
print(f"  numpy  {numpy.__version__}")
print(f"  pandas {pandas.__version__}")
for module in optional:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        print(f"  {module:6s} not installed - the heuristic model will be used")
        continue
    print(f"  {module:6s} {getattr(mod, '__version__', '?')}")
PY

# The installed torch wheel, not the import: importing torch here is the most
# expensive thing this script does, and `torch.cuda.is_available()` adds a CUDA
# runtime probe on top of it for a machine that has no GPU. The wheel
# filename already said which build it is, and assert_cpu_torch read that.
if [ "$SKIP_MODEL" = 0 ]; then
  TORCH_WHEEL="$("$VENV_PY" -c 'import importlib.metadata as m
try:
    print(m.distribution("torch").version)
except m.PackageNotFoundError:
    print("not installed")')"
  log "==> torch   $TORCH_WHEEL"
fi

cat <<EOF

venv ready: $VENV

Next:
  make tabfm-weights    # downloads the TabFM weights on first use
  make run-inference

Reminder: the TabFM weights are licensed under tabfm-non-commercial-v1.0.
Research only, no production trading. See Plan.md section 5.
EOF
