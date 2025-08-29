# # #!/usr/bin/env bash
# # set -euo pipefail

# # echo "[startup-hook] Begin (pip-only)"

# # REPO_DIR="/workspace/gfm-rag"
# # REQUIREMENTS_TXT="${REQUIREMENTS_TXT:-${REPO_DIR}/requirements.txt}"

# # # Speed up pip and keep logs clean
# # export PIP_DISABLE_PIP_VERSION_CHECK=1
# # export PIP_NO_INPUT=1
# # export PYTHONNOUSERSITE=1

# # # Upgrade base tooling
# # python -m pip install --upgrade pip setuptools wheel

# # # Install dependencies from requirements.txt if present
# # if [ -f "${REQUIREMENTS_TXT}" ]; then
# #   echo "[startup-hook] Installing requirements from ${REQUIREMENTS_TXT}"
# #   python -m pip install -r "${REQUIREMENTS_TXT}"
# # else
# #   echo "[startup-hook] No requirements.txt found at ${REQUIREMENTS_TXT}"
# # fi

# # # Install the repo in editable mode if build metadata exists
# # if [ -f "${REPO_DIR}/pyproject.toml" ] || [ -f "${REPO_DIR}/setup.py" ]; then
# #   echo "[startup-hook] Installing repo in editable mode"
# #   python -m pip install -e "${REPO_DIR}"
# # else
# #   echo "[startup-hook] Skipping editable install (no pyproject.toml/setup.py)"
# # fi

# # echo "[startup-hook] Using python: $(command -v python)"
# # python -V
# # pip -V
# # echo "[startup-hook] Done"
# #   "${ENV_PREFIX}/bin/python" -m ensurepip --upgrade || true
# #   "${ENV_PREFIX}/bin/python" -m pip install --upgrade pip setuptools wheel

# #   # Optional: install repo requirements
# #   if [ -f "${REQUIREMENTS_TXT}" ]; then
# #     echo "[startup-hook] Installing requirements from ${REQUIREMENTS_TXT}"
# #     "${ENV_PREFIX}/bin/python" -m pip install -r "${REQUIREMENTS_TXT}"
# #   fi

# #   # Optional: install repo itself in editable mode if metadata exists
# #   if [ -f "${REPO_DIR}/pyproject.toml" ] || [ -f "${REPO_DIR}/setup.py" ]; then
# #     echo "[startup-hook] Installing repo in editable mode"
# #     "${ENV_PREFIX}/bin/python" -m pip install -e "${REPO_DIR}"
# #   fi

# #   # Expose the env to the entrypoint by adjusting PATH/LD_LIBRARY_PATH
# #   export PATH="${ENV_PREFIX}/bin:${PATH}"
# #   export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# # else
# #   echo "[startup-hook] No environment.yml found, falling back to pip"
# #   if [ -f "${REQUIREMENTS_TXT}" ]; then
# #     python -m pip install --upgrade pip setuptools wheel
# #     python -m pip install -r "${REQUIREMENTS_TXT}"
# #   fi
# #   if [ -f "${REPO_DIR}/pyproject.toml" ] || [ -f "${REPO_DIR}/setup.py" ]; then
# #     python -m pip install -e "${REPO_DIR}"
# #   fi
# # fi

# # echo "[startup-hook] Using python: $(command -v python)"
# # python -V
# # pip -V
# # echo "[startup-hook] Done"

# # startup-hook.sh
# #!/usr/bin/env bash
# set -euo pipefail

# echo "[startup-hook] Begin"

# REPO_DIR="/workspace/gfm-rag"
# REQUIREMENTS_TXT="${REQUIREMENTS_TXT:-${REPO_DIR}/requirements.txt}"

# export PIP_DISABLE_PIP_VERSION_CHECK=1
# export PIP_NO_INPUT=1
# export PYTHONNOUSERSITE=1

# python -m ensurepip --upgrade || true
# python -m pip install --upgrade pip setuptools wheel
# python -m pip install sentence_transformers

# if [ -f "${REQUIREMENTS_TXT}" ]; then
#   echo "[startup-hook] Installing requirements from ${REQUIREMENTS_TXT}"
#   python -m pip install -r "${REQUIREMENTS_TXT}"
# else
#   echo "[startup-hook] No requirements.txt found at ${REQUIREMENTS_TXT}"
# fi

# if [ -f "${REPO_DIR}/pyproject.toml" ] || [ -f "${REPO_DIR}/setup.py" ]; then
#   echo "[startup-hook] Installing repo in editable mode"
#   python -m pip install -e "${REPO_DIR}"
# else
#   echo "[startup-hook] Skipping editable install (no pyproject.toml/setup.py)"
# fi

# echo "[startup-hook] Using python: $(command -v python)"
# python -V
# pip -V
# echo "[startup-hook] Done"

# scripts/startup-hook.sh
#!/usr/bin/env bash
set -euo pipefail

echo "[startup-hook] begin (poetry)"

REPO_DIR="/nfs/scratch_2/sohir.maskey/repos/gfm-rag"
REQ="${REQUIREMENTS_TXT:-$REPO_DIR/requirements.txt}"

# Base env hygiene
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PYTHONNOUSERSITE=1
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

# Caches on scratch
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/nfs/scratch_2/sohir.maskey/.cache/pip}"
export POETRY_CACHE_DIR="${POETRY_CACHE_DIR:-/nfs/scratch_2/sohir.maskey/.cache/poetry}"
export POETRY_VERSION="${POETRY_VERSION:-2.1.4}"
export POETRY_VIRTUALENVS_IN_PROJECT=true

python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install --upgrade --no-cache-dir pip setuptools wheel
python -m pip install --no-cache-dir determined

if [ -f "$REPO_DIR/pyproject.toml" ]; then
  echo "[startup-hook] Using Poetry ${POETRY_VERSION}"
  python -m pip install --no-cache-dir "poetry==${POETRY_VERSION}"
  poetry --version

  # Ensure Poetry uses the current Python (3.12 in the NGC image)
  poetry config virtualenvs.in-project true
  poetry config cache-dir "${POETRY_CACHE_DIR}"
  (cd "$REPO_DIR" && poetry env use "$(command -v python)")

  # Install from lock, without installing the repo itself as a package
  if [ -f "$REPO_DIR/poetry.lock" ]; then
    (cd "$REPO_DIR" && poetry install --no-interaction --no-ansi --sync --no-root)
  else
    (cd "$REPO_DIR" && poetry install --no-interaction --no-ansi --no-root)
  fi

  # Ensure determined is available inside the venv
  # (cd "$REPO_DIR" && poetry run python -m pip install --no-cache-dir determined)
  cd "$REPO_DIR" && poetry run python -m pip install --no-cache-dir determined

  # Prefer Poetry venv for the remainder of the job
  VENV_BIN="$REPO_DIR/.venv/bin"
  if [ -d "$VENV_BIN" ]; then
    export PATH="$VENV_BIN:$PATH"
  fi

#   poetry run python -c "import sys; print(f'[startup-hook] python {sys.version.split()[0]} (poetry) OK')"
#   poetry run python -c "import torch, transformers; print(f'[startup-hook] torch {torch.__version__}, transformers {transformers.__version__}')"
else
  echo "[startup-hook] No pyproject.toml; falling back to pip"
  python -m pip install -q --no-cache-dir determined
  if [ -f "$REQ" ]; then
    python -m pip install --no-cache-dir -r "$REQ"
  fi
  if [ -f "$REPO_DIR/setup.py" ]; then
    python -m pip install --no-cache-dir -e "$REPO_DIR"
  fi
  python -c "import sys; print(f'[startup-hook] python {sys.version.split()[0]} (pip) OK')"
fi

pip -V || true
echo "[startup-hook] done"
