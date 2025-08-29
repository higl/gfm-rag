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

echo "[startup-hook] begin"

REPO_DIR="/nfs/scratch_2/sohir.maskey/repos/gfm-rag"
REQ="${REQUIREMENTS_TXT:-$REPO_DIR/requirements.txt}"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PYTHONNOUSERSITE=1
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install --upgrade --no-cache-dir pip setuptools wheel
python -m pip install -q determined

if [ -f "$REQ" ]; then
  python -m pip install --no-cache-dir -r "$REQ"
fi

if [ -f "$REPO_DIR/pyproject.toml" ] || [ -f "$REPO_DIR/setup.py" ]; then
  python -m pip install --no-cache-dir -e "$REPO_DIR"
fi

python -c "import sys; print(f'[startup-hook] python {sys.version.split()[0]} OK')"
pip -V
echo "[startup-hook] done"
