# #!/bin/bash
# set -euo pipefail

# # Hugging Face caches
# export HF_HOME="${HF_HOME:-/nfs/scratch_2/sohir.maskey/.cache/huggingface}"
# export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
# export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
# mkdir -p "$HF_HOME"

# # Optional CPU/Tokenizer tuning
# export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# # Prefer local Ollama binary on PATH if available
# OLLAMA_LOCAL="${OLLAMA_LOCAL:-/nfs/scratch_2/hans_higl/padawan/ollama/bin/ollama}"
# if [ -x "${OLLAMA_LOCAL}" ]; then
#   export OLLAMA_BIN="${OLLAMA_LOCAL}"
#   export PATH="$(dirname "${OLLAMA_BIN}"):$PATH"
# fi

# # Pre-download sentence transformer model
# echo "Pre-downloading SentenceTransformer model..."
# python - <<'PY'
# from sentence_transformers import SentenceTransformer
# print('Downloading all-MiniLM-L6-v2...')
# SentenceTransformer('all-MiniLM-L6-v2')
# print('SentenceTransformer model downloaded successfully!')
# PY

# # LLM OpenIE toggle (align with your Hydra defaults)
# export GFM_OPENIE_USE_LLM="${GFM_OPENIE_USE_LLM:-true}"

# # If LLM OpenIE is requested, ensure Ollama is reachable
# if [ "${GFM_OPENIE_USE_LLM}" = "true" ]; then
#   if [ -n "${OLLAMA_BIN:-}" ] && [ -x "${OLLAMA_BIN}" ]; then
#     OLLAMA_CMD="${OLLAMA_BIN}"
#   elif command -v ollama >/dev/null 2>&1; then
#     OLLAMA_CMD="$(command -v ollama)"
#   else
#     OLLAMA_CMD=""
#   fi
#   if [ -n "${OLLAMA_CMD}" ]; then
#     if ! curl -sS --max-time 2 http://127.0.0.1:11434/ >/dev/null 2>&1; then
#       echo "Ollama found at ${OLLAMA_CMD} but server not running; start with: '${OLLAMA_CMD} serve &' and ensure model is pulled."
#       exit 1
#     fi
#     echo "Ollama server is reachable at http://127.0.0.1:11434/ (using ${OLLAMA_CMD})"
#   else
#     echo "Ollama not found on PATH; set OLLAMA_BIN or install/run Ollama."
#     exit 1
#   fi
# fi

# # Determine the single dataset to run (from Determined hyperparams or manual env)
# SINGLE_DATASET="${STAGE1_DATA_NAME:-${DET_HPARAM_DATA_NAME:-}}"
# if [ -z "${SINGLE_DATASET}" ]; then
#   echo "ERROR: No dataset specified. Set DET_HPARAM_DATA_NAME (preferred) or STAGE1_DATA_NAME."
#   exit 1
# fi

# # Determine available GPUs from Determined (or default to all visible) and pin to the first slot
# if [ -n "${DET_SLOT_IDS:-}" ]; then
#   IFS=',' read -r -a GPU_IDS <<< "${DET_SLOT_IDS}"
# elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
#   IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
# else
#   GPU_IDS=(0)
# fi
# export CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}"
# echo "Using GPU ${CUDA_VISIBLE_DEVICES} for dataset ${SINGLE_DATASET}"
# # New: show interpreter to verify startup-hook env is active
# echo "Python interpreter: $(command -v python)"
# python -V

# # Common knobs (env or Determined hyperparams)
# DATA_ROOT="${DATA_ROOT:-data}"
# KG_NUM_PROCESSES="${KG_NUM_PROCESSES:-${DET_HPARAM_KG_NUM_PROCESSES:-10}}"
# QA_NUM_PROCESSES="${QA_NUM_PROCESSES:-${DET_HPARAM_QA_NUM_PROCESSES:-10}}"
# EXTRA_OVERRIDES="${STAGE1_HYDRA_OVERRIDES:-}"

# # Run stage1 once for the selected dataset
# python -m gfmrag.workflow.stage1_index_dataset \
#   dataset.root="${DATA_ROOT}" \
#   dataset.data_name="${SINGLE_DATASET}" \
#   el_model._target_=gfmrag.kg_construction.entity_linking_model.simple_el_model.SimpleELModel \
#   el_model.model_name_or_path=all-MiniLM-L6-v2 \
#   el_model.root=tmp/simple_el \
#   kg_constructor.num_processes="${KG_NUM_PROCESSES}" \
#   qa_constructor.num_processes="${QA_NUM_PROCESSES}" \
#   ${EXTRA_OVERRIDES}

# echo "Dataset ${SINGLE_DATASET} completed."

#!/bin/bash
set -euo pipefail

# ---------- CPU threading ----------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=true

# ---------- Hugging Face cache ----------
export HF_HOME="${HF_HOME:-/nfs/scratch_2/sohir.maskey/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME"

# ---------- Optional local Ollama binary ----------
if [ -n "${OLLAMA_LOCAL:-}" ] && [ -x "${OLLAMA_LOCAL}" ]; then
  export OLLAMA_BIN="${OLLAMA_LOCAL}"
  export PATH="$(dirname "${OLLAMA_BIN}"):$PATH"
fi

# ---------- Pre-download SentenceTransformer once ----------
python - <<'PY'
from sentence_transformers import SentenceTransformer
print("Downloading all-MiniLM-L6-v2 ...")
SentenceTransformer('all-MiniLM-L6-v2')
print("OK")
PY

# ---------- LLM OpenIE gate (start Ollama if needed) ----------
if [ "${GFM_OPENIE_USE_LLM:-false}" = "true" ]; then
  if command -v curl >/dev/null 2>&1; then
    if ! curl -sS --max-time 2 http://127.0.0.1:11434/ >/dev/null 2>&1; then
      echo "Ollama server not reachable; starting in background..."
      ${OLLAMA_BIN:-ollama} serve &
      sleep 5  # Wait for startup
      if ! curl -sS --max-time 2 http://127.0.0.1:11434/ >/dev/null 2>&1; then
        echo "Failed to start Ollama server."
        exit 1
      fi
      echo "Ollama server started and reachable."
    else
      echo "Ollama server already running."
    fi
  fi
fi
curl -X POST http://127.0.0.1:11434/api/generate -H "Content-Type: application/json" -d '{"model":"mistral:latest","prompt":"test","stream":false}' >/dev/null 2>&1 &

# ---------- Inputs ----------
DATA_ROOT="${DATA_ROOT:-/nfs/scratch_2/sohir.maskey/gfm-rag/data}"

# precedence: CLI arg > DATA_NAME > Determined hyperparam env
DATA_NAME="${1:-${DATA_NAME:-${DET_HPARAM_data_name:-}}}"
if [ -z "${DATA_NAME:-}" ]; then
  echo "Missing data_name. Pass as arg, set DATA_NAME, or rely on DET_HPARAM_data_name."
  exit 1
fi

# # Optional Determined hyperparams
# KG_NP="${KG_NUM_PROCESSES:-${DET_HPARAM_kg_num_processes:-}}"
# QA_NP="${QA_NUM_PROCESSES:-${DET_HPARAM_qa_num_processes:-}}"

# # Convert to int
# KG_NP="${KG_NP:-0}"
# QA_NP="${QA_NP:-0}"

# ---------- Hydra overrides (NER/ER kept exactly as before) ----------
OVR=(
  "dataset.root=${DATA_ROOT}"
  "dataset.data_name=${DATA_NAME}"
  "el_model._target_=gfmrag.kg_construction.entity_linking_model.simple_el_model.SimpleELModel"
  "el_model.model_name_or_path=all-MiniLM-L6-v2"
  "el_model.root=tmp/simple_el"
)

# [ -n "${KG_NP}" ] && OVR+=("kg_num_processes=${KG_NP}")
# [ -n "${QA_NP}" ] && OVR+=("qa_num_processes=${QA_NP}")

# ---------- Run (single job, no parallelization) ----------
python -m gfmrag.workflow.stage1_index_dataset "${OVR[@]}"
