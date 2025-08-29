# #!/bin/bash

# # Set environment variables to fix Hugging Face cache issues
# export HF_HOME="/nfs/scratch_2/sohir.maskey/.cache/huggingface"
# export TRANSFORMERS_CACHE="$HF_HOME"
# export HF_DATASETS_CACHE="$HF_HOME/datasets"

# # Create cache directory if it doesn't exist
# mkdir -p "$HF_HOME"

# # If you have a local Ollama binary at a custom path, prefer it.
# # Update this path if your binary is elsewhere.
# OLLAMA_LOCAL="/nfs/scratch_2/hans_higl/padawan/ollama/bin/ollama"
# if [ -x "${OLLAMA_LOCAL}" ]; then
#     export OLLAMA_BIN="${OLLAMA_LOCAL}"
#     # Ensure the ollama directory is on PATH so 'command -v ollama' finds it
#     export PATH="$(dirname "${OLLAMA_BIN}"):$PATH"
# fi

# # Pre-download sentence transformer model
# echo "Pre-downloading SentenceTransformer model..."
# python -c "
# from sentence_transformers import SentenceTransformer
# try:
#     print('Downloading all-MiniLM-L6-v2...')
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     print('SentenceTransformer model downloaded successfully!')
# except Exception as e:
#     print(f'Error downloading model: {e}')
#     exit(1)
# "

# # Check if pre-download was successful
# if [ $? -ne 0 ]; then
#     echo "Failed to pre-download model. Exiting."
#     exit 1
# fi


# # Ensure OpenIE won't try an unreachable LLM: prefer local spaCy fallback
# export GFM_OPENIE_USE_LLM="true"    # disable LLM-based OpenIE (set to "true" to enable)
# # export GFM_OPENIE_FALLBACK="spacy"   # use 'spacy' (local) or 'simple' fallback extractor

# # If you want LLM mode instead, start Ollama or set an API key:
# # - Run Ollama (e.g. `ollama serve`) with the model configured, or
# # - Export OPENAI_API_KEY for OpenAI-based LLMs and unset GFM_OPENIE_USE_LLM

# # ---------------------------------------------------------------------
# # New: If user enables LLM OpenIE, verify Ollama is available and running.
# # This avoids "Connection refused" and provides explicit install/run steps.
# # ---------------------------------------------------------------------
# if [ "${GFM_OPENIE_USE_LLM}" = "true" ]; then
#     echo "LLM OpenIE requested (GFM_OPENIE_USE_LLM=true) — checking Ollama availability..."
#     # Prefer explicit OLLAMA_BIN if set, otherwise rely on command -v
#     if [ -n "${OLLAMA_BIN}" ] && [ -x "${OLLAMA_BIN}" ]; then
#         OLLAMA_CMD="${OLLAMA_BIN}"
#     elif command -v ollama >/dev/null 2>&1; then
#         OLLAMA_CMD="$(command -v ollama)"
#     else
#         OLLAMA_CMD=""
#     fi

#     if [ -n "${OLLAMA_CMD}" ]; then
#         # quick health check against local Ollama server (default port 11434)
#         if ! curl -sS --max-time 2 http://127.0.0.1:11434/ >/dev/null 2>&1; then
#             cat <<'MSG'

# Ollama executable found but the Ollama server does not appear to be running (connection refused).
# To run Ollama locally:
#   1) Start the Ollama daemon (in background or tmux), e.g.:
#      ${OLLAMA_CMD} serve &
#   2) Pull or install a model (example uses 'mistral' as in your config):
#      ${OLLAMA_CMD} pull mistral

# After the server is running and the model is pulled, re-run this script.

# MSG
#             exit 1
#         else
#             echo "Ollama server is reachable at http://127.0.0.1:11434/ (using ${OLLAMA_CMD})"
#         fi
#     else
#         cat <<'MSG'

# Ollama binary not found on PATH and no local OLLAMA_BIN configured.
# You said your ollama binary lives at /nfs/scratch_2/hans_higl/padawan/ollama/bin/ollama — ensure that path is executable.
# If so, re-run this script; the script will detect it automatically.
# If you prefer, install Ollama and run its server, for example:
#   curl -fsSL https://ollama.ai/install.sh | bash
#   ollama serve &
# Then pull a model, e.g.:
#   ollama pull mistral

# MSG
#         exit 1
#     fi
# fi

# # # Build the index for testing dataset
# # N_GPU=2
# # DATA_ROOT="data"
# # DATA_NAME_LIST="hotpotqa_test 2wikimultihopqa_test musique_test"
# # for DATA_NAME in ${DATA_NAME_LIST}; do
# #    echo "Processing $DATA_NAME..."
# #    python -m gfmrag.workflow.stage1_index_dataset \
# #    dataset.root=${DATA_ROOT} \
# #    dataset.data_name=${DATA_NAME} \
# #    el_model._target_=gfmrag.kg_construction.entity_linking_model.simple_el_model.SimpleELModel \
# #    el_model.model_name_or_path=all-MiniLM-L6-v2 \
# #    el_model.root=tmp/simple_el
   
# #    # Check if the command failed and exit if so
# #    if [ $? -ne 0 ]; then
# #        echo "Error processing $DATA_NAME. Exiting."
# #        exit 1
# #    fi
# # done

# # Build the index for training dataset
# N_GPU=2
# DATA_ROOT="data"
# DATA_NAME_LIST="hotpotqa_train musique_train 2wikimultihopqa_train"
# START_N=0
# END_N=19
# for i in $(seq ${START_N} ${END_N}); do
#    for DATA_NAME in ${DATA_NAME_LIST}; do
#       DATA_NAME=${DATA_NAME}${i}
#       echo "Processing $DATA_NAME..."
#       python -m gfmrag.workflow.stage1_index_dataset \
#       dataset.root=${DATA_ROOT} \
#       dataset.data_name=${DATA_NAME} \
#       el_model._target_=gfmrag.kg_construction.entity_linking_model.simple_el_model.SimpleELModel \
#       el_model.model_name_or_path=all-MiniLM-L6-v2 \
#       el_model.root=tmp/simple_el
      
#       # Check if the command failed and exit if so
#       if [ $? -ne 0 ]; then
#           echo "Error processing $DATA_NAME. Exiting."
#           exit 1
#       fi
#    done
# done

#!/bin/bash
set -euo pipefail

# Kill all children on exit
trap 'pkill -P $$ >/dev/null 2>&1 || true' EXIT

# CPU threading (tune to your box)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=true

# Hugging Face cache
export HF_HOME="/nfs/scratch_2/sohir.maskey/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"

# Optional local Ollama bin
OLLAMA_LOCAL="/nfs/scratch_2/hans_higl/padawan/ollama/bin/ollama"
if [ -x "${OLLAMA_LOCAL}" ]; then
  export OLLAMA_BIN="${OLLAMA_LOCAL}"
  export PATH="$(dirname "${OLLAMA_BIN}"):$PATH"
fi

# Pre-download ST model (once)
echo "Pre-downloading SentenceTransformer model..."
python - <<'PY'
from sentence_transformers import SentenceTransformer
print('Downloading all-MiniLM-L6-v2...')
SentenceTransformer('all-MiniLM-L6-v2')
print('OK')
PY

# Keep LLM OpenIE ON
export GFM_OPENIE_USE_LLM="true"
# Optional fallback used for sentences where LLM fails/timeouts
export GFM_OPENIE_FALLBACK="${GFM_OPENIE_FALLBACK:-spacy}"

# Verify Ollama is reachable
if ! curl -sS --max-time 2 http://127.0.0.1:11434/ >/dev/null 2>&1; then
  echo "Ollama server not reachable at http://127.0.0.1:11434/. Start it first: ${OLLAMA_BIN:-ollama} serve &"
  exit 1
fi

# Encourage the server to keep models hot and allow some concurrency (if supported by your Ollama version)
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
# If your server was started with this env, it will allow parallel requests; otherwise ignore
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}"

# Data shards
DATA_ROOT="data"
BASES=(hotpotqa_train musique_train 2wikimultihopqa_train)
START_N=15
END_N=15

# Concurrency (tune this; start with 2–4 when LLM is on GPU1)
MAX_JOBS="${MAX_JOBS:-4}"

LOG_DIR="logs/stage1_index_llm"
mkdir -p "${LOG_DIR}"

# Hydra overrides (add device/batch if your config supports it)
HYDRA_OVERRIDES=(
  "el_model._target_=gfmrag.kg_construction.entity_linking_model.simple_el_model.SimpleELModel"
  "el_model.model_name_or_path=all-MiniLM-L6-v2"
  "el_model.root=tmp/simple_el"
  # If your EL model supports device specification, uncomment ONE of these lines:
  # "el_model.device=cuda"          # generic CUDA
  # "el_model.device=cuda:0"        # explicitly GPU 0
  # "el_model.batch_size=1024"      # if supported; raise carefully to avoid OOM
  # "dataloader.num_workers=4"      # if supported by your pipeline
)

# Build task list
TASKS=()
for i in $(seq ${START_N} ${END_N}); do
  for base in "${BASES[@]}"; do
    TASKS+=("${base}${i}")
  done
done

echo "Launching ${#TASKS[@]} shards with MAX_JOBS=${MAX_JOBS}"
echo "Pinning Python to GPU 0 (CUDA_VISIBLE_DEVICES=0). Ollama should keep using GPU 1."

run_task() {
  local NAME="$1"
  echo "[START] ${NAME}  $(date)"
  # Pin this process to GPU 0 only, keep allocator sane
  CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}" \
  python -m gfmrag.workflow.stage1_index_dataset \
    dataset.root="${DATA_ROOT}" \
    dataset.data_name="${NAME}" \
    "${HYDRA_OVERRIDES[@]}" \
    > "${LOG_DIR}/${NAME}.out" 2> "${LOG_DIR}/${NAME}.err"
  local RC=$?
  if [ $RC -ne 0 ]; then
    echo "[FAIL ] ${NAME} (rc=$RC). See ${LOG_DIR}/${NAME}.err"
    return $RC
  fi
  echo "[DONE ] ${NAME}  $(date)"
}

active=0
declare -a pids=()

for NAME in "${TASKS[@]}"; do
  run_task "${NAME}" &
  pids+=($!)
  active=$((active + 1))

  # throttle to MAX_JOBS
  if [ "${active}" -ge "${MAX_JOBS}" ]; then
    if wait -n; then
      active=$((active - 1))
    else
      echo "A job failed — stopping."
      pkill -P $$ || true
      exit 1
    fi
  fi
done
# inspect via
# tail -n +1 -F logs/stage1_index_llm/*.out logs/stage1_index_llm/*.err

# Wait remaining
wait
echo "All tasks finished."