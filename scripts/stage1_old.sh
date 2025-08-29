#!/bin/bash

# Set environment variables to fix Hugging Face cache issues
export HF_HOME="/nfs/scratch_2/sohir.maskey/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Create cache directory if it doesn't exist
mkdir -p "$HF_HOME"

# If you have a local Ollama binary at a custom path, prefer it.
# Update this path if your binary is elsewhere.
OLLAMA_LOCAL="/nfs/scratch_2/hans_higl/padawan/ollama/bin/ollama"
if [ -x "${OLLAMA_LOCAL}" ]; then
    export OLLAMA_BIN="${OLLAMA_LOCAL}"
    # Ensure the ollama directory is on PATH so 'command -v ollama' finds it
    export PATH="$(dirname "${OLLAMA_BIN}"):$PATH"
fi

# Pre-download sentence transformer model
echo "Pre-downloading SentenceTransformer model..."
python -c "
from sentence_transformers import SentenceTransformer
try:
    print('Downloading all-MiniLM-L6-v2...')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('SentenceTransformer model downloaded successfully!')
except Exception as e:
    print(f'Error downloading model: {e}')
    exit(1)
"

# Check if pre-download was successful
if [ $? -ne 0 ]; then
    echo "Failed to pre-download model. Exiting."
    exit 1
fi


# Ensure OpenIE won't try an unreachable LLM: prefer local spaCy fallback
export GFM_OPENIE_USE_LLM="true"    # disable LLM-based OpenIE (set to "true" to enable)
# export GFM_OPENIE_FALLBACK="spacy"   # use 'spacy' (local) or 'simple' fallback extractor

# If you want LLM mode instead, start Ollama or set an API key:
# - Run Ollama (e.g. `ollama serve`) with the model configured, or
# - Export OPENAI_API_KEY for OpenAI-based LLMs and unset GFM_OPENIE_USE_LLM

# ---------------------------------------------------------------------
# New: If user enables LLM OpenIE, verify Ollama is available and running.
# This avoids "Connection refused" and provides explicit install/run steps.
# ---------------------------------------------------------------------
if [ "${GFM_OPENIE_USE_LLM}" = "true" ]; then
    echo "LLM OpenIE requested (GFM_OPENIE_USE_LLM=true) — checking Ollama availability..."
    # Prefer explicit OLLAMA_BIN if set, otherwise rely on command -v
    if [ -n "${OLLAMA_BIN}" ] && [ -x "${OLLAMA_BIN}" ]; then
        OLLAMA_CMD="${OLLAMA_BIN}"
    elif command -v ollama >/dev/null 2>&1; then
        OLLAMA_CMD="$(command -v ollama)"
    else
        OLLAMA_CMD=""
    fi

    if [ -n "${OLLAMA_CMD}" ]; then
        # quick health check against local Ollama server (default port 11434)
        if ! curl -sS --max-time 2 http://127.0.0.1:11434/ >/dev/null 2>&1; then
            cat <<'MSG'

Ollama executable found but the Ollama server does not appear to be running (connection refused).
To run Ollama locally:
  1) Start the Ollama daemon (in background or tmux), e.g.:
     ${OLLAMA_CMD} serve &
  2) Pull or install a model (example uses 'mistral' as in your config):
     ${OLLAMA_CMD} pull mistral

After the server is running and the model is pulled, re-run this script.

MSG
            exit 1
        else
            echo "Ollama server is reachable at http://127.0.0.1:11434/ (using ${OLLAMA_CMD})"
        fi
    else
        cat <<'MSG'

Ollama binary not found on PATH and no local OLLAMA_BIN configured.
You said your ollama binary lives at /nfs/scratch_2/hans_higl/padawan/ollama/bin/ollama — ensure that path is executable.
If so, re-run this script; the script will detect it automatically.
If you prefer, install Ollama and run its server, for example:
  curl -fsSL https://ollama.ai/install.sh | bash
  ollama serve &
Then pull a model, e.g.:
  ollama pull mistral

MSG
        exit 1
    fi
fi

# # Build the index for testing dataset
# N_GPU=2
# DATA_ROOT="data"
# DATA_NAME_LIST="hotpotqa_test 2wikimultihopqa_test musique_test"
# for DATA_NAME in ${DATA_NAME_LIST}; do
#    echo "Processing $DATA_NAME..."
#    python -m gfmrag.workflow.stage1_index_dataset \
#    dataset.root=${DATA_ROOT} \
#    dataset.data_name=${DATA_NAME} \
#    el_model._target_=gfmrag.kg_construction.entity_linking_model.simple_el_model.SimpleELModel \
#    el_model.model_name_or_path=all-MiniLM-L6-v2 \
#    el_model.root=tmp/simple_el
   
#    # Check if the command failed and exit if so
#    if [ $? -ne 0 ]; then
#        echo "Error processing $DATA_NAME. Exiting."
#        exit 1
#    fi
# done

# Build the index for training dataset
N_GPU=2
DATA_ROOT="data"
DATA_NAME_LIST="hotpotqa_train musique_train 2wikimultihopqa_train"
START_N=0
END_N=1
for i in $(seq ${START_N} ${END_N}); do
   for DATA_NAME in ${DATA_NAME_LIST}; do
      DATA_NAME=${DATA_NAME}${i}
      echo "Processing $DATA_NAME..."
      python -m gfmrag.workflow.stage1_index_dataset \
      dataset.root=${DATA_ROOT} \
      dataset.data_name=${DATA_NAME} \
      el_model._target_=gfmrag.kg_construction.entity_linking_model.simple_el_model.SimpleELModel \
      el_model.model_name_or_path=all-MiniLM-L6-v2 \
      el_model.root=tmp/simple_el
      
      # Check if the command failed and exit if so
      if [ $? -ne 0 ]; then
          echo "Error processing $DATA_NAME. Exiting."
          exit 1
      fi
   done
done