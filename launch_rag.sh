#!/bin/bash
# Launch the full RAG experiment inside a detached screen session.
# Usage: bash launch_rag.sh [session_name]
#
# Starts the GPU-enabled Ollama server (if not already running), then
# concatenates all Experiment Cell *.py files in order and runs them as a
# single Python process, writing stdout/stderr to results/run.log.
#
# Requires the official Ollama binary at /export/livia/home/vision/Gbaldessin/.local/ollama/bin/ollama
# (installed separately from the conda-bundled CPU-only version).
set -euo pipefail

SESSION="${1:-rag_experiment}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/results/run.log"
GPU="${EXPERIMENT_GPU_DEVICE:-2}"
OLLAMA_BIN="/export/livia/home/vision/Gbaldessin/.local/ollama/bin/ollama"
OLLAMA_MODELS="/export/livia/home/vision/Gbaldessin/.ollama/models"
OLLAMA_CUDA_LIB="/export/livia/home/vision/Gbaldessin/.local/ollama/lib/ollama/cuda_v12"
OLLAMA_LIB="/export/livia/home/vision/Gbaldessin/.local/ollama/lib/ollama"

CELLS=("Experiment Cell 1.py" "Experiment Cell 2.py" "Experiment Cell 3.py" "Experiment Cell 4.py")
for c in "${CELLS[@]}"; do
  if [ ! -f "$SCRIPT_DIR/$c" ]; then
    echo "Missing cell file: $c" >&2
    exit 1
  fi
done

if [ ! -x "$OLLAMA_BIN" ]; then
  echo "Ollama binary not found at $OLLAMA_BIN" >&2
  echo "Run the setup script to install the GPU-enabled Ollama binary." >&2
  exit 1
fi

if screen -list 2>/dev/null | grep -q "\.$SESSION"; then
  echo "Session '$SESSION' already running. Attach with: screen -r $SESSION"
  exit 1
fi

mkdir -p "$SCRIPT_DIR/results"

# Start the GPU-enabled Ollama server if it is not already listening.
if ! curl -sf http://localhost:11434/api/version > /dev/null 2>&1; then
  echo "Starting Ollama on GPU $GPU ..."
  CUDA_VISIBLE_DEVICES="$GPU" \
  OLLAMA_MODELS="$OLLAMA_MODELS" \
  LD_LIBRARY_PATH="$OLLAMA_LIB:$OLLAMA_CUDA_LIB:${LD_LIBRARY_PATH:-}" \
  nohup "$OLLAMA_BIN" serve > /tmp/ollama_serve.log 2>&1 &
  # Wait up to 30 s for Ollama to be ready before proceeding.
  for i in $(seq 1 30); do
    curl -sf http://localhost:11434/api/version > /dev/null 2>&1 && break
    sleep 1
  done
  curl -sf http://localhost:11434/api/version > /dev/null 2>&1 || {
    echo "Ollama did not start in time. Check /tmp/ollama_serve.log" >&2; exit 1
  }
  echo "Ollama ready."
else
  echo "Ollama already running."
fi

# Concatenate all cells into one runnable script.
# Stored at the project root (not inside results/) so that __file__ resolves
# to the correct PROJECT_DIR in Cell 1 and all relative paths stay valid.
COMBINED="$SCRIPT_DIR/_run_all_cells.py"
cat "$SCRIPT_DIR/Experiment Cell 1.py" \
    "$SCRIPT_DIR/Experiment Cell 2.py" \
    "$SCRIPT_DIR/Experiment Cell 3.py" \
    "$SCRIPT_DIR/Experiment Cell 4.py" > "$COMBINED"

screen -dmS "$SESSION" bash -c "
cd '$SCRIPT_DIR'
EXPERIMENT_GPU_DEVICE=${GPU} python '$COMBINED' 2>&1 | tee '$LOG'
echo '--- Experiment finished. Press Enter to close session. ---'
read
"

echo "Launched in background screen session: $SESSION"
echo "Attach:  screen -r $SESSION"
echo "Detach:  Ctrl+A then D"
echo "Monitor: tail -f $LOG"
