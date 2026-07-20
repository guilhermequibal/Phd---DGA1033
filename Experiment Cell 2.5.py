# =============================================================
# CELL 2.5: EXECUTION ENVIRONMENT — SCREEN + RESOURCE SETUP
#
# STRATEGY:
# This cell serves two purposes before launching the long-running
# experiment cells (3 and 4):
#
# 1. Environment verification — confirms that the resource caps defined
#    in Cell 1 (CPU threads, GPU assignment) are actually active in the
#    current process. If Cell 1 was not run first, the caps may be absent
#    and the job could violate shared-server policies.
#
# 2. Headless launcher — writes a shell script (launch_rag.sh) that runs
#    the full experiment inside a `screen` session. This is necessary
#    because Cells 3 and 4 can take several hours; if the SSH or notebook
#    connection drops, the kernel dies and all accumulated results are
#    lost. Running inside `screen` decouples the process from the
#    terminal, so the job continues even after the browser tab is closed.
#
# QUICK-START (run in a server terminal, not inside Jupyter):
#
#   screen -S meuteste            # create a named session
#   python my_experiment.py       # run the script inside it
#   # Press Ctrl+A then D         # detach — safe to close the tab now
#
# RECONNECT LATER:
#   screen -ls                    # list active sessions
#   screen -r meuteste            # reconnect and see what is running
# =============================================================
import os

# Verify against the REQUIRED_ENV dict defined in Cell 1 as the single
# source of truth. If thread counts or GPU assignment change in Cell 1,
# this check automatically reflects the new expected values.
missing = [k for k, v in REQUIRED_ENV.items() if os.environ.get(k) != v]
if missing:
    print(f"⚠️ Run Cell 1 first — these env vars are not set correctly: {missing}")
else:
    print(f"✅ Resource limits verified "
          f"(threads={N_THREADS}, GPU={REQUIRED_ENV['CUDA_VISIBLE_DEVICES']}).")

# ------------------------------------------------------------------
# Write a convenience launcher to kick off the full experiment from a
# screen session without keeping Jupyter open.
# ------------------------------------------------------------------
# Fail early if any required cell file is missing, so the launcher is
# never written in a broken state.
CELL_FILES = ["Experiment Cell 1.py", "Experiment Cell 2.py", "Experiment Cell 3.py", "Experiment Cell 4.py"]
missing_files = [f for f in CELL_FILES
                 if not os.path.exists(os.path.join(PROJECT_DIR, f))]

if missing_files:
    print(f"⚠️ Launcher NOT written — missing cell files: {missing_files}")
else:
    # OLLAMA_BIN points to the GPU-enabled official binary installed under ~/.local.
    # The conda-bundled `ollama` binary ships without CUDA backend libraries and
    # falls back to CPU, which makes 70B inference impractically slow on this server.
    OLLAMA_BIN_PATH = os.path.join(
        os.path.expanduser("~"), ".local", "ollama", "bin", "ollama"
    )
    OLLAMA_MODELS_PATH = os.path.join(os.path.expanduser("~"), ".ollama", "models")
    OLLAMA_CUDA_LIB = os.path.join(
        os.path.expanduser("~"), ".local", "ollama", "lib", "ollama", "cuda_v12"
    )
    OLLAMA_LIB = os.path.join(
        os.path.expanduser("~"), ".local", "ollama", "lib", "ollama"
    )

    LAUNCHER = f'''#!/bin/bash
# Launch the full RAG experiment inside a detached screen session.
# Usage: bash launch_rag.sh [session_name]
#
# Starts the GPU-enabled Ollama server (if not already running), then
# concatenates all Experiment Cell *.py files in order and runs them as a
# single Python process, writing stdout/stderr to 03_outputs/run.log.
#
# Requires the official Ollama binary at {OLLAMA_BIN_PATH}
# (installed separately from the conda-bundled CPU-only version).
set -euo pipefail

SESSION="${{1:-rag_experiment}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/03_outputs/run.log"
GPU="${{EXPERIMENT_GPU_DEVICE:-2}}"
OLLAMA_BIN="{OLLAMA_BIN_PATH}"
OLLAMA_MODELS="{OLLAMA_MODELS_PATH}"
OLLAMA_CUDA_LIB="{OLLAMA_CUDA_LIB}"
OLLAMA_LIB="{OLLAMA_LIB}"

CELLS=("Experiment Cell 1.py" "Experiment Cell 2.py" "Experiment Cell 3.py" "Experiment Cell 4.py")
for c in "${{CELLS[@]}}"; do
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

if screen -list 2>/dev/null | grep -q "\\.$SESSION"; then
  echo "Session '$SESSION' already running. Attach with: screen -r $SESSION"
  exit 1
fi

mkdir -p "$SCRIPT_DIR/03_outputs"

# Start the GPU-enabled Ollama server if it is not already listening.
if ! curl -sf http://localhost:11434/api/version > /dev/null 2>&1; then
  echo "Starting Ollama on GPU $GPU ..."
  CUDA_VISIBLE_DEVICES="$GPU" \\
  OLLAMA_MODELS="$OLLAMA_MODELS" \\
  LD_LIBRARY_PATH="$OLLAMA_LIB:$OLLAMA_CUDA_LIB:${{LD_LIBRARY_PATH:-}}" \\
  nohup "$OLLAMA_BIN" serve > /tmp/ollama_serve.log 2>&1 &
  # Wait up to 30 s for Ollama to be ready before proceeding.
  for i in $(seq 1 30); do
    curl -sf http://localhost:11434/api/version > /dev/null 2>&1 && break
    sleep 1
  done
  curl -sf http://localhost:11434/api/version > /dev/null 2>&1 || {{
    echo "Ollama did not start in time. Check /tmp/ollama_serve.log" >&2; exit 1
  }}
  echo "Ollama ready."
else
  echo "Ollama already running."
fi

# Concatenate all cells into one runnable script.
# Stored at the project root (not inside 03_outputs/) so that __file__
# resolves to the correct PROJECT_DIR in Cell 1 and all relative paths stay
# valid — the 01_frozen / 02_adjustable / 03_outputs subfolders hang off it.
COMBINED="$SCRIPT_DIR/_run_all_cells.py"
cat "$SCRIPT_DIR/Experiment Cell 1.py" \\
    "$SCRIPT_DIR/Experiment Cell 2.py" \\
    "$SCRIPT_DIR/Experiment Cell 3.py" \\
    "$SCRIPT_DIR/Experiment Cell 4.py" > "$COMBINED"

screen -dmS "$SESSION" bash -c "
cd '$SCRIPT_DIR'
EXPERIMENT_GPU_DEVICE=${{GPU}} python '$COMBINED' 2>&1 | tee '$LOG'
echo '--- Experiment finished. Press Enter to close session. ---'
read
"

echo "Launched in background screen session: $SESSION"
echo "Attach:  screen -r $SESSION"
echo "Detach:  Ctrl+A then D"
echo "Monitor: tail -f $LOG"
'''

    launcher_path = os.path.join(PROJECT_DIR, "launch_rag.sh")
    with open(launcher_path, "w") as f:
        f.write(LAUNCHER)
    os.chmod(launcher_path, 0o755)

    print(f"✅ Launcher script written → {launcher_path}")
    print()
    print("To run the full experiment headlessly:")
    print("   bash launch_rag.sh            # starts session 'rag_experiment'")
    print("   bash launch_rag.sh meuteste   # or give a custom session name")
    print()
    print("To monitor:")
    print("   screen -r rag_experiment      # attach")
    print("   tail -f 03_outputs/run.log    # or just watch the log file")
