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
    LAUNCHER = r'''#!/bin/bash
# Launch the full RAG experiment inside a detached screen session.
# Usage: bash launch_rag.sh [session_name]
#
# Concatenates all Experiment Cell *.py files in order and runs them as a single
# Python process, writing stdout/stderr to results/run.log.
set -euo pipefail

SESSION="${1:-rag_experiment}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/results/run.log"

CELLS=("Experiment Cell 1.py" "Experiment Cell 2.py" "Experiment Cell 3.py" "Experiment Cell 4.py")
for c in "${CELLS[@]}"; do
  if [ ! -f "$SCRIPT_DIR/$c" ]; then
    echo "Missing cell file: $c" >&2
    exit 1
  fi
done

if screen -list 2>/dev/null | grep -q "\.$SESSION"; then
  echo "Session '$SESSION' already running. Attach with: screen -r $SESSION"
  exit 1
fi

mkdir -p "$SCRIPT_DIR/results"

# Concatenate all cells into one runnable script
COMBINED="$SCRIPT_DIR/results/_run_all_cells.py"
cat "$SCRIPT_DIR/Experiment Cell 1.py" \
    "$SCRIPT_DIR/Experiment Cell 2.py" \
    "$SCRIPT_DIR/Experiment Cell 3.py" \
    "$SCRIPT_DIR/Experiment Cell 4.py" > "$COMBINED"

screen -dmS "$SESSION" bash -c "
cd '$SCRIPT_DIR'
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python '$COMBINED' 2>&1 | tee '$LOG'
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
    print("   tail -f results/run.log       # or just watch the log file")
