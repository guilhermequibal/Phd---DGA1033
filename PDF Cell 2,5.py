# =============================================================
# CELL 5: EXECUTION ENVIRONMENT — SCREEN + RESOURCE SETUP
# =============================================================
#
# RUN THIS CELL BEFORE LAUNCHING CELLS 3 AND 4 in a new session,
# OR use the generated launch_rag.sh to run everything headlessly.
#
# ─── WHY SCREEN? ─────────────────────────────────────────────
# Cells 3 and 4 can take hours. If your SSH/notebook connection
# drops, the kernel dies and all progress is lost.
# Running inside `screen` decouples the process from the terminal:
# the job keeps running even after you close the browser tab.
#
# QUICK-START (run in a server terminal, not inside Jupyter):
#
#   screen -S meuteste            # create a session named "meuteste"
#   python my_experiment.py       # run your script inside it
#   # Press Ctrl+A then D         # detach — safe to close the tab now
#
# RECONNECT LATER:
#   screen -ls                    # list active sessions
#   screen -r meuteste            # reconnect and see what is running
#
# MENTAL RULE: everything that needs to survive a notebook shutdown
# must run inside `screen` (or `tmux`, its modern equivalent).
#
# ─── GPU / CPU RULES (violations cause instant kill) ─────────
#  • One GPU per user → always set CUDA_VISIBLE_DEVICES=0 (done in Cell 1)
#  • Thread caps → set in Cell 1 before any ML import
# =============================================================
import os

# FIX: verify against the single source of truth defined in Cell 1
#      (REQUIRED_ENV) instead of re-hardcoding the expected values here.
#      Now if Cell 1 changes the thread count, this check follows automatically.
missing = [k for k, v in REQUIRED_ENV.items() if os.environ.get(k) != v]
if missing:
    print(f"⚠️ Run Cell 1 first — these env vars are not set correctly: {missing}")
else:
    print(f"✅ Resource limits verified "
          f"(threads={N_THREADS}, GPU={REQUIRED_ENV['CUDA_VISIBLE_DEVICES']}).")

# ------------------------------------------------------------------
# Write a convenience launcher so you can kick off the full experiment
# from a screen session without keeping Jupyter open.
# ------------------------------------------------------------------
# FIX: fail early if any cell file is missing, instead of writing a launcher
#      that would silently `cat` nothing and run a broken script.
CELL_FILES = ["PDF Cell 1.py", "PDF Cell 2.py", "PDF Cell 3.py", "PDF Cell 4.py"]
missing_files = [f for f in CELL_FILES
                 if not os.path.exists(os.path.join(PROJECT_DIR, f))]

if missing_files:
    print(f"⚠️ Launcher NOT written — missing cell files: {missing_files}")
else:
    # FIX: bash now uses `set -euo pipefail`, checks each file exists at run
    #      time, and reads CUDA_VISIBLE_DEVICES from the environment instead of
    #      hardcoding 0. The cell list matches CELL_FILES above to avoid drift.
    LAUNCHER = r'''#!/bin/bash
# Launch the full RAG experiment inside a detached screen session.
# Usage: bash launch_rag.sh [session_name]
#
# The script concatenates all PDF Cell *.py files in order and runs them as a
# single Python process, writing output to results/run.log.
set -euo pipefail

SESSION="${1:-rag_experiment}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/results/run.log"

CELLS=("PDF Cell 1.py" "PDF Cell 2.py" "PDF Cell 3.py" "PDF Cell 4.py")
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
cat "$SCRIPT_DIR/PDF Cell 1.py" \
    "$SCRIPT_DIR/PDF Cell 2.py" \
    "$SCRIPT_DIR/PDF Cell 3.py" \
    "$SCRIPT_DIR/PDF Cell 4.py" > "$COMBINED"

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
