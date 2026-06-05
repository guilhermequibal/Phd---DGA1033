# =============================================================
# CELL 5: EXECUTION ENVIRONMENT — SCREEN + RESOURCE SETUP
# =============================================================
#
# RUN THIS CELL BEFORE LAUNCHING CELLS 3 AND 4 in a new session,
# OR use the generated launch_rag.sh to run everything headlessly.
#
# ─── WHY SCREEN? ─────────────────────────────────────────────
# Cells 3 and 4 can take hours.  If your SSH/notebook connection
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
# This applies to both plain scripts and Jupyter kernels.
#
# ─── GPU / CPU RULES (violations cause instant kill) ─────────
# • One GPU per user → always set CUDA_VISIBLE_DEVICES=0  (done in Cell 1)
# • Thread caps      → set in Cell 1 before any ML import
# =============================================================

import os

# Verify that the resource limits from Cell 1 are active.
required_vars = {
    'OMP_NUM_THREADS':        '8',
    'OPENBLAS_NUM_THREADS':   '8',
    'MKL_NUM_THREADS':        '8',
    'VECLIB_MAXIMUM_THREADS': '8',
    'NUMEXPR_NUM_THREADS':    '8',
    'CUDA_VISIBLE_DEVICES':   '0',
}
missing = [k for k, v in required_vars.items() if os.environ.get(k) != v]
if missing:
    print(f'⚠️  Run Cell 1 first — these env vars are not set: {missing}')
else:
    print('✅ Resource limits verified (threads=8, GPU=0).')

# ------------------------------------------------------------------
# Write a convenience launcher so you can kick off the full experiment
# from a screen session without keeping Jupyter open.
# ------------------------------------------------------------------
LAUNCHER = """\
#!/bin/bash
# Launch the full RAG experiment inside a detached screen session.
# Usage: bash launch_rag.sh [session_name]
#
# The script concatenates all PDF Cell *.py files in order and runs
# them as a single Python process, writing output to results/run.log.
set -e

SESSION="${1:-rag_experiment}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/results/run.log"

if screen -list 2>/dev/null | grep -q "\\.$SESSION"; then
    echo "Session '$SESSION' already running. Attach with: screen -r $SESSION"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/results"

# Concatenate all cells into one runnable script
COMBINED="$SCRIPT_DIR/results/_run_all_cells.py"
cat "$SCRIPT_DIR/PDF Cell 1.py" \\
    "$SCRIPT_DIR/PDF Cell 2.py" \\
    "$SCRIPT_DIR/PDF Cell 3.py" \\
    "$SCRIPT_DIR/PDF Cell 4.py" > "$COMBINED"

screen -dmS "$SESSION" bash -c "
    cd '$SCRIPT_DIR'
    CUDA_VISIBLE_DEVICES=0 python '$COMBINED' 2>&1 | tee '$LOG'
    echo '--- Experiment finished. Press Enter to close session. ---'
    read
"

echo "Launched in background screen session: $SESSION"
echo "Attach:  screen -r $SESSION"
echo "Detach:  Ctrl+A then D"
echo "Monitor: tail -f $LOG"
"""

launcher_path = os.path.join(PROJECT_DIR, 'launch_rag.sh')
with open(launcher_path, 'w') as f:
    f.write(LAUNCHER)
os.chmod(launcher_path, 0o755)

print(f'✅ Launcher script written → {launcher_path}')
print()
print('To run the full experiment headlessly:')
print('  bash launch_rag.sh              # starts session "rag_experiment"')
print('  bash launch_rag.sh meuteste     # or give a custom session name')
print()
print('To monitor:')
print('  screen -r rag_experiment        # attach')
print('  tail -f results/run.log         # or just watch the log file')
