#!/bin/bash

NOTEBOOK="$1"
SESSION_NAME="notebook_runner"

if [ -z "$NOTEBOOK" ]; then
  echo "Usage: $0 path/to/notebook.ipynb"
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Please install it first."
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Python is not available in PATH. Make sure your environment is activated."
  exit 1
fi

# Resolve script directory and runner path relative to this file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_SCRIPT="$SCRIPT_DIR/run_notebook_as_script.py"

if [ ! -f "$RUNNER_SCRIPT" ]; then
  echo "Runner script '$RUNNER_SCRIPT' not found in current directory."
  exit 1
fi

# Try to activate project venv if present
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ACTIVATE_VENV=""
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  ACTIVATE_VENV="source $PROJECT_ROOT/.venv/bin/activate && "
fi

# Start new tmux session and run the script from notebooks dir
tmux new-session -d -s "$SESSION_NAME" "cd $SCRIPT_DIR && ${ACTIVATE_VENV}python $RUNNER_SCRIPT \"$NOTEBOOK\"; read -n 1 -s -r -p 'Press any key to exit...'" 

echo "✅ Started notebook in tmux session '$SESSION_NAME'."
echo "You can attach to it anytime with:"
echo "  tmux attach -t $SESSION_NAME"