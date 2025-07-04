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

# Path to the Python runner script (adjust if in another directory)
RUNNER_SCRIPT="run_notebook_as_script.py"

if [ ! -f "$RUNNER_SCRIPT" ]; then
  echo "Runner script '$RUNNER_SCRIPT' not found in current directory."
  exit 1
fi

# Start new tmux session and run the script
tmux new-session -d -s "$SESSION_NAME" "python $RUNNER_SCRIPT \"$NOTEBOOK\"; read -n 1 -s -r -p 'Press any key to exit...'" 

echo "✅ Started notebook in tmux session '$SESSION_NAME'."
echo "You can attach to it anytime with:"
echo "  tmux attach -t $SESSION_NAME"