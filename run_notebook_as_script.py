import subprocess
import sys
import os

def convert_notebook_to_script(notebook_path):
    if not os.path.isfile(notebook_path):
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    print(f"Converting notebook to script: {notebook_path}")
    result = subprocess.run([
        "jupyter", "nbconvert", "--to", "script", notebook_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("Failed to convert notebook:\n", result.stderr)
        sys.exit(1)

    script_path = notebook_path.replace(".ipynb", ".py")
    print(f"Notebook converted to: {script_path}")
    return script_path

def run_script(script_path):
    print(f"Running script: {script_path}")
    result = subprocess.run(
        ["python", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("----- Output -----\n", result.stdout)
    if result.stderr:
        print("----- Errors -----\n", result.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_notebook_as_script.py <notebook.ipynb>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    script_path = convert_notebook_to_script(notebook_path)
    run_script(script_path)