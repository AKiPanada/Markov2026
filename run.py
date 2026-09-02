import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV = ROOT / ".venv"

# Find Python files, excluding this launcher
python_files = sorted(
    p for p in ROOT.glob("*.py")
    if p.name != Path(__file__).name
)

if not python_files:
    print("No Python files found.")
    sys.exit(1)

print("Choose a file:")
for i, file in enumerate(python_files, 1):
    print(f"{i}. {file.name}")

while True:
    try:
        choice = int(input("\nChoose a file (number): "))
        if 1 <= choice <= len(python_files):
            python_file = python_files[choice - 1]
            break
    except ValueError:
        pass

    print("Invalid choice, try again.")

# Create virtual environment if it doesn't exist
if not VENV.exists():
    print("\nCreating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])

# Get the Python executable inside the venv.
if os.name == "nt":
    venv_python = VENV / "Scripts" / "python.exe"
else:
    venv_python = VENV / "bin" / "python"

# Install dependencies
print("\nInstalling requirements...")
subprocess.check_call([
    str(venv_python),
    "-m",
    "pip",
    "install",
    "-r",
    str(ROOT / "requirements.txt"),
])

# Run selected script
print(f"\n\nRUNNING [{python_file.name}]:\n")
subprocess.check_call([str(venv_python), str(python_file)])