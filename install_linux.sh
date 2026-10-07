#!/bin/bash
# OpenLPT One-Click Installer for Linux
# Author: Sinchy (Sinchy/pyOpenLPT)
#
# Usage: bash install_linux.sh [ENV_NAME]      (default: OpenLPT)

set -e # Exit on error

ENV_NAME="${1:-OpenLPT}"

echo "=========================================="
echo "      OpenLPT One-Click Installer (Linux)"
echo "=========================================="
echo ""
echo "This script will:"
echo "1. Check for system build dependencies"
echo "2. Install Miniforge3 (Conda + Mamba) if needed"
echo "3. Create a Conda environment named '$ENV_NAME'"
echo "4. Install all dependencies and the OpenLPT package"
echo ""

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# --- STEP 0: System Dependencies Check ---
echo "[0/4] Checking for system build tools..."
MISSING_TOOLS=()
if ! command -v gcc &> /dev/null; then MISSING_TOOLS+=("gcc"); fi
if ! command -v g++ &> /dev/null; then MISSING_TOOLS+=("g++"); fi
if ! command -v cmake &> /dev/null; then MISSING_TOOLS+=("cmake"); fi

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    echo ""
    echo "[!] Warning: Missing build tools: ${MISSING_TOOLS[*]}"
    echo "    To install them on common Linux distributions:"
    echo "    Ubuntu/Debian: sudo apt update && sudo apt install build-essential cmake libomp-dev"
    echo "    Fedora/CentOS: sudo dnf groupinstall \"Development Tools\" && sudo dnf install cmake libomp-devel"
    echo "    HPC Cluster:   module load gcc cmake"
    echo ""
    read -p "Press ENTER to continue anyway (Conda might provide some tools), or Ctrl+C to cancel..."
else
    echo "[OK] Found basic build tools."
fi

# --- STEP 1: Check/Install Conda ---
echo ""
echo "[1/4] Checking for Conda/Mamba..."
if ! command -v conda &> /dev/null; then
    echo "[INFO] Conda not found. Automatically downloading Miniforge3..."
    
    ARCH=$(uname -m)
    INSTALLER="Miniforge3-Linux-${ARCH}.sh"
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}"
    
    echo "Downloading ${URL}..."
    curl -L -O "$URL"
    
    echo "Installing Miniforge3 to $HOME/miniforge3..."
    bash "$INSTALLER" -b -p "$HOME/miniforge3"
    rm "$INSTALLER"
    
    echo "Initializing Conda..."
    "$HOME/miniforge3/bin/conda" init bash
    
    # Activate for the current script
    source "$HOME/miniforge3/bin/activate"
    echo "[SUCCESS] Miniforge3 installed!"
else
    echo "[OK] Conda found."
fi

# --- STEP 2: Create the Conda Environment with its Dependencies ---
echo ""
echo "[2/4] Creating Conda Environment '$ENV_NAME' with the dependencies from conda-forge..."

# Proper way to activate conda in a script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Using mamba if available (installed by Miniforge), otherwise conda
if command -v mamba &> /dev/null; then CONDA_CMD=mamba; else CONDA_CMD=conda; fi

# requirements.txt without its pip-only lines (not on conda-forge; step 3), plus qt6-webengine (the camera
# calibration's Report tab)
CONDA_REQ="$(mktemp)"
grep -v "pip-only" requirements.txt > "$CONDA_REQ"
echo "qt6-webengine" >> "$CONDA_REQ"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[INFO] Environment '$ENV_NAME' exists: updating its packages."
    $CONDA_CMD install -n "$ENV_NAME" --override-channels -c conda-forge --file "$CONDA_REQ" -y
else
    $CONDA_CMD create -n "$ENV_NAME" --override-channels -c conda-forge python=3.11 --file "$CONDA_REQ" -y
fi
rm -f "$CONDA_REQ"

# --- STEP 3: Activate; pip-only Dependencies ---
echo ""
echo "[3/4] Activating Environment and Installing the pip-only Dependencies..."
conda activate "$ENV_NAME"
# requirements.txt's lines marked pip-only (pycine)
for spec in $(grep "pip-only" requirements.txt | grep -v "^#" | awk '{print $1}'); do
    pip install "$spec"
done

# --- STEP 4: Build and Install OpenLPT ---
echo ""
echo "[4/4] Building and Installing OpenLPT..."
echo "This will compile the C++ core. This may take a minute..."

# Clean previous build artifacts
rm -rf build/
rm -rf *.egg-info

# Install the package with GUI extras by default
pip install ".[gui]" --no-build-isolation

echo ""
echo "=========================================="
echo "      Installation Complete!"
echo "=========================================="
echo ""
echo "To use OpenLPT:"
echo "  1. Run: conda activate $ENV_NAME"
echo "  2. Run: python GUI.py"
echo ""
echo "Note: If you are on a remote server/HPC without a GUI,"
echo "      you can use the Python API (import pyopenlpt)."
echo ""

# Ask to launch GUI (only if DISPLAY is set)
if [ -n "$DISPLAY" ]; then
    read -p "Launch OpenLPT GUI now? (y/n): " launch
    if [[ "$launch" == "y" || "$launch" == "Y" ]]; then
        python GUI.py
    fi
else
    echo "[INFO] No display detected (Typical for HPC/Servers). Skipping GUI launch."
fi
