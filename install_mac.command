#!/bin/bash
# Usage: bash install_mac.command [ENV_NAME]   (default: OpenLPT; OPENLPT_NO_LAUNCH=1 skips the GUI launch at the end)
ENV_NAME="${1:-OpenLPT}"
echo "=========================================="
echo "      OpenLPT One-Click Installer"
echo "=========================================="
echo ""
echo "This script will:"
echo "1. Create a Conda environment named '$ENV_NAME'"
echo "2. Install all dependencies"
echo "3. Install the OpenLPT package"
echo ""
read -p "Press enter to continue..."

# --- Xcode Command Line Tools Check ---
echo ""
echo "[0.5/4] Checking for Xcode Command Line Tools..."
if ! xcode-select -p &> /dev/null; then
    echo ""
    echo "[ERROR] Xcode Command Line Tools not found!"
    echo "        These are REQUIRED for compiling C++ extensions on macOS."
    echo ""
    echo "Please run the following command in your terminal:"
    echo "    xcode-select --install"
    echo ""
    echo "Follow the pop-up prompts to install, then RUN THIS SCRIPT AGAIN."
    echo "================================================================"
    read -p "Press Enter to exit..."
    exit 1
else
    echo "[OK] Xcode Command Line Tools found."
fi

# --- libomp (OpenMP) Check ---
echo ""
echo "[0.7/4] Checking for libomp (OpenMP Support)..."
if ! command -v brew &> /dev/null; then
    echo "[INFO] Homebrew not found. We will rely on Conda for OpenMP."
elif ! brew list libomp &> /dev/null; then
    echo "[INFO] libomp not found via brew. Recommending: brew install libomp"
    # Optional: brew install libomp
fi
# --------------------------------------

cd "$(dirname "$0")"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo ""
    echo "[!] Conda/Mamba not found."
    echo "[!] Should I automatically download and install Miniforge3 (Minimal Conda + Mamba)?"
    echo "    Target: $HOME/miniforge3"
    echo ""
    read -p "Press ENTER to install Miniforge3 (or Ctrl+C to cancel)..."
    
    echo ""
    echo "[0/3] Installing Miniforge3..."
    ARCH=$(uname -m)
    INSTALLER="Miniforge3-MacOSX-${ARCH}.sh"
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}"
    
    echo "Downloading ${URL}..."
    curl -L -O "$URL"
    
    echo "Installing to $HOME/miniforge3..."
    bash "$INSTALLER" -b -p "$HOME/miniforge3"
    rm "$INSTALLER"
    
    # Initialize for future shells
    "$HOME/miniforge3/bin/conda" init zsh
    "$HOME/miniforge3/bin/conda" init bash
    
    # Activate for current script
    source "$HOME/miniforge3/bin/activate"
    
    echo "Miniforge3 installed successfully!"
fi

echo ""
echo "[1/3] Creating Conda Environment '$ENV_NAME' with the dependencies from conda-forge..."
# Try to find conda hook
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
# mamba if available (installed by Miniforge), otherwise conda
if command -v mamba &> /dev/null; then CONDA_CMD=mamba; else CONDA_CMD=conda; fi
# requirements.txt without its pip-only lines (not on conda-forge; step 3), plus qt6-webengine (the camera
# calibration's Report tab) on Apple silicon. conda-forge has none for Intel Macs, where the Report tab opens the
# page in the browser instead.
CONDA_REQ="$(mktemp)"
grep -v "pip-only" requirements.txt > "$CONDA_REQ"
if [ "$(uname -m)" = "arm64" ]; then
    echo "qt6-webengine" >> "$CONDA_REQ"
fi
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[INFO] Environment '$ENV_NAME' exists: updating its packages."
    $CONDA_CMD install -n "$ENV_NAME" --override-channels -c conda-forge --file "$CONDA_REQ" -y
else
    $CONDA_CMD create -n "$ENV_NAME" --override-channels -c conda-forge python=3.11 --file "$CONDA_REQ" -y
fi
if [ $? -ne 0 ]; then
    echo "[Error] Installing the conda-forge packages failed."
    rm -f "$CONDA_REQ"
    read -p "Press enter to exit"
    exit 1
fi
rm -f "$CONDA_REQ"

echo ""
echo "[2/3] Activating Environment..."
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    conda activate "$ENV_NAME"
else
    echo "[Warning] Could not find conda.sh. Assuming environment is correct or manual activation needed."
fi

echo ""
echo "[3/3] Installing the pip-only Dependencies..."
# requirements.txt's lines marked pip-only (pycine)
for spec in $(grep "pip-only" requirements.txt | grep -v "^#" | awk '{print $1}'); do
    pip install "$spec"
    if [ $? -ne 0 ]; then
        echo "[Error] pip install $spec failed."
        read -p "Press enter to exit"
        exit 1
    fi
done

echo ""
echo "[3.5/4] Configuring Compiler Flags for macOS (OpenMP)..."
# 1. Force a clean build to ensure CMake picks up new flags
rm -rf build/
rm -rf _skbuild/  # just in case

# 2. Set environment variables that CMake respects automatically
# If brew libomp exists, it's usually more reliable
if command -v brew &> /dev/null && [ -d "$(brew --prefix libomp)" ]; then
    LIBOMP_PATH=$(brew --prefix libomp)
    export OpenMP_ROOT="$LIBOMP_PATH"
    export LDFLAGS="-L$LIBOMP_PATH/lib -Wl,-rpath,$LIBOMP_PATH/lib $LDFLAGS"
    export CPPFLAGS="-I$LIBOMP_PATH/include $CPPFLAGS"
    export CXXFLAGS="-I$LIBOMP_PATH/include $CXXFLAGS"
else
    export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib $LDFLAGS"
    export CPPFLAGS="-I$CONDA_PREFIX/include $CPPFLAGS"
    export CXXFLAGS="-I$CONDA_PREFIX/include $CXXFLAGS"
fi

echo "Environment configured for OpenMP."

echo ""
echo "[4/4] Installing OpenLPT..."
pip install . --no-build-isolation
if [ $? -ne 0 ]; then
    echo "[Error] Pip install failed."
    read -p "Press enter to exit"
    exit 1
fi

echo ""
echo "=========================================="
echo "      Installation Complete!"
echo "=========================================="
if [ "$OPENLPT_NO_LAUNCH" != "1" ]; then
    echo "Launching OpenLPT GUI..."
    python GUI.py
fi
