# ============================================================================
# BFSNet CPU Dockerfile
# ============================================================================
# Lightweight container for CPU-only training and inference.
#
# Build:
#   docker build -t bfsnet:cpu -f docker/Dockerfile .
#
# Run (foreground):
#   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/runs:/app/runs \
#       bfsnet:cpu python train_fmnist_bfs.py --epochs 10
#
# Run in Background (detached mode - survives terminal close/sleep):
#   docker run -d \
#       --name bfsnet_training \
#       -v $(pwd)/data:/app/data \
#       -v $(pwd)/runs:/app/runs \
#       -v $(pwd)/configs:/app/configs \
#       bfsnet:cpu python bfs_training_matrix.py \
#           --config configs/experiment-config.yaml \
#           --infer_script infer_fmnist_bfs.py
#
# Monitor Background Run:
#   docker logs -f bfsnet_training         # Follow live output
#   docker logs --tail 50 bfsnet_training  # Last 50 lines
#   docker ps                              # Check if still running
#
# Stop Background Run:
#   docker stop bfsnet_training            # Graceful stop
#   docker rm bfsnet_training              # Remove container
#   docker rm -f bfsnet_training           # Force stop and remove
#
# Interactive:
#   docker run --rm -it -v $(pwd):/app bfsnet:cpu bash
#
# Installed Python Packages:
#   - PyTorch 2.1.0 (CPU) - Deep learning framework
#   - torchvision - Vision utilities and datasets (FashionMNIST)
#   - tqdm - Progress bars for training matrix
#   - pyyaml - YAML config file support
#   - pandas - Data analysis for results
#   - matplotlib/seaborn - Visualization
#   - scikit-learn - ML utilities
#
# Changelog:
#   2025-12-15 - Added tqdm verification step, updated documentation for
#                detached mode running, updated labels for new features,
#                fixed directory creation order (create as root before user switch)
#   2025-09-18 - Initial CPU Dockerfile
# ============================================================================

# ----------------------------------------------------------------------------
# Build Arguments
# ----------------------------------------------------------------------------
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.1.0

# ----------------------------------------------------------------------------
# Base Image
# ----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# ----------------------------------------------------------------------------
# System Dependencies
# ----------------------------------------------------------------------------
FROM base AS system-deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    # Git for potential package installs
    git \
    # Image processing (for torchvision)
    libjpeg-dev \
    libpng-dev \
    # Scientific computing
    libopenblas-dev \
    liblapack-dev \
    # Utilities
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ----------------------------------------------------------------------------
# Python Dependencies
# ----------------------------------------------------------------------------
FROM system-deps AS python-deps

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch CPU version (smaller than CUDA version)
ARG PYTORCH_VERSION
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION}+cpu \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Verify PyTorch installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Copy and install project requirements
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages with updated versions
# CRITICAL: tqdm is REQUIRED for progress bars in bfs_training_matrix.py
RUN pip install --no-cache-dir \
    pyyaml>=6.0.0 \
    tqdm>=4.66.0 \
    pandas>=2.0.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    networkx>=3.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0

# Verify tqdm installation (required for progress bars in bfs_training_matrix.py)
RUN python -c "import tqdm; print(f'tqdm installed successfully, version: {tqdm.__version__}')"

# Optional: Install pytest-env for environment variable support in tests
RUN pip install --no-cache-dir pytest-env

# ----------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------
FROM python-deps AS app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash bfsnet

# Set working directory (still running as root at this point)
WORKDIR /app

# Create necessary directories AS ROOT (before switching to bfsnet user)
# This ensures we have permission to create all directories
RUN mkdir -p /app/data /app/runs /app/figures /app/configs

# Change ownership of the entire /app directory to bfsnet user
# This must happen BEFORE the COPY command and BEFORE switching users
RUN chown -R bfsnet:bfsnet /app

# Now switch to non-root user
USER bfsnet

# Copy application code (as bfsnet user, directories already exist with correct ownership)
COPY --chown=bfsnet:bfsnet . .

# ----------------------------------------------------------------------------
# Environment Configuration
# ----------------------------------------------------------------------------
# BFSNet settings
ENV BFSNET_DEVICE=cpu \
    BFSNET_DATA_ROOT=/app/data \
    BFSNET_RUNS_ROOT=/app/runs \
    BFSNET_CONFIG=/app/configs/experiment-config.yaml \
    BFSNET_LOG_LEVEL=INFO

# Performance settings for CPU
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4

# ----------------------------------------------------------------------------
# Health Check
# ----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import bfs_model; print('OK')" || exit 1

# ----------------------------------------------------------------------------
# Default Command
# ----------------------------------------------------------------------------
CMD ["python", "train_fmnist_bfs.py", "--help"]

# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------
LABEL maintainer="BFSNet Project" \
    version="1.4.0" \
    description="BFSNet CPU Training and Inference Container" \
    pytorch.version="2.1.0" \
    pytorch.device="cpu" \
    features="tqdm progress bars, YAML config, timing extraction, latency measurement" \
    org.opencontainers.image.source="https://github.com/bfsnet/bfsnet" \
    org.opencontainers.image.title="BFSNet CPU" \
    org.opencontainers.image.description="CPU-only container for BFSNet training and inference"