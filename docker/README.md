# BFSNet & BoeNet Docker Setup

**Repository**: BoeNet (formerly BFSNet)  
**Purpose**: Docker containerization for vision (BFSNet - COMPLETE) and language (BoeNet - ACTIVE) tasks  
**Version**: v2.0.1  
**Last Updated**: December 30, 2025

---

## 📋 Overview

This directory contains Docker configurations for both:
1. **BFSNet (Vision)** - ✅ COMPLETE - FashionMNIST training and inference
2. **BoeNet (Language)** - ✅ ACTIVE - WikiText-2 training with ByteTokenizer

### Current Training Status

| Project | Status | Dataset | Best Result |
|---------|--------|---------|-------------|
| BFSNet v2.0.0 | ✅ Complete | FashionMNIST | 87.42% accuracy |
| BoeNet v2.0.1 | ✅ Training | WikiText-2 | PPL 11.55 (22x improvement) |

### Key Differences: Vision vs. Language Docker

| Aspect | BFSNet (Vision) | BoeNet (Language) |
|--------|-----------------|-------------------|
| **Base Dataset** | FashionMNIST (images) | WikiText-2 (text via HuggingFace) |
| **Dataset Size** | ~30MB | ~10MB (cached) |
| **Tokenizer** | N/A | ByteTokenizer (UTF-8 bytes) |
| **vocab_size** | 10 (classes) | 256 (bytes) |
| **Dependencies** | torchvision | datasets, transformers |
| **Storage** | /data (images) | /data (HuggingFace cache) |

---

## 🚀 Quick Start

### Verify GPU Access First

```bash
# Test NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

**Expected Output (RTX 5080 example)**:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 591.59         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 5080 ...    On  |   00000000:02:00.0 Off |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### Build BoeNet CUDA Image

```bash
# Build image
docker build -t boenet:cuda -f docker/Dockerfile.cuda .
```

### Run Training (Background - Recommended)

```bash
# Linux/macOS
docker run -d --gpus all \
    --name boenet_sweep \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    -v $(pwd)/configs:/app/configs \
    -v $(pwd)/boenet:/app/boenet \
    boenet:cuda python boenet_training_matrix.py \
        --config configs/experiment-config.yaml

# Windows PowerShell
docker run -d --gpus all --name boenet_sweep -v ${PWD}/data:/app/data -v ${PWD}/runs:/app/runs -v ${PWD}/configs:/app/configs -v ${PWD}/boenet:/app/boenet boenet:cuda python boenet_training_matrix.py --config configs/experiment-config.yaml
```

### Monitor Training

```bash
# Follow live logs (Ctrl+C to detach without stopping)
docker logs -f boenet_sweep

# Last 50 lines
docker logs --tail 50 boenet_sweep

# Check if running
docker ps

# Stop gracefully
docker stop boenet_sweep

# Remove container
docker rm boenet_sweep

# Force stop and remove
docker rm -f boenet_sweep
```

---

## ⚠️ CRITICAL: GPU Flag Required

**The `--gpus all` flag is REQUIRED for GPU training!**

Without it, training runs on CPU only (10-20x slower).

| Command | GPU Used | Speed |
|---------|----------|-------|
| `docker run --rm ...` | ❌ CPU only | ~7 sec/epoch |
| `docker run --rm --gpus all ...` | ✅ RTX 5080 | ~0.5 sec/epoch |

### Common Mistake

```bash
# WRONG - Missing --gpus all (runs on CPU!)
docker run --rm -v ${PWD}/data:/app/data ... boenet:cuda python boenet_training_matrix.py

# CORRECT - With --gpus all
docker run --rm --gpus all -v ${PWD}/data:/app/data ... boenet:cuda python boenet_training_matrix.py
```

---

## 📁 Directory Structure

```
docker/
├── README.md                    # This file (v2.0.1)
├── Dockerfile.cuda              # BoeNet CUDA image (v1.0.3)
├── Dockerfile                   # BFSNet CPU image (historical)
├── docker-compose.yaml          # Orchestration for all services
└── docker-config.yaml           # Environment and device settings
```

---

## 🐳 Docker Images

### BoeNet Image (Language - Active)

| Image | Base | Purpose | Size | PyTorch | Status |
|-------|------|---------|------|---------|--------|
| `boenet:cuda` | `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` | GPU training/inference | ~10 GB | 2.7.1 cu128 | ✅ ACTIVE |

**Key Features**:
- CUDA 12.8.0 with cuDNN (Blackwell RTX 50 series support)
- PyTorch 2.7.1 with cu128 wheels
- HuggingFace datasets for WikiText-2
- ByteTokenizer (UTF-8 byte-level encoding)
- HF_HOME configured for cache persistence

### BFSNet Images (Vision - Historical)

| Image | Base | Purpose | Size | PyTorch | Status |
|-------|------|---------|------|---------|--------|
| `bfsnet:cpu` | `python:3.10-slim-bookworm` | CPU training/inference | ~2.5 GB | 2.1.0 CPU | ✅ Complete |
| `bfsnet:cuda` | `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` | GPU training/inference | ~8-10 GB | 2.7.1 cu128 | ✅ Complete |

---

## 🔧 Dockerfile.cuda Specification (v1.0.3)

**Location**: `docker/Dockerfile.cuda`

```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA 12.8 support
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/data /app/runs

# CRITICAL: Set HuggingFace cache directory for persistence
ENV HF_HOME=/app/data/.cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TOKENIZERS_PARALLELISM=false

CMD ["python", "boenet_training_matrix.py"]
```

**Key Configuration (v1.0.3)**:
- ✅ `HF_HOME=/app/data/.cache` - HuggingFace cache persists across runs
- ✅ CUDA 12.8.0 base - Supports Blackwell (RTX 5080/5090)
- ✅ PyTorch 2.7.1 cu128 - Latest stable with CUDA 12.8
- ✅ `TOKENIZERS_PARALLELISM=false` - Avoids tokenizer warnings

---

## 📦 Volume Mounts & Data Persistence

### BoeNet (Language) Volume Strategy

| Container Path | Host Path | Purpose | Persistence |
|----------------|-----------|---------|-------------|
| `/app/data` | `./data` | HuggingFace cache, WikiText-2 | ✅ **CRITICAL** |
| `/app/runs` | `./runs` | Training outputs, checkpoints, CSV | ✅ **CRITICAL** |
| `/app/configs` | `./configs` | Configuration YAML files | ✅ Read-only |
| `/app/boenet` | `./boenet` | Source code (for development) | ✅ Optional |

**Data Directory Structure**:
```
data/
├── .cache/                     # HuggingFace cache (HF_HOME)
│   └── huggingface/
│       └── datasets/
│           └── wikitext/       # WikiText-2 dataset
└── (auto-created by HuggingFace)

runs/
└── YYYYMMDD_HHMMSS/           # Training run directory
    ├── matrix_results.csv      # All experiment results
    ├── config_name_1/
    │   ├── config_name_1.pt    # Model checkpoint
    │   └── training.log        # Training log
    └── config_name_2/
        └── ...
```

### BFSNet (Vision) Volume Strategy

| Container Path | Host Path | Purpose | Persistence |
|----------------|-----------|---------|-------------|
| `/app/data` | `./data` | FashionMNIST (auto-downloaded) | ✅ Persist |
| `/app/runs` | `./runs` | Training outputs, checkpoints | ✅ **CRITICAL** |
| `/app/configs` | `./configs` | Configuration YAML files | ✅ Read-only |

---

## 🚀 Detached Mode (Recommended for Long Runs)

Detached mode (`-d` flag) allows training to continue even if:
- Your terminal closes
- Your SSH session disconnects
- Your computer goes to sleep

### BoeNet Training Matrix (400 cells)

```bash
# Start in background
docker run -d --gpus all \
    --name boenet_sweep \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    -v $(pwd)/configs:/app/configs \
    -v $(pwd)/boenet:/app/boenet \
    boenet:cuda python boenet_training_matrix.py \
        --config configs/experiment-config.yaml

# Windows PowerShell (single line)
docker run -d --gpus all --name boenet_sweep -v ${PWD}/data:/app/data -v ${PWD}/runs:/app/runs -v ${PWD}/configs:/app/configs -v ${PWD}/boenet:/app/boenet boenet:cuda python boenet_training_matrix.py --config configs/experiment-config.yaml
```

### Key Flags Explained

| Flag | Purpose |
|------|---------|
| `-d` | Detached mode (runs in background) |
| `--gpus all` | **REQUIRED** - Enables GPU access |
| `--name boenet_sweep` | Names container for easy reference |
| `-v $(pwd)/data:/app/data` | Mount data directory |
| `-v $(pwd)/runs:/app/runs` | Mount runs directory |

**Note**: We removed `--rm` so the container persists after completion (for log access).

### Monitoring Commands

```bash
# Follow live output
docker logs -f boenet_sweep

# View last 100 lines, then follow
docker logs --tail 100 -f boenet_sweep

# Check container status
docker ps                    # Running containers
docker ps -a                 # All containers (including stopped)

# Get container resource usage
docker stats boenet_sweep

# Inspect container details
docker inspect boenet_sweep
```

### Cleanup Commands

```bash
# Stop gracefully (waits for current epoch to finish)
docker stop boenet_sweep

# Force stop immediately
docker kill boenet_sweep

# Remove stopped container
docker rm boenet_sweep

# Force stop and remove in one command
docker rm -f boenet_sweep

# Remove all stopped containers
docker container prune
```

---

## 🎛️ Environment Variables

### BoeNet Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `/app/data/.cache` | HuggingFace cache directory (CRITICAL for persistence) |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device selection |
| `TOKENIZERS_PARALLELISM` | `false` | Disable tokenizer parallelism warnings |
| `PYTHONUNBUFFERED` | `1` | Unbuffered Python output |

### Setting Custom Environment Variables

```bash
# Use specific GPU
docker run -d --gpus all \
    -e CUDA_VISIBLE_DEVICES=1 \
    --name boenet_sweep \
    ...

# Override HuggingFace cache location
docker run -d --gpus all \
    -e HF_HOME=/app/data/hf_cache \
    --name boenet_sweep \
    ...
```

---

## 🖥️ GPU Configuration

### Supported GPU Architectures

| Architecture | GPUs | Compute Capability | Status |
|--------------|------|-------------------|--------|
| Pascal | GTX 10 series | sm_60, sm_61 | ⚪ Not tested |
| Volta | Titan V, Quadro GV100 | sm_70 | ⚪ Not tested |
| Turing | RTX 20 series | sm_75 | ⚪ Not tested |
| Ampere | RTX 30 series, A100 | sm_80, sm_86 | ✅ Tested |
| Ada Lovelace | RTX 40 series | sm_89 | ⚪ Not tested |
| Hopper | H100 | sm_90 | ⚪ Not tested |
| **Blackwell** | **RTX 50 series** | **sm_120** | ✅ **Tested (RTX 5080)** |

### RTX 5080 (Blackwell) Requirements

- NVIDIA driver 590.x or newer
- CUDA 12.8+ toolkit (in container)
- PyTorch 2.7+ with cu128 wheels
- Docker with NVIDIA Container Toolkit

### Verify GPU Support

```bash
# Check host driver and CUDA version
nvidia-smi

# Verify Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Check PyTorch CUDA in container
docker run --rm --gpus all boenet:cuda python -c \
    "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Check CUDA architecture support
docker run --rm --gpus all boenet:cuda python -c \
    "import torch; print(f'CUDA arch list: {torch.cuda.get_arch_list()}')"
```

### Windows Docker Desktop + WSL2

For Windows users with NVIDIA GPU:

1. **Ensure WSL2 backend** is enabled in Docker Desktop
2. **Install NVIDIA drivers for WSL2** (not regular Windows drivers)
   - Download from: https://developer.nvidia.com/cuda/wsl
3. **Enable GPU in Docker Desktop**:
   - Settings → Resources → WSL Integration → Enable for your distro
4. **Test GPU access**:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
   ```

### Multi-GPU Training

```bash
# Use specific GPUs
docker run -d --gpus '"device=0,1"' \
    --name boenet_multigpu \
    ...

# Use all GPUs
docker run -d --gpus all \
    --name boenet_allgpu \
    ...
```

---

## 🐛 Troubleshooting

### Issue 1: GPU Not Detected in Container

**Symptom**:
```
WARNING: The NVIDIA Driver was not detected. GPU functionality will not be available.
```

**Cause**: Missing `--gpus all` flag in docker run command.

**Solution**:
```bash
# Add --gpus all to your command
docker run --rm --gpus all ...
```

### Issue 2: Container Name Already in Use

**Symptom**:
```
docker: Error response from daemon: Conflict. The container name "/boenet_sweep" is already in use
```

**Solution**:
```bash
# Remove the old container
docker rm -f boenet_sweep

# Then run again
docker run -d --gpus all --name boenet_sweep ...
```

### Issue 3: Permission Denied on Volumes

**Symptom**:
```
PermissionError: [Errno 13] Permission denied: '/app/data/.cache'
```

**Solution**:
```bash
# Linux: Fix ownership
sudo chown -R $USER:$USER data runs configs

# Or run container as current user
docker run --rm --user $(id -u):$(id -g) ...
```

### Issue 4: IndexError: index out of range in self

**Symptom**:
```
IndexError: index out of range in self
File "/app/boenet/model.py", line 744, in forward
    embedded = self.embedding(x)
```

**Cause**: Token IDs exceed vocab_size=256. This was caused by CharTokenizer using `ord()` which returns Unicode code points > 255.

**Solution**: Update to data_utils.py v2.0.1 with ByteTokenizer:
```python
# OLD (broken)
def encode(self, text: str) -> List[int]:
    return [ord(c) for c in text]  # Can return values > 255

# NEW (fixed in v2.0.1)
def encode(self, text: str) -> List[int]:
    return list(text.encode('utf-8'))  # Always 0-255
```

### Issue 5: HuggingFace Dataset Re-downloading

**Symptom**: WikiText-2 downloads every time container starts.

**Cause**: HF_HOME not set or not pointing to mounted volume.

**Solution**: Ensure Dockerfile has:
```dockerfile
ENV HF_HOME=/app/data/.cache
```

And mount the data volume:
```bash
-v $(pwd)/data:/app/data
```

### Issue 6: Out of Memory (OOM)

**Symptom**:
```
CUDA out of memory. Tried to allocate X MiB
```

**Solution**:
```bash
# Reduce batch size via config or command line
docker run -d --gpus all \
    --name boenet_sweep \
    ...
    boenet:cuda python boenet_training_matrix.py \
        --config configs/experiment-config.yaml \
        --batch_size 32
```

### Issue 7: Blackwell GPU (RTX 50 series) Not Working

**Symptom**: PyTorch doesn't recognize sm_120 architecture.

**Solution**:
1. Ensure NVIDIA driver 590.x or newer
2. Rebuild image with latest Dockerfile.cuda (v1.0.3)
3. Verify PyTorch version:
```bash
docker run --rm --gpus all boenet:cuda python -c \
    "import torch; print(torch.__version__)"
# Should be 2.7.1 or later
```

---

## 📈 Performance Optimization

### 1. Use Shared Memory for DataLoader

```bash
# Increase shared memory (default is often too small)
docker run -d --gpus all --shm-size=2g \
    --name boenet_sweep \
    ...
```

### 2. Pin CPU Cores for Consistent Performance

```bash
# Use specific CPU cores
docker run -d --gpus all --cpuset-cpus="0-7" \
    --name boenet_sweep \
    ...
```

### 3. Limit Memory Usage

```bash
# Limit container memory
docker run -d --gpus all --memory=32g \
    --name boenet_sweep \
    ...
```

### 4. Performance Comparison

| Configuration | Epoch Time | Notes |
|---------------|------------|-------|
| CPU only (no `--gpus all`) | ~7 sec | 10-20x slower |
| RTX 5080 with `--gpus all` | ~0.5 sec | Full GPU utilization |
| RTX 3090 with `--gpus all` | ~0.6 sec | Previous gen still fast |

---

## 📊 Training Matrix Results

### Current Progress (December 2025)

The 400-cell training matrix explores:
- **Epochs**: 5, 10, 15, 20
- **seq_len**: 64, 128
- **embed_dim**: 32, 64
- **max_children (K)**: 0 (dense), 3 (BFS)
- **threshold**: 0.3, 0.35, 0.4, 0.42, 0.5
- **lambda_efficiency**: 0.0, 0.01, 0.05

### Best Results So Far

| Configuration | Val PPL | Train PPL | Notes |
|---------------|---------|-----------|-------|
| K=0, sl=128, ed=64, ep=20 | **11.55** | 11.57 | Dense baseline |
| K=0, sl=128, ed=64, ep=15 | 11.55 | 11.57 | Same as 20 epochs |
| K=0, sl=128, ed=32, ep=20 | 11.55 | 11.57 | Smaller embed works |
| K=0, sl=64, ed=64, ep=20 | 11.55 | 11.56 | Shorter seq OK |

### Token Range Validation

```
[sanity:wikitext2] batch=64x64 dtype=torch.int64 
input_ids[min,max]=[10,226] labels[min,max]=[10,226] 
vocab_size=256 [OK]
```

✅ All tokens within [0, 255] after ByteTokenizer fix.

---

## 🐳 Docker Compose (Optional)

### docker-compose.yaml

```yaml
version: '3.8'

services:
  boenet-gpu:
    build:
      context: ..
      dockerfile: docker/Dockerfile.cuda
    image: boenet:cuda
    container_name: boenet_sweep
    volumes:
      - ../data:/app/data
      - ../runs:/app/runs
      - ../configs:/app/configs
      - ../boenet:/app/boenet
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: python boenet_training_matrix.py --config configs/experiment-config.yaml
```

### Using Docker Compose

```bash
# Build
docker-compose -f docker/docker-compose.yaml build

# Run (detached)
docker-compose -f docker/docker-compose.yaml up -d boenet-gpu

# View logs
docker-compose -f docker/docker-compose.yaml logs -f boenet-gpu

# Stop
docker-compose -f docker/docker-compose.yaml down
```

---

## 📝 Installed Python Packages

### BoeNet (Language) Packages

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.7.1 (cu128) | Deep learning framework |
| datasets | >=2.16.0 | HuggingFace dataset loading |
| transformers | >=4.36.0 | Tokenizers and models |
| tqdm | >=4.66.0 | Progress bars |
| pandas | >=2.0.0 | Data analysis |
| pyyaml | >=6.0.0 | YAML config files |
| matplotlib | >=3.7.0 | Visualization |

---

## 🗺️ Changelog

### v2.0.1 (December 30, 2025)
- ✅ Added `--gpus all` requirement documentation
- ✅ Added RTX 5080 (Blackwell) verification
- ✅ Added detached mode (`-d`) documentation
- ✅ Updated troubleshooting for ByteTokenizer fix
- ✅ Added training matrix results section
- ✅ Added Windows PowerShell commands

### v1.0.3 (December 29, 2025)
- ✅ Added `HF_HOME=/app/data/.cache` for HuggingFace cache persistence
- ✅ Fixed dataset re-download issue

### v1.0.2 (December 28, 2025)
- ✅ Updated to CUDA 12.8.0 for Blackwell support
- ✅ Updated to PyTorch 2.7.1 cu128

### v1.0.1 (December 27, 2025)
- ✅ Initial BoeNet Docker configuration
- ✅ Volume mount strategy

---

## 📞 Support

### BoeNet (Language)
- 📚 See main `README.md` for architecture details
- 🔧 See `boenet/utils/data_utils.py` v2.0.1 for ByteTokenizer
- 🚧 Project status: ✅ TRAINING (400-cell matrix)

### BFSNet (Vision)
- 📚 See `docs/bfsnet_architecture.md` for architecture details
- ✅ Project status: COMPLETE

### General
- 📧 Contact: [your-email@example.com]
- 📝 Issues: Contact project owner (closed source)

---

**Last Updated**: December 30, 2025  
**Docker Version**: v2.0.1  
**Status**: BoeNet ✅ TRAINING with GPU | BFSNet ✅ COMPLETE  
**Tested On**: RTX 5080 (Blackwell), RTX 3090 (Ampere)

**⚠️ Proprietary Software**: This project is closed source. All rights reserved.

---

## Quick Reference

### Start Training (Copy-Paste Ready)

**Linux/macOS**:
```bash
docker run -d --gpus all --name boenet_sweep -v $(pwd)/data:/app/data -v $(pwd)/runs:/app/runs -v $(pwd)/configs:/app/configs -v $(pwd)/boenet:/app/boenet boenet:cuda python boenet_training_matrix.py --config configs/experiment-config.yaml
```

**Windows PowerShell**:
```powershell
docker run -d --gpus all --name boenet_sweep -v ${PWD}/data:/app/data -v ${PWD}/runs:/app/runs -v ${PWD}/configs:/app/configs -v ${PWD}/boenet:/app/boenet boenet:cuda python boenet_training_matrix.py --config configs/experiment-config.yaml
```

### Monitor:
```bash
docker logs -f boenet_sweep
```

### Stop:
```bash
docker rm -f boenet_sweep
```