# BFSNet & BoeNet Docker Setup

**Repository**: BoeNet (formerly BFSNet)  
**Purpose**: Docker containerization for vision (BFSNet - COMPLETE) and language (BoeNet - IN PROGRESS) tasks  
**Last Updated**: December 20, 2025

---

## 📋 Overview

This directory contains Docker configurations for both:
1. **BFSNet (Vision)** - ✅ COMPLETE - FashionMNIST training and inference
2. **BoeNet (Language)** - 🚧 IN PROGRESS - Character/word-level language modeling

### Key Differences: Vision vs. Language Docker

| Aspect | BFSNet (Vision) | BoeNet (Language) |
|--------|-----------------|-------------------|
| **Base Dataset** | FashionMNIST (images) | Text files (Shakespeare, TinyStories) |
| **Dataset Size** | ~30MB | 1MB-2GB+ |
| **Mounting** | torchvision auto-download | Manual text file mounting |
| **Preprocessing** | Image normalization | Tokenization (on-the-fly) |
| **Dependencies** | torchvision | tokenizers, transformers, datasets |
| **Storage** | /data (images) | /data (text files + tokenizer caches) |

---

## 🚀 Quick Start

### BFSNet (Vision - Historical Reference)

#### CPU-Only
```bash
# Build CPU image
docker build -t bfsnet:cpu -f docker/Dockerfile .

# Run training
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    bfsnet:cpu python train_fmnist_bfs.py --epochs 10
```

#### GPU (CUDA)
```bash
# Build CUDA image
docker build -t bfsnet:cuda -f docker/Dockerfile.cuda .

# Run with GPU
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    bfsnet:cuda python train_fmnist_bfs.py --epochs 10
```

### BoeNet (Language - Active Development)

#### CPU-Only
```bash
# Build BoeNet CPU image
docker build -t boenet:cpu -f docker/Dockerfile.boenet .

# Run character-level training
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    -v $(pwd)/configs:/app/configs \
    boenet:cpu python train_char_boenet.py \
        --config configs/boenet/char-level-test.yaml
```

#### GPU (CUDA)
```bash
# Build BoeNet CUDA image
docker build -t boenet:cuda -f docker/Dockerfile.boenet.cuda .

# Run with GPU
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    -v $(pwd)/configs:/app/configs \
    boenet:cuda python train_char_boenet.py \
        --config configs/boenet/char-level-test.yaml
```

---

## 📁 Directory Structure
```
docker/
├── README.md                    # This file
│
├── Dockerfile                   # BFSNet CPU image (vision)
├── Dockerfile.cuda              # BFSNet CUDA image (vision)
│
├── Dockerfile.boenet            # BoeNet CPU image (language) - NEW
├── Dockerfile.boenet.cuda       # BoeNet CUDA image (language) - NEW
│
├── docker-compose.yaml          # Orchestration for all services
├── docker-config.yaml           # Environment and device settings
└── .dockerignore                # Build context exclusions
```

---

## 🐳 Docker Images

### BFSNet Images (Vision - COMPLETE)

| Image | Base | Purpose | Size | PyTorch | Status |
|-------|------|---------|------|---------|--------|
| `bfsnet:cpu` | `python:3.10-slim-bookworm` | CPU training/inference | ~2.5 GB | 2.1.0 CPU | ✅ COMPLETE |
| `bfsnet:cuda` | `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` | GPU training/inference | ~8-10 GB | 2.7.1 cu128 | ✅ COMPLETE |

**Key Features**:
- FashionMNIST auto-download via torchvision
- Image preprocessing pipelines
- Training matrix support
- Inference with latency measurement

### BoeNet Images (Language - IN PROGRESS)

| Image | Base | Purpose | Size | PyTorch | Status |
|-------|------|---------|------|---------|--------|
| `boenet:cpu` | `python:3.11-slim-bookworm` | CPU training/inference | ~3 GB | 2.7.1 CPU | 🚧 IN PROGRESS |
| `boenet:cuda` | `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` | GPU training/inference | ~10 GB | 2.7.1 cu128 | 🚧 IN PROGRESS |

**Key Features**:
- Text dataset mounting (Shakespeare, TinyStories, etc.)
- Tokenizer caching (BPE, character-level)
- Sequence processing pipelines
- Text generation support

**Additional Dependencies**:
```
tokenizers>=0.15.0
transformers>=4.36.0
datasets>=2.16.0
sentencepiece>=0.1.99
```

---

## 🔧 Dockerfile Specifications

### BFSNet: Dockerfile (CPU) - HISTORICAL

**Location**: `docker/Dockerfile`
```dockerfile
FROM python:3.10-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 bfsnet && chown -R bfsnet:bfsnet /app
USER bfsnet

# Set environment variables
ENV PYTHONUNBUFFERED=1

CMD ["python", "train_fmnist_bfs.py"]
```

**Key Characteristics**:
- Python 3.10 (stable for vision tasks)
- PyTorch 2.1.0 CPU-only
- torchvision for FashionMNIST
- tqdm for progress bars
- Non-root user for security

### BFSNet: Dockerfile.cuda (GPU) - HISTORICAL

**Location**: `docker/Dockerfile.cuda`
```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA 12.8 support
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 bfsnet && chown -R bfsnet:bfsnet /app
USER bfsnet

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "train_fmnist_bfs.py"]
```

**Key Characteristics**:
- CUDA 12.8.0 (supports Blackwell RTX 50 series)
- Python 3.11
- PyTorch 2.7.1 with cu128 wheels
- Supports sm_120 (RTX 5080/5090)

### BoeNet: Dockerfile.boenet (CPU) - NEW

**Location**: `docker/Dockerfile.boenet`
```dockerfile
FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch CPU
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install language modeling dependencies
RUN pip install --no-cache-dir \
    tokenizers>=0.15.0 \
    transformers>=4.36.0 \
    datasets>=2.16.0 \
    sentencepiece>=0.1.99

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for text data
RUN mkdir -p /app/data/text /app/data/tokenizers

# Create non-root user
RUN useradd -m -u 1000 boenet && chown -R boenet:boenet /app
USER boenet

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

CMD ["python", "train_char_boenet.py"]
```

**Key Differences from BFSNet**:
- ✅ Python 3.11 (latest stable)
- ✅ tokenizers, transformers, datasets libraries
- ✅ sentencepiece for BPE tokenization
- ✅ /data/text and /data/tokenizers directories
- ✅ TOKENIZERS_PARALLELISM=false (avoids warnings)

### BoeNet: Dockerfile.boenet.cuda (GPU) - NEW

**Location**: `docker/Dockerfile.boenet.cuda`
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

# Install language modeling dependencies
RUN pip install --no-cache-dir \
    tokenizers>=0.15.0 \
    transformers>=4.36.0 \
    datasets>=2.16.0 \
    sentencepiece>=0.1.99

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for text data
RUN mkdir -p /app/data/text /app/data/tokenizers

# Create non-root user
RUN useradd -m -u 1000 boenet && chown -R boenet:boenet /app
USER boenet

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TOKENIZERS_PARALLELISM=false

CMD ["python", "train_char_boenet.py"]
```

**Key Features**:
- Same CUDA 12.8.0 base as BFSNet (Blackwell support)
- PyTorch 2.7.1 cu128 (latest)
- Full language modeling stack
- Text data directories

---

## 📦 Volume Mounts & Data Persistence

### BFSNet (Vision) Volume Strategy

| Container Path | Host Path | Purpose | Persistence |
|----------------|-----------|---------|-------------|
| `/app/data` | `./data` | FashionMNIST images (auto-downloaded) | ✅ Persist |
| `/app/runs` | `./runs` | Training outputs, checkpoints, CSV | ✅ **CRITICAL** |
| `/app/configs` | `./configs` | Configuration YAML files | ✅ Read-only |

**Data Directory Structure (Vision)**:
```
data/
└── FashionMNIST/              # Auto-downloaded by torchvision
    ├── raw/
    │   ├── train-images-idx3-ubyte.gz
    │   ├── train-labels-idx1-ubyte.gz
    │   ├── t10k-images-idx3-ubyte.gz
    │   └── t10k-labels-idx1-ubyte.gz
    └── processed/
        ├── training.pt
        └── test.pt
```

### BoeNet (Language) Volume Strategy

| Container Path | Host Path | Purpose | Persistence |
|----------------|-----------|---------|-------------|
| `/app/data` | `./data` | Text datasets + tokenizer caches | ✅ Persist |
| `/app/runs` | `./runs` | Training outputs, checkpoints, CSV | ✅ **CRITICAL** |
| `/app/configs` | `./configs` | Configuration YAML files | ✅ Read-only |

**Data Directory Structure (Language)**:
```
data/
├── text/                      # Text datasets (manually added)
│   ├── shakespeare.txt        # ~1MB
│   ├── war_and_peace.txt      # ~3MB
│   └── tinystories/           # ~2GB
│       ├── train.txt
│       └── val.txt
│
├── tokenizers/                # Cached tokenizers
│   ├── char-ascii/
│   │   └── vocab.json
│   └── bpe-gpt2/
│       ├── vocab.json
│       └── merges.txt
│
└── processed/                 # Preprocessed data (optional)
    ├── shakespeare_train.pt
    └── shakespeare_val.pt
```

**Key Differences**:
1. **Manual Dataset Preparation**: Text files must be added to `data/text/` before training
2. **Tokenizer Caching**: Pre-trained tokenizers cached in `data/tokenizers/`
3. **Larger Datasets**: TinyStories (2GB), OpenWebText (40GB) require more storage

---

## 🚀 Detached Mode (Recommended for Long Runs)

Detached mode allows training to continue even if:
- Your terminal closes
- Your SSH session disconnects
- Your computer goes to sleep

### BFSNet (Vision) - Detached Training
```bash
# Build image
docker build -t bfsnet:cuda -f docker/Dockerfile.cuda .

# Run training matrix in background
docker run -d --gpus all \
    --name bfsnet_final_sweep \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    -v $(pwd)/configs:/app/configs \
    bfsnet:cuda python bfs_training_matrix.py \
        --config configs/bfsnet/experiment-config.yaml \
        --infer_script infer_fmnist_bfs.py

# Monitor progress
docker logs -f bfsnet_final_sweep

# Stop and cleanup
docker stop bfsnet_final_sweep
docker rm bfsnet_final_sweep
```

### BoeNet (Language) - Detached Training
```bash
# Build image
docker build -t boenet:cuda -f docker/Dockerfile.boenet.cuda .

# Run character-level training in background
docker run -d --gpus all \
    --name boenet_char_phase1 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    -v $(pwd)/configs:/app/configs \
    boenet:cuda python train_char_boenet.py \
        --config configs/boenet/char-level-full.yaml

# Monitor progress
docker logs -f boenet_char_phase1

# Stop and cleanup
docker stop boenet_char_phase1
docker rm boenet_char_phase1
```

### Monitoring Commands
```bash
# Follow live output (Ctrl+C to detach without stopping)
docker logs -f <container_name>

# View last 50 lines
docker logs --tail 50 <container_name>

# View last 100 lines, then follow
docker logs --tail 100 -f <container_name>

# Check if container is still running
docker ps

# Check container status (including stopped)
docker ps -a

# Get container resource usage
docker stats <container_name>
```

---

## 🎛️ Environment Variables

### BFSNet (Vision) Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `BFSNET_DEVICE` | `auto` | Device: `auto`, `cpu`, `cuda`, `cuda:0` |
| `BFSNET_DATA_ROOT` | `/app/data` | Dataset directory |
| `BFSNET_RUNS_ROOT` | `/app/runs` | Output directory |
| `CUDA_VISIBLE_DEVICES` | (all) | GPU device selection |
| `OMP_NUM_THREADS` | `4` | OpenMP thread count |
| `MKL_NUM_THREADS` | `4` | MKL thread count |

### BoeNet (Language) Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `BOENET_DEVICE` | `auto` | Device: `auto`, `cpu`, `cuda`, `cuda:0` |
| `BOENET_DATA_ROOT` | `/app/data` | Text dataset directory |
| `BOENET_RUNS_ROOT` | `/app/runs` | Output directory |
| `TOKENIZERS_PARALLELISM` | `false` | Disable tokenizer parallelism warnings |
| `CUDA_VISIBLE_DEVICES` | (all) | GPU device selection |
| `HF_HOME` | `/app/data/huggingface` | Hugging Face cache directory |

### Setting Environment Variables
```bash
# BFSNet example
docker run --rm \
    -e BFSNET_DEVICE=cuda:0 \
    -e OMP_NUM_THREADS=8 \
    --gpus all \
    bfsnet:cuda python train_fmnist_bfs.py

# BoeNet example
docker run --rm \
    -e BOENET_DEVICE=cuda:0 \
    -e TOKENIZERS_PARALLELISM=false \
    -e HF_HOME=/app/data/huggingface \
    --gpus all \
    boenet:cuda python train_char_boenet.py
```

---

## 🖥️ GPU Configuration

### Supported GPU Architectures

Both BFSNet and BoeNet CUDA images support all major NVIDIA GPU architectures:

| Architecture | GPUs | Compute Capability | Tested |
|--------------|------|-------------------|--------|
| Pascal | GTX 10 series | sm_60, sm_61 | ⚪ Not tested |
| Volta | Titan V, Quadro GV100 | sm_70 | ⚪ Not tested |
| Turing | RTX 20 series | sm_75 | ⚪ Not tested |
| Ampere | RTX 30 series, A100 | sm_80, sm_86 | ✅ Tested |
| Ada Lovelace | RTX 40 series | sm_89 | ⚪ Not tested |
| Hopper | H100 | sm_90 | ⚪ Not tested |
| **Blackwell** | **RTX 50 series (5080, 5090)** | **sm_120** | ✅ **Tested** |

**Blackwell Notes**:
- RTX 5080/5090 use compute capability **sm_120**
- Requires CUDA 12.8+ toolkit
- Requires PyTorch 2.7+ with cu128 wheels
- Host driver must support CUDA >= 12.8 (driver 570.x or newer)

### NVIDIA Container Toolkit

**Installation (Ubuntu)**:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verifying GPU Support
```bash
# Check host driver and CUDA version
nvidia-smi

# Verify Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Check PyTorch CUDA in BFSNet container
docker run --rm --gpus all bfsnet:cuda python -c \
    "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Check PyTorch CUDA in BoeNet container
docker run --rm --gpus all boenet:cuda python -c \
    "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Multi-GPU Training
```bash
# Use specific GPUs
docker run --rm --gpus '"device=0,1"' \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    boenet:cuda python train_char_boenet.py --gpu_ids 0,1

# Use all GPUs
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    boenet:cuda python train_char_boenet.py
```

---

## 📊 Dataset Preparation

### BFSNet (Vision) - Auto-Download

FashionMNIST automatically downloads on first run:
```python
# No manual preparation needed!
from torchvision.datasets import FashionMNIST

# Auto-downloads to /app/data/FashionMNIST/
train_data = FashionMNIST('/app/data', train=True, download=True)
test_data = FashionMNIST('/app/data', train=False, download=True)
```

### BoeNet (Language) - Manual Preparation

Text datasets must be manually added to `data/text/`:

#### Shakespeare (Character-Level)
```bash
# Download Shakespeare corpus
mkdir -p data/text
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    -O data/text/shakespeare.txt

# Verify
ls -lh data/text/shakespeare.txt
# Should be ~1.1 MB
```

#### War and Peace (Character-Level)
```bash
# Download War and Peace
wget https://www.gutenberg.org/files/2600/2600-0.txt \
    -O data/text/war_and_peace.txt

# Verify
ls -lh data/text/war_and_peace.txt
# Should be ~3.2 MB
```

#### TinyStories (Word-Level)
```bash
# Download TinyStories
mkdir -p data/text/tinystories
cd data/text/tinystories

# Download train/val splits
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# Rename for consistency
mv TinyStoriesV2-GPT4-train.txt train.txt
mv TinyStoriesV2-GPT4-valid.txt val.txt

# Verify
ls -lh
# train.txt should be ~2.1 GB
# val.txt should be ~50 MB
```

#### Using Inside Docker
```bash
# Mount text data directory
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    -v $(pwd)/configs:/app/configs \
    boenet:cuda python train_char_boenet.py \
        --config configs/boenet/char-level-test.yaml \
        --dataset /app/data/text/shakespeare.txt
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Container Name Already in Use
```
docker: Error response from daemon: Conflict. The container name "/boenet_phase1" is already in use
```

**Solution**:
```bash
# Remove the old container
docker rm -f boenet_phase1

# Then run again
docker run -d --gpus all --name boenet_phase1 ...
```

#### 2. Permission Denied on Volumes
```bash
# Fix ownership (Linux)
sudo chown -R $USER:$USER data runs configs

# Or run container as current user
docker run --rm --user $(id -u):$(id -g) \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    boenet:cpu python train_char_boenet.py
```

#### 3. GPU Not Detected
```bash
# Verify NVIDIA driver
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Check CUDA in container
docker run --rm --gpus all boenet:cuda python -c \
    "import torch; print(torch.cuda.is_available())"
```

#### 4. Blackwell GPU (RTX 50 series) Not Working

Ensure you have:
- NVIDIA driver 570.x or newer
- Using Dockerfile.cuda or Dockerfile.boenet.cuda with CUDA 12.8.0
- PyTorch 2.7.1 with cu128 wheels
```bash
# Rebuild with latest Dockerfile
docker build --no-cache -t boenet:cuda -f docker/Dockerfile.boenet.cuda .

# Verify sm_120 support
docker run --rm --gpus all boenet:cuda python -c \
    "import torch; print(f'CUDA arch list: {torch.cuda.get_arch_list()}')"
# Should include 'sm_120'
```

#### 5. Out of Memory (OOM)

**For Vision (BFSNet)**:
```bash
# Reduce batch size
docker run --rm --gpus all bfsnet:cuda \
    python train_fmnist_bfs.py --batch_size 32
```

**For Language (BoeNet)**:
```bash
# Reduce batch size and/or sequence length
docker run --rm --gpus all boenet:cuda \
    python train_char_boenet.py \
        --batch_size 32 \
        --seq_len 128
```

#### 6. Text Dataset Not Found
```
FileNotFoundError: data/text/shakespeare.txt
```

**Solution**:
```bash
# Verify text file exists on host
ls -lh data/text/shakespeare.txt

# Verify mount path
docker run --rm \
    -v $(pwd)/data:/app/data \
    boenet:cpu ls -lh /app/data/text/

# If missing, download manually (see Dataset Preparation section)
```

#### 7. Tokenizer Warnings
```
huggingface/tokenizers: The current process just got forked...
```

**Solution**:
```bash
# Disable tokenizer parallelism
docker run --rm \
    -e TOKENIZERS_PARALLELISM=false \
    boenet:cpu python train_char_boenet.py
```

---

## 📈 Performance Optimization

### 1. Use `--shm-size` for DataLoader Workers
```bash
# Increase shared memory for parallel data loading
docker run --rm --shm-size=2g --gpus all \
    boenet:cuda python train_char_boenet.py
```

### 2. Pin CPU for Consistent Benchmarks
```bash
# Use specific CPU cores
docker run --rm --cpuset-cpus="0-3" \
    bfsnet:cpu python train_fmnist_bfs.py
```

### 3. Use Host Network for Faster Data Access
```bash
# Use host network (careful with ports)
docker run --rm --network host --gpus all \
    boenet:cuda python train_char_boenet.py
```

### 4. Cache pip Packages Between Builds
```bash
# Use BuildKit cache
docker build --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t boenet:cuda -f docker/Dockerfile.boenet.cuda .
```

### 5. Pre-download Datasets

**BFSNet (Vision)**:
```bash
# Pre-download FashionMNIST
docker run --rm -v $(pwd)/data:/app/data bfsnet:cpu \
    python -c "from torchvision.datasets import FashionMNIST; FashionMNIST('/app/data', download=True)"
```

**BoeNet (Language)**:
```bash
# Manually download text datasets (see Dataset Preparation section)
# Pre-process and cache
docker run --rm -v $(pwd)/data:/app/data boenet:cpu \
    python scripts/preprocess_text.py --dataset shakespeare
```

---

## 🔬 Development & Debugging

### Interactive Shell
```bash
# BFSNet interactive shell
docker run --rm -it --gpus all \
    -v $(pwd):/app \
    bfsnet:cuda bash

# BoeNet interactive shell
docker run --rm -it --gpus all \
    -v $(pwd):/app \
    boenet:cuda bash
```

### Check Python Environment
```bash
# List installed packages
docker run --rm boenet:cpu pip list

# Check specific package
docker run --rm boenet:cpu python -c "import transformers; print(transformers.__version__)"

# Check CUDA availability
docker run --rm --gpus all boenet:cuda python -c \
    "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### Run Tests Inside Container
```bash
# BFSNet tests
docker run --rm \
    -v $(pwd):/app \
    bfsnet:cpu pytest tests/unit/ -v

# BoeNet tests
docker run --rm \
    -v $(pwd):/app \
    boenet:cpu pytest tests/boenet/ -v
```

---

## 📦 Docker Compose

### docker-compose.yaml

**Location**: `docker/docker-compose.yaml`
```yaml
version: '3.8'

services:
  # BFSNet CPU (Vision - Historical)
  bfsnet-cpu:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: bfsnet:cpu
    volumes:
      - ../data:/app/data
      - ../runs:/app/runs
      - ../configs:/app/configs
    command: python train_fmnist_bfs.py --epochs 10

  # BFSNet GPU (Vision - Historical)
  bfsnet-gpu:
    build:
      context: ..
      dockerfile: docker/Dockerfile.cuda
    image: bfsnet:cuda
    volumes:
      - ../data:/app/data
      - ../runs:/app/runs
      - ../configs:/app/configs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: python train_fmnist_bfs.py --epochs 10

  # BoeNet CPU (Language - Active)
  boenet-cpu:
    build:
      context: ..
      dockerfile: docker/Dockerfile.boenet
    image: boenet:cpu
    volumes:
      - ../data:/app/data
      - ../runs:/app/runs
      - ../configs:/app/configs
    command: python train_char_boenet.py --config configs/boenet/char-level-test.yaml

  # BoeNet GPU (Language - Active)
  boenet-gpu:
    build:
      context: ..
      dockerfile: docker/Dockerfile.boenet.cuda
    image: boenet:cuda
    volumes:
      - ../data:/app/data
      - ../runs:/app/runs
      - ../configs:/app/configs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: python train_char_boenet.py --config configs/boenet/char-level-test.yaml
```

### Using Docker Compose
```bash
# Build all images
docker-compose -f docker/docker-compose.yaml build

# Run BFSNet CPU
docker-compose -f docker/docker-compose.yaml up bfsnet-cpu

# Run BFSNet GPU (detached)
docker-compose -f docker/docker-compose.yaml up -d bfsnet-gpu

# Run BoeNet CPU
docker-compose -f docker/docker-compose.yaml up boenet-cpu

# Run BoeNet GPU (detached)
docker-compose -f docker/docker-compose.yaml up -d boenet-gpu

# View logs
docker-compose -f docker/docker-compose.yaml logs -f boenet-gpu

# Stop all services
docker-compose -f docker/docker-compose.yaml down
```

---

## 📝 Installed Python Packages

### BFSNet (Vision) Packages

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.7.1 (CUDA) / 2.1.0 (CPU) | Deep learning framework |
| torchvision | Latest | Vision datasets and transforms |
| tqdm | >=4.66.0 | Progress bars |
| pandas | >=2.0.0 | Data analysis |
| matplotlib | >=3.7.0 | Visualization |
| pyyaml | >=6.0.0 | YAML config files |

### BoeNet (Language) Packages

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.7.1 (CUDA/CPU) | Deep learning framework |
| tokenizers | >=0.15.0 | Fast tokenization (BPE, char) |
| transformers | >=4.36.0 | Pre-trained models and tokenizers |
| datasets | >=2.16.0 | Dataset loading utilities |
| sentencepiece | >=0.1.99 | SentencePiece tokenization |
| tqdm | >=4.66.0 | Progress bars |
| pandas | >=2.0.0 | Data analysis |
| pyyaml | >=6.0.0 | YAML config files |

---

## 🗺️ Roadmap

### ✅ Completed (BFSNet)
- [x] CPU Dockerfile
- [x] CUDA Dockerfile with Blackwell support
- [x] Docker Compose orchestration
- [x] Volume mount strategy
- [x] Detached mode support
- [x] GPU verification scripts

### 🚧 In Progress (BoeNet)
- [x] CPU Dockerfile (CREATED)
- [x] CUDA Dockerfile (CREATED)
- [ ] Text dataset auto-download scripts
- [ ] Tokenizer cache optimization
- [ ] Multi-node distributed training support

### ⏳ Planned (Future)
- [ ] Kubernetes manifests
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] Pre-built images on Docker Hub
- [ ] CI/CD integration

---

## 📞 Support

### BFSNet (Vision)
- 📚 See `docs/bfsnet_architecture.md` for architecture details
- 🐛 Known issues documented in `BFSNET_FINAL_REPORT.md`
- ✅ Project status: COMPLETE

### BoeNet (Language)
- 📚 See `docs/boenet_architecture.md` for architecture details (IN PROGRESS)
- 🎯 See `BOENET_VISION.md` for project goals
- 🚧 Project status: IN PROGRESS (Phase 1)

### General
- 📧 Contact: [your-email@example.com]
- 📝 Issues: Contact project owner (closed source)

---

**Last Updated**: December 20, 2025  
**Status**: BFSNet Docker ✅ COMPLETE | BoeNet Docker 🚧 IN PROGRESS  
**Tested On**: RTX 5080, RTX 5090 (Blackwell), RTX 3090 (Ampere)

**⚠️ Proprietary Software**: This project is closed source. All rights reserved.