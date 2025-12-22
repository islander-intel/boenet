# BFSNet & BoeNet Scripts Documentation

**Repository**: BoeNet (formerly BFSNet)  
**Purpose**: Utility scripts for vision (BFSNet - COMPLETE) and language (BoeNet - IN PROGRESS) tasks  
**Last Updated**: December 20, 2025

---

## 📋 Overview

This directory contains utility scripts for both:
1. **BFSNet (Vision)** - ✅ COMPLETE - Legacy scripts for reference
2. **BoeNet (Language)** - 🚧 IN PROGRESS - Active development scripts

### Directory Structure
```
scripts/
├── README.md                        # This file
│
├── bfsnet/                          # BFSNet (Vision) - LEGACY
│   ├── README.md                    # BFSNet-specific documentation
│   ├── bench_sparse_vs_dense.py     # Performance benchmarking
│   ├── analyze_policy.py            # Policy distribution analysis
│   ├── visualize_tree.py            # Tree structure visualization
│   └── export_results.py            # CSV/JSON result export
│
└── boenet/                          # BoeNet (Language) - ACTIVE
    ├── README.md                    # BoeNet-specific documentation
    ├── download_shakespeare.py      # Download Shakespeare dataset
    ├── download_tinystories.py      # Download TinyStories dataset
    ├── preprocess_text.py           # Text preprocessing utilities
    ├── tokenizer_utils.py           # Tokenization helpers
    ├── analyze_perplexity.py        # Perplexity analysis
    └── generate_text.py             # Text generation utilities
```

---

## 🎯 Quick Start

### BFSNet (Vision - Legacy Reference)
```bash
# Benchmark sparse vs dense execution
python scripts/bfsnet/bench_sparse_vs_dense.py \
    --k_values 2,4,6 \
    --batch_sizes 32,64,128 \
    --device cuda

# Analyze trained policy distribution
python scripts/bfsnet/analyze_policy.py \
    --ckpt checkpoints/bfsnet_final.pt \
    --output figures/policy_distribution.png

# Visualize BFS tree structure
python scripts/bfsnet/visualize_tree.py \
    --ckpt checkpoints/bfsnet_final.pt \
    --sample_image data/sample.png \
    --output figures/tree_structure.png
```

### BoeNet (Language - Active Development)
```bash
# Download Shakespeare dataset
python scripts/boenet/download_shakespeare.py \
    --output data/text/shakespeare.txt

# Download TinyStories dataset
python scripts/boenet/download_tinystories.py \
    --output data/text/tinystories/

# Preprocess text data
python scripts/boenet/preprocess_text.py \
    --input data/text/shakespeare.txt \
    --output data/processed/shakespeare_train.pt \
    --vocab_size 256 \
    --seq_len 128

# Train character-level tokenizer
python scripts/boenet/tokenizer_utils.py \
    --mode train \
    --input data/text/shakespeare.txt \
    --output data/tokenizers/char-ascii/

# Generate text from checkpoint
python scripts/boenet/generate_text.py \
    --ckpt checkpoints/boenet_char.pt \
    --prompt "To be or not to be" \
    --max_tokens 100 \
    --temperature 0.8
```

---

## 📚 BFSNet Scripts (Vision - LEGACY)

**⚠️ STATUS**: BFSNet project is **COMPLETE**. These scripts are preserved for:
- Historical reference
- Methodology documentation
- Code reuse in BoeNet

### Core Scripts

#### 1. bench_sparse_vs_dense.py

**Purpose**: Benchmark sparse vs dense execution modes for performance analysis.

**Location**: `scripts/bfsnet/bench_sparse_vs_dense.py`

**Usage**:
```bash
# Basic benchmark
python scripts/bfsnet/bench_sparse_vs_dense.py

# Custom parameters
python scripts/bfsnet/bench_sparse_vs_dense.py \
    --k_values 2,4,6,8 \
    --batch_sizes 32,64,128,256 \
    --hidden_dim 128 \
    --max_depth 3 \
    --num_iterations 100 \
    --device cuda \
    --output benchmarks/results.csv
```

**Arguments**:
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--k_values` | int[] | [2,4,6] | K values to benchmark |
| `--batch_sizes` | int[] | [32,64,128] | Batch sizes |
| `--hidden_dim` | int | 128 | Hidden dimension |
| `--max_depth` | int | 3 | Tree depth |
| `--num_iterations` | int | 100 | Benchmark iterations |
| `--device` | str | "auto" | Device to use |
| `--output` | str | None | Output CSV file |

**Output**:
```
BFSNet Sparse vs Dense Benchmark
================================
Device: NVIDIA GeForce RTX 5080

K=2, Batch=64, Depth=3
----------------------
Mode       | Forward (ms) | Backward (ms) | Throughput
-----------+--------------+---------------+-----------
sparse     |        1.234 |         2.345 |    51,234
soft_full  |        2.567 |         4.890 |    24,567
Speedup    |        2.08x |         2.08x |      2.08x
```

**Status**: ✅ COMPLETE - Working reference implementation

---

#### 2. analyze_policy.py

**Purpose**: Analyze learned policy distribution to determine optimal greedy threshold.

**Location**: `scripts/bfsnet/analyze_policy.py`

**Usage**:
```bash
# Analyze policy from checkpoint
python scripts/bfsnet/analyze_policy.py \
    --ckpt checkpoints/bfsnet_final.pt \
    --num_samples 1000 \
    --output figures/policy_distribution.png

# Compare multiple checkpoints
python scripts/bfsnet/analyze_policy.py \
    --ckpts checkpoints/lambda_0.01.pt,checkpoints/lambda_0.05.pt \
    --labels "λ=0.01","λ=0.05" \
    --output figures/policy_comparison.png
```

**Arguments**:
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ckpt` | str | required | Path to checkpoint |
| `--ckpts` | str | None | Comma-separated checkpoint paths (for comparison) |
| `--labels` | str | None | Labels for comparison plot |
| `--num_samples` | int | 1000 | Number of samples to analyze |
| `--output` | str | required | Output image path |
| `--device` | str | "cpu" | Device to use |

**Output**:
```
Policy Distribution Analysis
============================
Checkpoint: checkpoints/bfsnet_final.pt

Statistics:
  Mean grow_prob: 0.4457
  Std dev:        0.0157
  Min:            0.3771
  Max:            0.4567
  % >= 0.5:       0.00%

Recommended greedy_threshold: 0.42

Distribution histogram saved to: figures/policy_distribution.png
```

**Status**: ✅ COMPLETE - Critical for threshold tuning

---

#### 3. visualize_tree.py

**Purpose**: Visualize BFS tree structure for specific inputs.

**Location**: `scripts/bfsnet/visualize_tree.py`

**Usage**:
```bash
# Visualize tree for single image
python scripts/bfsnet/visualize_tree.py \
    --ckpt checkpoints/bfsnet_final.pt \
    --sample_image data/sample.png \
    --output figures/tree_structure.png

# Visualize with different thresholds
python scripts/bfsnet/visualize_tree.py \
    --ckpt checkpoints/bfsnet_final.pt \
    --sample_image data/sample.png \
    --thresholds 0.3,0.42,0.5 \
    --output figures/tree_threshold_comparison.png
```

**Arguments**:
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ckpt` | str | required | Path to checkpoint |
| `--sample_image` | str | required | Input image path |
| `--thresholds` | str | "0.5" | Comma-separated thresholds |
| `--output` | str | required | Output image path |
| `--show_probabilities` | flag | False | Show grow_prob values on nodes |

**Output**: Tree visualization showing:
- Root node
- Expanded children at each depth
- Grow probabilities (if `--show_probabilities`)
- Final classification

**Status**: ✅ COMPLETE - Useful for debugging

---

#### 4. export_results.py

**Purpose**: Export training matrix results to various formats.

**Location**: `scripts/bfsnet/export_results.py`

**Usage**:
```bash
# Export to Excel with formatted tables
python scripts/bfsnet/export_results.py \
    --input runs/20251218_120000/matrix_results.csv \
    --output results/bfsnet_final_results.xlsx \
    --format excel

# Export to LaTeX table
python scripts/bfsnet/export_results.py \
    --input runs/20251218_120000/matrix_results.csv \
    --output results/bfsnet_table.tex \
    --format latex \
    --top_n 10

# Export summary statistics
python scripts/bfsnet/export_results.py \
    --input runs/20251218_120000/matrix_results.csv \
    --output results/bfsnet_summary.md \
    --format markdown
```

**Status**: ✅ COMPLETE - Useful for reporting

---

### Legacy Scripts (Not Actively Maintained)

These scripts are preserved but may require updates:

| Script | Purpose | Status |
|--------|---------|--------|
| `profile_memory.py` | Memory profiling | ⚠️ May need updates |
| `export_onnx.py` | ONNX model export | ⚠️ Untested |
| `visualize_activations.py` | Activation heatmaps | ⚠️ May need updates |

---

## 🚀 BoeNet Scripts (Language - ACTIVE)

**⚠️ STATUS**: BoeNet is in **ACTIVE DEVELOPMENT**. Scripts are being created as needed.

### Phase 1: Character-Level Scripts

#### 1. download_shakespeare.py (✅ READY)

**Purpose**: Download Shakespeare corpus for character-level training.

**Location**: `scripts/boenet/download_shakespeare.py`

**Usage**:
```bash
# Download to default location
python scripts/boenet/download_shakespeare.py

# Download to custom location
python scripts/boenet/download_shakespeare.py \
    --output data/text/shakespeare.txt
```

**Implementation**:
```python
#!/usr/bin/env python3
"""Download Shakespeare corpus for character-level training."""

import argparse
import urllib.request
from pathlib import Path

def download_shakespeare(output_path: str):
    """Download tiny shakespeare dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Shakespeare corpus from {url}...")
    urllib.request.urlretrieve(url, output_path)
    
    # Verify download
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Downloaded successfully: {output_path}")
    print(f"   Size: {size_mb:.2f} MB")
    
    # Print first few lines
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(5)]
    print(f"\nFirst few lines:")
    print("".join(lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Shakespeare corpus')
    parser.add_argument('--output', type=str, default='data/text/shakespeare.txt',
                        help='Output file path')
    args = parser.parse_args()
    
    download_shakespeare(args.output)
```

**Status**: ✅ READY - Simple download script

---

#### 2. download_tinystories.py (✅ READY)

**Purpose**: Download TinyStories dataset for word-level training.

**Location**: `scripts/boenet/download_tinystories.py`

**Usage**:
```bash
# Download to default location
python scripts/boenet/download_tinystories.py

# Download to custom location
python scripts/boenet/download_tinystories.py \
    --output data/text/tinystories/
```

**Implementation**:
```python
#!/usr/bin/env python3
"""Download TinyStories dataset for word-level training."""

import argparse
import urllib.request
from pathlib import Path

def download_tinystories(output_dir: str):
    """Download TinyStories train and validation sets."""
    base_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
    files = {
        "train.txt": f"{base_url}/TinyStoriesV2-GPT4-train.txt",
        "val.txt": f"{base_url}/TinyStoriesV2-GPT4-valid.txt"
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, url in files.items():
        output_path = output_dir / filename
        print(f"Downloading {filename} from {url}...")
        urllib.request.urlretrieve(url, output_path)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Downloaded: {output_path} ({size_mb:.2f} MB)")
    
    print(f"\n✅ All files downloaded to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download TinyStories dataset')
    parser.add_argument('--output', type=str, default='data/text/tinystories/',
                        help='Output directory path')
    args = parser.parse_args()
    
    download_tinystories(args.output)
```

**Status**: ✅ READY - Downloads ~2GB dataset

---

#### 3. preprocess_text.py (🚧 IN PROGRESS)

**Purpose**: Preprocess text files for efficient training.

**Location**: `scripts/boenet/preprocess_text.py`

**Planned Usage**:
```bash
# Preprocess Shakespeare (character-level)
python scripts/boenet/preprocess_text.py \
    --input data/text/shakespeare.txt \
    --output data/processed/shakespeare_train.pt \
    --mode char \
    --vocab_size 256 \
    --seq_len 128 \
    --train_split 0.9

# Preprocess TinyStories (word-level)
python scripts/boenet/preprocess_text.py \
    --input data/text/tinystories/train.txt \
    --output data/processed/tinystories_train.pt \
    --mode bpe \
    --vocab_size 50257 \
    --seq_len 256
```

**Planned Arguments**:
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | required | Input text file |
| `--output` | str | required | Output .pt file |
| `--mode` | str | "char" | Tokenization mode: "char" or "bpe" |
| `--vocab_size` | int | 256 | Vocabulary size |
| `--seq_len` | int | 128 | Sequence length |
| `--train_split` | float | 0.9 | Train/val split ratio |

**Status**: 🚧 IN PROGRESS - Needed for Phase 1

---

#### 4. tokenizer_utils.py (🚧 IN PROGRESS)

**Purpose**: Train and manage tokenizers (character-level, BPE).

**Location**: `scripts/boenet/tokenizer_utils.py`

**Planned Usage**:
```bash
# Train character-level tokenizer
python scripts/boenet/tokenizer_utils.py \
    --mode train \
    --input data/text/shakespeare.txt \
    --output data/tokenizers/char-ascii/ \
    --tokenizer_type char

# Train BPE tokenizer
python scripts/boenet/tokenizer_utils.py \
    --mode train \
    --input data/text/tinystories/train.txt \
    --output data/tokenizers/bpe-gpt2/ \
    --tokenizer_type bpe \
    --vocab_size 50257

# Test tokenizer
python scripts/boenet/tokenizer_utils.py \
    --mode test \
    --tokenizer_path data/tokenizers/char-ascii/ \
    --text "To be or not to be"
```

**Status**: 🚧 IN PROGRESS - Critical for Phase 1

---

#### 5. generate_text.py (🚧 IN PROGRESS)

**Purpose**: Generate text from trained BoeNet models.

**Location**: `scripts/boenet/generate_text.py`

**Planned Usage**:
```bash
# Generate from character-level model
python scripts/boenet/generate_text.py \
    --ckpt checkpoints/boenet_char.pt \
    --prompt "To be or not to be" \
    --max_tokens 100 \
    --temperature 0.8

# Generate with different sampling strategies
python scripts/boenet/generate_text.py \
    --ckpt checkpoints/boenet_char.pt \
    --prompt "Once upon a time" \
    --max_tokens 200 \
    --temperature 0.8 \
    --top_k 40 \
    --top_p 0.9
```

**Planned Arguments**:
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ckpt` | str | required | Model checkpoint path |
| `--prompt` | str | required | Input prompt |
| `--max_tokens` | int | 100 | Maximum tokens to generate |
| `--temperature` | float | 0.8 | Sampling temperature |
| `--top_k` | int | 40 | Top-k sampling |
| `--top_p` | float | 0.9 | Nucleus sampling |
| `--greedy_threshold` | float | 0.42 | BFS greedy threshold |

**Status**: 🚧 IN PROGRESS - Needed for Phase 1 validation

---

#### 6. analyze_perplexity.py (⏳ PLANNED)

**Purpose**: Analyze perplexity across different inputs and compare to baselines.

**Planned Usage**:
```bash
# Analyze perplexity on validation set
python scripts/boenet/analyze_perplexity.py \
    --ckpt checkpoints/boenet_char.pt \
    --dataset data/text/shakespeare.txt \
    --split val \
    --output results/perplexity_analysis.json

# Compare to baseline
python scripts/boenet/analyze_perplexity.py \
    --ckpt checkpoints/boenet_char.pt \
    --baseline_ckpt checkpoints/lstm_baseline.pt \
    --dataset data/text/shakespeare.txt \
    --output results/perplexity_comparison.png
```

**Status**: ⏳ PLANNED - Phase 1 evaluation

---

### Phase 2: Word-Level Scripts (Planned)

| Script | Purpose | Status |
|--------|---------|--------|
| `train_bpe_tokenizer.py` | Train BPE tokenizer on TinyStories | ⏳ PLANNED |
| `analyze_vocabulary.py` | Vocabulary coverage analysis | ⏳ PLANNED |
| `compare_baselines.py` | Compare to GPT-2 small, LSTM | ⏳ PLANNED |
| `export_onnx.py` | Export BoeNet to ONNX | ⏳ PLANNED |

---

## 🔄 Migration Guide: BFSNet → BoeNet Scripts

### Conceptual Mapping

| BFSNet Script | BoeNet Equivalent | Key Changes |
|---------------|-------------------|-------------|
| `bench_sparse_vs_dense.py` | `bench_efficiency.py` | Measure FLOPs instead of nodes |
| `analyze_policy.py` | `analyze_policy.py` | Same concept, different inputs |
| `visualize_tree.py` | `visualize_sequence_tree.py` | Show tree per token |
| `export_results.py` | `export_results.py` | Same, add perplexity metrics |

### Reusable Components

These BFSNet utilities can be directly reused in BoeNet:

1. **Policy Analysis** (`analyze_policy.py`)
   - REINFORCE policy gradients are identical
   - Grow_prob distribution analysis applies
   - Threshold tuning methodology same

2. **Benchmarking Framework** (`bench_sparse_vs_dense.py`)
   - Forward/backward timing logic
   - GPU synchronization
   - Statistical analysis

3. **Export Utilities** (`export_results.py`)
   - CSV/JSON/LaTeX export
   - Formatting helpers
   - Summary statistics

### New Capabilities Needed for BoeNet

1. **Text Processing**
   - Tokenization (character, BPE)
   - Vocabulary management
   - Sequence batching

2. **Language Metrics**
   - Perplexity calculation
   - BLEU score (later)
   - Generation quality metrics

3. **Sequence Visualization**
   - Tree per token (not per sample)
   - Temporal dependencies
   - Attention-like visualizations

---

## 🛠️ Development Guidelines

### Creating New Scripts

1. **File Naming Convention**
```
   <verb>_<noun>.py
   Examples:
   - download_shakespeare.py
   - analyze_perplexity.py
   - visualize_sequence.py
```

2. **Script Template**
```python
   #!/usr/bin/env python3
   """
   Brief description of what this script does.
   
   Example usage:
       python script_name.py --arg1 value1 --arg2 value2
   """
   
   import argparse
   from pathlib import Path
   
   def main(args):
       """Main function."""
       # Implementation here
       pass
   
   if __name__ == '__main__':
       parser = argparse.ArgumentParser(description='Script description')
       parser.add_argument('--arg1', type=str, required=True,
                           help='Argument 1 description')
       parser.add_argument('--arg2', type=int, default=100,
                           help='Argument 2 description')
       args = parser.parse_args()
       
       main(args)
```

3. **Documentation Requirements**
   - Docstring at top explaining purpose
   - Example usage in docstring
   - Help text for all arguments
   - Update this README with script info

4. **Error Handling**
   - Validate inputs (file exists, valid values)
   - Clear error messages
   - Graceful failure (don't crash mid-process)

5. **Testing**
   - Test with minimal inputs
   - Test edge cases
   - Document expected runtime

---

## 📊 Script Status Summary

### BFSNet Scripts (Vision - COMPLETE)

| Script | Status | Purpose | Priority |
|--------|--------|---------|----------|
| `bench_sparse_vs_dense.py` | ✅ COMPLETE | Performance benchmarking | Reference |
| `analyze_policy.py` | ✅ COMPLETE | Policy distribution analysis | Reference |
| `visualize_tree.py` | ✅ COMPLETE | Tree visualization | Reference |
| `export_results.py` | ✅ COMPLETE | Result export (Excel, LaTeX) | Reference |

### BoeNet Scripts (Language - IN PROGRESS)

| Script | Status | Purpose | Priority | Phase |
|--------|--------|---------|----------|-------|
| `download_shakespeare.py` | ✅ READY | Download Shakespeare | 🔴 CRITICAL | 1 |
| `download_tinystories.py` | ✅ READY | Download TinyStories | 🟡 HIGH | 2 |
| `preprocess_text.py` | 🚧 IN PROGRESS | Text preprocessing | 🔴 CRITICAL | 1 |
| `tokenizer_utils.py` | 🚧 IN PROGRESS | Tokenizer management | 🔴 CRITICAL | 1 |
| `generate_text.py` | 🚧 IN PROGRESS | Text generation | 🔴 CRITICAL | 1 |
| `analyze_perplexity.py` | ⏳ PLANNED | Perplexity analysis | 🟡 HIGH | 1 |
| `train_bpe_tokenizer.py` | ⏳ PLANNED | BPE training | 🟡 HIGH | 2 |
| `compare_baselines.py` | ⏳ PLANNED | Baseline comparison | 🟢 MEDIUM | 2 |

---

## 🔍 Debugging Scripts

### Quick Diagnostics
```bash
# Check if text dataset is accessible
python -c "
from pathlib import Path
p = Path('data/text/shakespeare.txt')
print(f'Exists: {p.exists()}')
print(f'Size: {p.stat().st_size / 1024:.2f} KB')
"

# Test tokenizer
python -c "
text = 'To be or not to be'
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encoded = [stoi[ch] for ch in text]
decoded = ''.join([itos[i] for i in encoded])
print(f'Original: {text}')
print(f'Encoded:  {encoded}')
print(f'Decoded:  {decoded}')
assert text == decoded
print('✅ Tokenizer working!')
"

# Check checkpoint structure
python -c "
import torch
ckpt = torch.load('checkpoints/boenet_char.pt', map_location='cpu')
print('Checkpoint keys:', list(ckpt.keys()))
if 'config' in ckpt:
    print('Config:', ckpt['config'])
"
```

---

## 📖 Examples

### BFSNet: Full Analysis Pipeline
```bash
# 1. Train model
python train_fmnist_bfs.py \
    --epochs 10 \
    --lambda_efficiency 0.05 \
    --save_path checkpoints/bfsnet_analysis.pt

# 2. Analyze policy distribution
python scripts/bfsnet/analyze_policy.py \
    --ckpt checkpoints/bfsnet_analysis.pt \
    --output figures/policy_dist.png

# 3. Visualize tree for sample
python scripts/bfsnet/visualize_tree.py \
    --ckpt checkpoints/bfsnet_analysis.pt \
    --sample_image data/sample_boot.png \
    --output figures/boot_tree.png

# 4. Benchmark performance
python scripts/bfsnet/bench_sparse_vs_dense.py \
    --k_values 3 \
    --output benchmarks/final_benchmark.csv

# 5. Export results
python scripts/bfsnet/export_results.py \
    --input runs/20251218_120000/matrix_results.csv \
    --output results/bfsnet_final.xlsx
```

### BoeNet: Phase 1 Setup
```bash
# 1. Download dataset
python scripts/boenet/download_shakespeare.py

# 2. Preprocess text
python scripts/boenet/preprocess_text.py \
    --input data/text/shakespeare.txt \
    --output data/processed/shakespeare.pt

# 3. Train tokenizer
python scripts/boenet/tokenizer_utils.py \
    --mode train \
    --input data/text/shakespeare.txt \
    --output data/tokenizers/char-ascii/

# 4. Train model
python train_char_boenet.py \
    --config configs/boenet/char-level-test.yaml

# 5. Generate text
python scripts/boenet/generate_text.py \
    --ckpt checkpoints/boenet_char.pt \
    --prompt "To be or not to be" \
    --max_tokens 100

# 6. Analyze perplexity
python scripts/boenet/analyze_perplexity.py \
    --ckpt checkpoints/boenet_char.pt \
    --dataset data/text/shakespeare.txt
```

---

## 📞 Support

### BFSNet Scripts
- 📚 See `scripts/bfsnet/README.md` for detailed documentation
- 📖 See `docs/bfsnet_architecture.md` for context
- ✅ Status: COMPLETE - Reference only

### BoeNet Scripts
- 📚 See `scripts/boenet/README.md` for detailed documentation (IN PROGRESS)
- 📖 See `docs/boenet_architecture.md` for context (IN PROGRESS)
- 🚧 Status: IN PROGRESS - Active development

### General
- 📧 Contact: [your-email@example.com]
- 📝 Issues: Contact project owner (closed source)

---

## 🗺️ Roadmap

### ✅ Completed (BFSNet)
- [x] Performance benchmarking
- [x] Policy analysis tools
- [x] Tree visualization
- [x] Result export utilities

### 🚧 In Progress (BoeNet Phase 1)
- [x] Dataset download scripts (Shakespeare, TinyStories)
- [ ] Text preprocessing pipeline
- [ ] Tokenizer utilities
- [ ] Text generation tools
- [ ] Perplexity analysis

### ⏳ Planned (BoeNet Phase 2+)
- [ ] BPE tokenizer training
- [ ] Baseline comparison tools
- [ ] ONNX export
- [ ] Distributed training utilities

---

**Last Updated**: December 20, 2025  
**Status**: BFSNet scripts ✅ COMPLETE | BoeNet scripts 🚧 IN PROGRESS (Phase 1)  
**Next Priority**: Complete `preprocess_text.py`, `tokenizer_utils.py`, `generate_text.py` for Phase 1

**⚠️ Proprietary Software**: This project is closed source. All rights reserved.