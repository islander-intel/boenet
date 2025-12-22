#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet_training_matrix.py (v1.0.0 - Language Model)

Run a *full factorial* matrix of BoeNet experiments on Shakespeare/TinyStories by spawning:
  * train_boenet.py   (training: logs + checkpoint)
  * infer_boenet.py   (inference: prints ONE JSON object OR a plain-text perplexity line)

Converted from bfs_training_matrix.py (Vision) to boenet_training_matrix.py (Language)
--------------------------------------------------------------------------------------
Key Changes:
  - REMOVED: FashionMNIST references, accuracy metrics, class names
  - ADDED: vocab_size, seq_len, embed_dim sweep parameters
  - CHANGED: Accuracy columns -> Perplexity columns
  - CHANGED: Calls train_boenet.py and infer_boenet.py instead of BFSNet scripts
  - UNCHANGED: All sweep logic, factorial matrix, subprocess runners, progress bar

v1.0.0 Greedy Threshold Enhancement (same as BFSNet v2.0.0)
-----------------------------------------------------------
  - greedy_threshold sweep parameter (0.30-0.50 range)
  - lambda_efficiency sweep parameter (0.0-0.1 range)
  - Coupled (lambda, threshold) pair testing

What this collects
------------------
Training (parsed from stdout):
  - Per-epoch: Train/Val Loss, Train/Val PPL, policy_loss, avg_nodes
  - Best Val PPL and epoch (derived from epoch lines)
  - Timing from trainer's __SUMMARY__ JSON:
      - total_training_time_sec: Total wall time for all epochs
      - avg_epoch_time_sec: Average time per epoch
      - last_epoch_time_sec: Time for the final epoch
  - Policy metrics: lambda_efficiency, num_rollouts, beta_entropy

Inference (preferred JSON, fallback text):
  - If inference prints JSON to stdout, we map fields:
      Perplexity:  val_ppl
      Loss:        val_loss
      Latency:     latency_ms_mean, latency_ms_p50, latency_ms_p90, latency_ms_p99
      Policy:      debug_policy_mean_grow_prob, debug_policy_above_threshold_pct
      Nodes:       avg_nodes_per_position, theoretical_max_nodes, sparsity_percent
      Misc:        device, num_samples, model_bytes

Outputs
-------
Under runs/YYYYmmdd_HHMMSS/:
  - matrix_results.csv           (one row per run; stable header)
  - matrix_results.jsonl         (same rows as JSON objects)
  - <tag>_rep*/run_###.log       (raw trainer stdout)
  - <tag>_rep*/infer_###.log     (raw inference stdout)
  - <tag>_rep*/infer_###.json    (the parsed inference JSON we used, if any)

Full-factorial grid
-------------------
Use CLI list options (comma-separated) to define the factorial grid, or specify
them in the config file. We generate *all* combinations across these lists:

  Language Model Parameters (NEW):
  * vocab_size:             --vocab_size_list (e.g., 256)
  * seq_len:                --seq_len_list (e.g., 64,128,256)
  * embed_dim:              --embed_dim_list (e.g., 32,64,128)
  * dataset:                --dataset (shakespeare, tinystories)

  v1.0.0 Policy Parameters:
  * lambda_efficiency:      --lambda_efficiency_list (e.g., 0.0,0.01,0.05,0.1)
  * greedy_threshold:       --greedy_threshold_list  (e.g., 0.30,0.35,0.40,0.42,0.45,0.50)
  * num_rollouts:           --num_rollouts_list      (typically 3)
  * beta_entropy:           --beta_entropy_list      (typically 0.01)
  * beta_policy:            --beta_policy_list       (typically 0.5)

  Architecture Parameters:
  * K (children):           --k_values  (e.g., 0,1,2,3)  - K=0 is the dense baseline
  * Pooling modes:          --poolings  (learned,sum,mean)
  * Hidden dims:            --hidden_dims
  * Learning rates:         --lrs
  * Batch sizes:            --batch_sizes
  * Weight decays:          --weight_decays
  * Max depths:             --max_depths

K=0 (Dense Baseline) Handling
-----------------------------
When K=0 (dense/MLP baseline):
  - max_children is set to 0
  - pooling_mode is forced to "mean" (pooling is meaningless for dense)
  - max_depth is forced to 1 (depth is meaningless for dense)
  - greedy_threshold is ignored (no growth decisions in dense model)
  - lambda_efficiency is ignored (no efficiency penalty in dense model)

Config File Format (YAML)
-------------------------
Example configs/boenet-config.yaml:
```yaml
# BoeNet v1.0.0 Training Matrix Configuration
sweep:
  # Language Model Parameters
  vocab_size_list: [256]
  seq_len_list: [64, 128]
  embed_dim_list: [32, 64]
  
  # v1.0.0 Policy Parameters
  lambda_efficiency_list: [0.0, 0.05, 0.1]
  greedy_threshold_list: [0.40, 0.42, 0.50]
  num_rollouts_list: [3]
  beta_entropy_list: [0.01]
  beta_policy_list: [0.5]
  
  # Architecture Parameters
  k_values: [0, 2, 3]
  poolings: [mean]
  hidden_dims: [64, 128]
  lrs: [0.001]
  batch_sizes: [64]
  weight_decays: [0.0]
  max_depths: [2]
  epochs_list: [10]

training:
  epochs: 10
  repeats: 1
  seed0: 42
  dataset: shakespeare

inference:
  infer_samples: 1000
  cpu_only: true
  debug_policy: true

paths:
  save_root: runs
  data_root: ./data
  train_script: train_boenet.py
  infer_script: infer_boenet.py
```

Progress Bar
------------
This script uses tqdm to display progress:
  - Current run / total runs
  - Percentage complete
  - Elapsed time
  - Estimated time remaining (ETA)
  - Current configuration tag

Author: BoeNet project (converted from BFSNet)
Version: 1.0.0
Date: 2025-12-22
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Try to import PyYAML for config file support
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    print("[info] tqdm not installed. Install with: pip install tqdm for progress bar support.")


# ------------------------------ Regex parsers ------------------------------ #

# Trainer epoch line (v1.0.0 format for language model)
RE_EPOCH_V1_LM = re.compile(
    r"Train Loss:\s*([0-9.eE+-]+)\s*\(lm=([0-9.eE+-]+),\s*policy=([0-9.eE+-]+)\)\s*\|\s*"
    r"Val Loss:\s*([0-9.eE+-]+)\s*\n\s*"
    r"Train PPL:\s*([0-9.]+)\s*\|\s*Val PPL:\s*([0-9.]+)\s*\|\s*"
    r"Avg Nodes/Position:\s*([0-9.]+)",
    re.MULTILINE
)

# Optional one-line JSON summary
RE_JSON_ANY = re.compile(r"^\s*(?:__SUMMARY__\s*)?(\{.*\})\s*$")

# Inference plain-text fallback for perplexity
RE_INFER_PPL_TEXT = re.compile(r"Val PPL:\s*([0-9.]+)", re.IGNORECASE)


# --------------------------- Small parsing helpers -------------------------- #

def _extract_float(pat: re.Pattern, text: str) -> Optional[float]:
    m = pat.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in d.items()}

def _safe_int(x: Any) -> Optional[int]:
    try:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x) if x.is_integer() else None
        return int(str(x))
    except Exception:
        return None

def _safe_float(x: Any) -> Optional[float]:
    """Safely convert a value to float, returning None on failure."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def _slug(v: Any) -> str:
    s = str(v)
    s = s.replace("->", "to").replace(".", "p").replace("-", "m").replace("+", "plus")
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    return s

def _merge(base: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    out = dict(base)
    out.update(overrides)
    return out

def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# --------------------------- Config file loading ---------------------------- #

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not _HAS_YAML:
        print(f"[config] Warning: PyYAML not installed. Install with: pip install pyyaml")
        return {}
    
    path = Path(config_path)
    if not path.exists():
        print(f"[config] Config file not found: {config_path}")
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"[config] Loaded configuration from: {config_path}")
        return config if config else {}
    except Exception as e:
        print(f"[config] Error loading config file: {e}")
        return {}


def get_config_value(config: Dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    """Get a value from a nested config dict."""
    if section in config and isinstance(config[section], dict):
        return config[section].get(key, default)
    return default


# --------------------------- Grid parsing utilities ------------------------- #

def _parse_list_str(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]

def _parse_list_float(raw: str) -> List[float]:
    return [float(tok) for tok in _parse_list_str(raw)]

def _parse_list_int(raw: str) -> List[int]:
    return [int(tok) for tok in _parse_list_str(raw)]


# --------------------------- Experiment definition -------------------------- #

def _exp_tag(
    k: int,
    pooling: str,
    hidden_dim: int,
    embed_dim: int,
    seq_len: int,
    lr: float,
    bs: int,
    wd: float,
    depth: int,
    lambda_eff: float,
    greedy_thresh: float,
    num_rollouts: int,
) -> str:
    """
    Compact but unique tag for a factorial cell (v1.0.0 language model).
    
    Includes embed_dim and seq_len for language model experiments.
    """
    parts = [
        f"k{k}",
        f"pool{pooling}",
        f"hd{hidden_dim}",
        f"ed{embed_dim}",
        f"sl{seq_len}",
        f"lr{_slug(lr)}",
        f"bs{bs}",
        f"wd{_slug(wd)}",
        f"d{depth}",
        f"lam{_slug(lambda_eff)}",
        f"thr{_slug(greedy_thresh)}",
        f"roll{num_rollouts}",
    ]
    return "_".join(parts)


def build_factorial_matrix(
    k_values: Sequence[int],
    poolings: Sequence[str],
    hidden_dims: Sequence[int],
    embed_dims: Sequence[int],
    seq_lens: Sequence[int],
    lrs: Sequence[float],
    batch_sizes: Sequence[int],
    weight_decays: Sequence[float],
    max_depths: Sequence[int],
    lambda_efficiency_list: Sequence[float],
    greedy_threshold_list: Sequence[float],
    num_rollouts_list: Sequence[int],
    beta_entropy_list: Sequence[float],
    beta_policy_list: Sequence[float],
    vocab_size: int = 256,
    dataset: str = "shakespeare",
) -> List[Dict[str, Any]]:
    """
    Build the full factorial list of experiment configs (v1.0.0 Language Model).
    
    K=0 (Dense Baseline) Handling:
    ------------------------------
    When K=0, the model becomes a simple MLP where:
      - max_children=0 (no branching)
      - pooling_mode="mean" (pooling is meaningless for dense)
      - max_depth=1 (depth is meaningless for dense)
      - greedy_threshold is ignored (no growth decisions)
      - lambda_efficiency is ignored (no efficiency penalty)
      - num_rollouts=1 (single forward pass)
    
    K>0 (BFS Tree) Handling:
    ------------------------
    For K>0, all parameters are swept as specified including the v1.0.0
    policy parameters (lambda_efficiency, greedy_threshold).
    """
    base_common = dict(
        lr_schedule="cosine",
        grad_clip=1.0,
        opt="adamw",
        vocab_size=vocab_size,
        dataset=dataset,
    )

    exps: List[Dict[str, Any]] = []

    for k in k_values:
        # K=0 optimization: single configuration for dense baseline
        if k == 0:
            for hd in hidden_dims:
                for ed in embed_dims:
                    for sl in seq_lens:
                        for lr in lrs:
                            for bs in batch_sizes:
                                for wd in weight_decays:
                                    tag = _exp_tag(k, "mean", hd, ed, sl, lr, bs, wd, 1, 0.0, 0.5, 1)
                                    cfg = _merge(
                                        base_common,
                                        tag=tag,
                                        batch_size=bs,
                                        hidden_dim=hd,
                                        embed_dim=ed,
                                        seq_len=sl,
                                        lr=lr,
                                        weight_decay=wd,
                                        max_depth=1,
                                        max_children=0,
                                        pooling_mode="mean",
                                        # v1.0.0 policy params (ignored for K=0)
                                        num_rollouts=1,
                                        lambda_efficiency=0.0,
                                        beta_entropy=0.01,
                                        beta_policy=0.5,
                                        greedy_threshold=0.5,
                                    )
                                    exps.append(cfg)
        else:
            # K>0: Full sweep including v1.0.0 policy parameters
            for depth in max_depths:
                for hd in hidden_dims:
                    for ed in embed_dims:
                        for sl in seq_lens:
                            for lr in lrs:
                                for bs in batch_sizes:
                                    for wd in weight_decays:
                                        for pooling in poolings:
                                            for lambda_eff in lambda_efficiency_list:
                                                for greedy_thresh in greedy_threshold_list:
                                                    for num_roll in num_rollouts_list:
                                                        for beta_ent in beta_entropy_list:
                                                            for beta_pol in beta_policy_list:
                                                                tag = _exp_tag(
                                                                    k, pooling, hd, ed, sl, lr, bs, wd, depth,
                                                                    lambda_eff, greedy_thresh, num_roll
                                                                )
                                                                cfg = _merge(
                                                                    base_common,
                                                                    tag=tag,
                                                                    batch_size=bs,
                                                                    hidden_dim=hd,
                                                                    embed_dim=ed,
                                                                    seq_len=sl,
                                                                    lr=lr,
                                                                    weight_decay=wd,
                                                                    max_depth=depth,
                                                                    max_children=k,
                                                                    pooling_mode=pooling,
                                                                    # v1.0.0 policy params
                                                                    num_rollouts=num_roll,
                                                                    lambda_efficiency=lambda_eff,
                                                                    beta_entropy=beta_ent,
                                                                    beta_policy=beta_pol,
                                                                    greedy_threshold=greedy_thresh,
                                                                )
                                                                exps.append(cfg)
    return exps


# -------------------------- Run recorders & helpers ------------------------- #

class RunRecorder:
    """Collects metrics for one training run (v1.0.0 Language Model)."""
    def __init__(self, run_id: int, tag: str):
        self.run_id = run_id
        self.tag = tag
        self.epochs_total: Optional[int] = None
        self.metrics_by_epoch: Dict[int, Dict[str, Any]] = {}
        self.best_val_ppl: float = float("inf")
        self.best_epoch: Optional[int] = None
        self.trainer_summary: Optional[Dict[str, Any]] = None

    def parse_v1_epoch_lines(self, text: str):
        """Parse v1.0.0 epoch output format for language model."""
        for m in RE_EPOCH_V1_LM.finditer(text):
            train_loss = float(m.group(1))
            lm_loss = float(m.group(2))
            policy_loss = float(m.group(3))
            val_loss = float(m.group(4))
            train_ppl = float(m.group(5))
            val_ppl = float(m.group(6))
            avg_nodes = float(m.group(7))
            
            e = len(self.metrics_by_epoch) + 1
            
            self.metrics_by_epoch[e] = dict(
                epoch=e,
                train_loss=train_loss,
                lm_loss=lm_loss,
                policy_loss=policy_loss,
                val_loss=val_loss,
                train_ppl=train_ppl,
                val_ppl=val_ppl,
                avg_nodes=avg_nodes,
            )
            
            if val_ppl < self.best_val_ppl:
                self.best_val_ppl = val_ppl
                self.best_epoch = e

    def parse_summary_json(self, line: str):
        m = RE_JSON_ANY.match(line.strip())
        if not m:
            return
        try:
            self.trainer_summary = json.loads(m.group(1))
        except Exception:
            pass

    def row_for_csv(self, static_cfg: Dict[str, Any], infer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        last_e = max(self.metrics_by_epoch.keys()) if self.metrics_by_epoch else None
        last = self.metrics_by_epoch.get(last_e, {}) if last_e else {}

        # Timing extraction from summary JSON
        avg_epoch_time: Optional[float] = None
        total_training_time: Optional[float] = None
        last_epoch_time: Optional[float] = None
        
        if self.trainer_summary is not None:
            time_data = self.trainer_summary.get("time")
            if isinstance(time_data, dict):
                avg_epoch_time = _safe_float(time_data.get("epoch_avg_s"))
                total_training_time = _safe_float(time_data.get("total_s"))
                last_epoch_time = _safe_float(time_data.get("last_epoch_s"))

        row: Dict[str, Any] = dict(
            run_id=self.run_id,
            tag=self.tag,
            epochs=self.epochs_total or len(self.metrics_by_epoch),
            val_ppl_last=last.get("val_ppl"),
            val_loss_last=last.get("val_loss"),
            train_ppl_last=last.get("train_ppl"),
            avg_nodes_last=last.get("avg_nodes"),
            policy_loss_last=last.get("policy_loss"),
            val_ppl_best=(self.best_val_ppl if self.best_val_ppl != float("inf") else None),
            best_epoch=self.best_epoch,
            total_training_time_sec=total_training_time,
            avg_epoch_time_sec=avg_epoch_time,
            last_epoch_time_sec=last_epoch_time,
        )

        # Add static config
        keep = [
            "batch_size", "hidden_dim", "embed_dim", "seq_len", "vocab_size",
            "max_depth", "max_children", "pooling_mode", "lr", "weight_decay", 
            "lr_schedule", "grad_clip", "opt", "dataset",
            # v1.0.0 policy params
            "num_rollouts", "lambda_efficiency", "beta_entropy", "beta_policy",
            "greedy_threshold",
            "seed",
        ]
        for k in keep:
            if k in static_cfg:
                row[k] = static_cfg[k]

        # Add inference results (perplexity instead of accuracy)
        if infer:
            row.update({
                "infer_val_ppl": infer.get("val_ppl"),
                "infer_val_loss": infer.get("val_loss"),
                "infer_avg_nodes": infer.get("avg_nodes_per_position"),
                "infer_sparsity_percent": infer.get("sparsity_percent"),
                "infer_latency_ms_mean": infer.get("latency_ms_mean"),
                "infer_latency_ms_p50": infer.get("latency_ms_p50"),
                "infer_latency_ms_p90": infer.get("latency_ms_p90"),
                "infer_latency_ms_p99": infer.get("latency_ms_p99"),
                "infer_device": infer.get("device"),
                "model_bytes": infer.get("model_bytes"),
                "checkpoint_path": infer.get("checkpoint_path"),
                # v1.0.0 policy analysis
                "infer_mean_grow_prob": infer.get("debug_policy_mean_grow_prob"),
                "infer_above_threshold_pct": infer.get("debug_policy_above_threshold_pct"),
            })
        else:
            row.update({
                "infer_val_ppl": None,
                "infer_val_loss": None,
                "infer_avg_nodes": None,
                "infer_sparsity_percent": None,
                "infer_latency_ms_mean": None,
                "infer_latency_ms_p50": None,
                "infer_latency_ms_p90": None,
                "infer_latency_ms_p99": None,
                "infer_device": None,
                "model_bytes": None,
                "checkpoint_path": static_cfg.get("save_path"),
                "infer_mean_grow_prob": None,
                "infer_above_threshold_pct": None,
            })

        return row


# ---------------------------- Subprocess runners ---------------------------- #

def _tee_process(cmd: List[str], log_file: Optional[Path] = None, 
                 env: Optional[Dict[str, str]] = None) -> Tuple[str, int]:
    """
    Runs a subprocess, tees stdout to console and an optional file.
    Returns (full_stdout_text, return_code).
    """
    lines: List[str] = []
    lf = log_file.open("w", encoding="utf-8") if log_file else None

    if log_file:
        print(f"[proc] {' '.join(shlex.quote(c) for c in cmd)}")
        print(f"[proc] logging -> {log_file}")

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        env=(env or os.environ.copy()),
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            lines.append(line)
            if lf:
                lf.write(line)
        ret = proc.wait()

    if lf:
        lf.flush()
        lf.close()

    if ret != 0:
        print(f"[proc][warn] Command exited with code {ret}")

    return "".join(lines), ret


def run_training(train_script: str, python_exe: str, run_dir: Path, 
                 run_id: int, cfg: Dict[str, Any]) -> Tuple[RunRecorder, Path, str, bool]:
    """Launch one trainer run; parse logs into a RunRecorder."""
    tag = cfg.get("tag", f"run_{run_id:03d}")
    log_path = run_dir / f"run_{run_id:03d}.log"
    ckpt_path = run_dir / f"{tag}.pt"

    args = dict(cfg)
    args["save_path"] = str(ckpt_path)

    cmd = [python_exe, "-u", train_script]
    for k, v in args.items():
        if k == "tag":
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]

    recorder = RunRecorder(run_id=run_id, tag=tag)

    stdout_text, ret = _tee_process(cmd, log_file=log_path)
    
    # Parse v1.0.0 format (language model)
    recorder.parse_v1_epoch_lines(stdout_text)
    
    # Parse summary JSON
    for line in stdout_text.split('\n'):
        recorder.parse_summary_json(line)

    success = (ret == 0) and os.path.isfile(ckpt_path)
    if not success:
        if ret != 0:
            print(f"[train][warn] Training returned non-zero code ({ret}).")
        if not os.path.isfile(ckpt_path):
            print(f"[train][warn] Checkpoint missing: {ckpt_path}")

    return recorder, log_path, str(ckpt_path), success


# ----------------------------- Inference helpers ---------------------------- #

def _parse_infer_json_from_stdout(text: str) -> Optional[Dict[str, Any]]:
    """Find the LAST JSON object printed on stdout; parse and map fields."""
    last_json_obj: Optional[Dict[str, Any]] = None

    for line in text.split('\n'):
        s = line.strip()
        m = RE_JSON_ANY.match(s)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    last_json_obj = obj
            except Exception:
                pass

    if last_json_obj is None:
        return None

    lk = _lower_keys(last_json_obj)

    # Map perplexity and loss (CHANGED from accuracy)
    val_ppl = lk.get("val_ppl", lk.get("perplexity"))
    val_loss = lk.get("val_loss")

    # Map latencies (unchanged)
    lmean = lk.get("latency_ms_mean", lk.get("mean_ms", lk.get("latency_mean_ms")))
    lp50 = lk.get("latency_ms_p50", lk.get("p50_ms", lk.get("latency_p50_ms")))
    lp90 = lk.get("latency_ms_p90", lk.get("p90_ms", lk.get("latency_p90_ms")))
    lp99 = lk.get("latency_ms_p99", lk.get("p99_ms", lk.get("latency_p99_ms")))

    # Map nodes (CHANGED from avg_nodes_per_example to avg_nodes_per_position)
    avg_nodes = lk.get("avg_nodes_per_position", lk.get("avg_nodes"))
    sparsity = lk.get("sparsity_percent")

    # v1.0.0 policy analysis
    mean_grow = lk.get("debug_policy_mean_grow_prob", lk.get("debug_policy_mean"))
    above_thresh = lk.get("debug_policy_above_threshold_pct")

    # Misc
    device = lk.get("device")
    n_samples = lk.get("num_samples")
    model_b = lk.get("model_bytes")

    norm = {
        "val_ppl": float(val_ppl) if val_ppl is not None else None,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "latency_ms_mean": float(lmean) if lmean is not None else None,
        "latency_ms_p50": float(lp50) if lp50 is not None else None,
        "latency_ms_p90": float(lp90) if lp90 is not None else None,
        "latency_ms_p99": float(lp99) if lp99 is not None else None,
        "avg_nodes_per_position": float(avg_nodes) if avg_nodes is not None else None,
        "sparsity_percent": float(sparsity) if sparsity is not None else None,
        "device": device,
        "num_samples": _safe_int(n_samples),
        "model_bytes": _safe_int(model_b),
        "debug_policy_mean_grow_prob": float(mean_grow) if mean_grow is not None else None,
        "debug_policy_above_threshold_pct": float(above_thresh) if above_thresh is not None else None,
    }
    return norm


def run_inference(
    infer_script: str,
    python_exe: str,
    run_dir: Path,
    run_id: int,
    checkpoint: str,
    dataset: str,
    infer_samples: int,
    cpu_only: bool,
    debug_policy: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path], Optional[Path]]:
    """Launch inference; parse JSON from stdout."""
    if not os.path.isfile(checkpoint):
        print("[infer][warn] Skipping inference; checkpoint not found:", checkpoint)
        return None, None, None

    cmd = [python_exe, "-u", infer_script,
           "--ckpt", checkpoint,
           "--dataset", dataset,
           "--samples", str(infer_samples)]
    
    if cpu_only:
        cmd.append("--cpu")
    if debug_policy:
        cmd.append("--debug_policy")

    infer_log = run_dir / f"infer_{run_id:03d}.log"
    stdout_text, _ = _tee_process(cmd, log_file=infer_log)

    parsed = _parse_infer_json_from_stdout(stdout_text)
    if parsed is None:
        # Fallback to text parsing for perplexity
        val_ppl: Optional[float] = None
        for line in stdout_text.split('\n'):
            m = RE_INFER_PPL_TEXT.search(line)
            if m:
                try:
                    val_ppl = float(m.group(1))
                    break
                except Exception:
                    pass
        if val_ppl is None:
            print("[infer][warn] Could not parse inference JSON or plain-text perplexity.")
            return None, None, infer_log
        parsed = {
            "val_ppl": val_ppl,
            "val_loss": None,
            "latency_ms_mean": None,
            "latency_ms_p50": None,
            "latency_ms_p90": None,
            "latency_ms_p99": None,
            "avg_nodes_per_position": None,
            "sparsity_percent": None,
            "device": "cpu" if cpu_only else None,
            "num_samples": infer_samples,
            "model_bytes": None,
            "debug_policy_mean_grow_prob": None,
            "debug_policy_above_threshold_pct": None,
        }

    parsed["checkpoint_path"] = checkpoint
    try:
        if parsed.get("model_bytes") in (None, 0):
            parsed["model_bytes"] = int(os.path.getsize(checkpoint))
    except Exception:
        pass

    json_path = (Path(run_dir) / f"infer_{run_id:03d}.json")
    try:
        with json_path.open("w", encoding="utf-8") as fj:
            json.dump(parsed, fj, ensure_ascii=False, indent=2)
    except Exception:
        json_path = None

    return parsed, json_path, infer_log


# --------------------------------- CLI ------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a full-factorial matrix of BoeNet v1.0.0 language model experiments."
    )
    # Config file
    p.add_argument("--config", type=str, default="configs/boenet-config.yaml",
                   help="Path to YAML config file")
    
    # Scripts
    p.add_argument("--train_script", type=str, default=None)
    p.add_argument("--infer_script", type=str, default=None)

    # Paths
    p.add_argument("--save_root", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None)

    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--epochs_list", type=str, default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--seed0", type=int, default=None)
    p.add_argument("--dataset", type=str, default=None,
                   help="Dataset (shakespeare, tinystories)")

    # Language Model Parameters (NEW)
    p.add_argument("--vocab_size", type=int, default=None)
    p.add_argument("--seq_len_list", type=str, default=None,
                   help="Comma-separated seq_len values (e.g., 64,128,256)")
    p.add_argument("--embed_dim_list", type=str, default=None,
                   help="Comma-separated embed_dim values (e.g., 32,64,128)")

    # v1.0.0 Policy parameters
    p.add_argument("--lambda_efficiency_list", type=str, default=None,
                   help="Comma-separated lambda values (e.g., 0.0,0.01,0.05,0.1)")
    p.add_argument("--greedy_threshold_list", type=str, default=None,
                   help="Comma-separated threshold values (e.g., 0.30,0.40,0.42,0.50)")
    p.add_argument("--num_rollouts_list", type=str, default=None)
    p.add_argument("--beta_entropy_list", type=str, default=None)
    p.add_argument("--beta_policy_list", type=str, default=None)

    # Architecture parameters
    p.add_argument("--k_values", type=str, default=None)
    p.add_argument("--poolings", type=str, default=None)
    p.add_argument("--hidden_dims", type=str, default=None)
    p.add_argument("--lrs", type=str, default=None)
    p.add_argument("--batch_sizes", type=str, default=None)
    p.add_argument("--weight_decays", type=str, default=None)
    p.add_argument("--max_depths", type=str, default=None)

    # Inference
    p.add_argument("--infer_samples", type=int, default=None)
    p.add_argument("--cpu_only", action="store_true", default=None)
    p.add_argument("--debug_policy", action="store_true", default=None)

    # Executable
    p.add_argument("--python", type=str, default=sys.executable)

    return p


# --------------------------------- main ------------------------------------ #

def main():
    args = build_arg_parser().parse_args()
    
    # Load config
    config = load_config(args.config)
    
    def get_val(cli_val, config_section, config_key, default):
        if cli_val is not None:
            return cli_val
        cfg_val = get_config_value(config, config_section, config_key, None)
        return cfg_val if cfg_val is not None else default
    
    def get_list_int(cli_str, config_section, config_key, default_str):
        if cli_str is not None:
            return _parse_list_int(cli_str)
        cfg_val = get_config_value(config, config_section, config_key, None)
        if cfg_val is not None:
            return [int(x) for x in cfg_val]
        return _parse_list_int(default_str)
    
    def get_list_float(cli_str, config_section, config_key, default_str):
        if cli_str is not None:
            return _parse_list_float(cli_str)
        cfg_val = get_config_value(config, config_section, config_key, None)
        if cfg_val is not None:
            return [float(x) for x in cfg_val]
        return _parse_list_float(default_str)
    
    def get_list_str(cli_str, config_section, config_key, default_str):
        if cli_str is not None:
            return [x.strip() for x in cli_str.split(',')]
        cfg_val = get_config_value(config, config_section, config_key, None)
        if cfg_val is not None:
            return [str(x) for x in cfg_val]
        return [x.strip() for x in default_str.split(',')]

    # Language Model Parameters (NEW)
    vocab_size = get_val(args.vocab_size, 'sweep', 'vocab_size', 256)
    seq_len_list = get_list_int(args.seq_len_list, 'sweep', 'seq_len_list', "128")
    embed_dim_list = get_list_int(args.embed_dim_list, 'sweep', 'embed_dim_list', "64")
    dataset = get_val(args.dataset, 'training', 'dataset', "shakespeare")

    # v1.0.0 Policy parameters
    lambda_efficiency_list = get_list_float(args.lambda_efficiency_list, 'sweep', 'lambda_efficiency_list', "0.05")
    greedy_threshold_list = get_list_float(args.greedy_threshold_list, 'sweep', 'greedy_threshold_list', "0.5")
    num_rollouts_list = get_list_int(args.num_rollouts_list, 'sweep', 'num_rollouts_list', "3")
    beta_entropy_list = get_list_float(args.beta_entropy_list, 'sweep', 'beta_entropy_list', "0.01")
    beta_policy_list = get_list_float(args.beta_policy_list, 'sweep', 'beta_policy_list', "0.5")

    # Architecture parameters
    k_values = get_list_int(args.k_values, 'sweep', 'k_values', "0,2,3")
    poolings = get_list_str(args.poolings, 'sweep', 'poolings', "mean")
    hidden_dims = get_list_int(args.hidden_dims, 'sweep', 'hidden_dims', "128")
    lrs = get_list_float(args.lrs, 'sweep', 'lrs', "0.001")
    batch_sizes = get_list_int(args.batch_sizes, 'sweep', 'batch_sizes', "64")
    weight_decays = get_list_float(args.weight_decays, 'sweep', 'weight_decays', "0.0")
    max_depths = get_list_int(args.max_depths, 'sweep', 'max_depths', "2")

    # Paths and training
    train_script = get_val(args.train_script, 'paths', 'train_script', "train_boenet.py")
    infer_script = get_val(args.infer_script, 'paths', 'infer_script', "infer_boenet.py")
    save_root = get_val(args.save_root, 'paths', 'save_root', "runs")
    data_root = get_val(args.data_root, 'paths', 'data_root', "./data")
    
    epochs = get_val(args.epochs, 'training', 'epochs', 10)
    repeats = get_val(args.repeats, 'training', 'repeats', 1)
    seed0 = get_val(args.seed0, 'training', 'seed0', 42)
    
    # Inference
    infer_samples = get_val(args.infer_samples, 'inference', 'infer_samples', 1000)
    cpu_only = get_val(args.cpu_only, 'inference', 'cpu_only', True)
    debug_policy = get_val(args.debug_policy, 'inference', 'debug_policy', True)

    # Timestamped output root
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(save_root) / ts
    out_root.mkdir(parents=True, exist_ok=True)

    # Build matrix
    matrix = build_factorial_matrix(
        k_values=k_values,
        poolings=poolings,
        hidden_dims=hidden_dims,
        embed_dims=embed_dim_list,
        seq_lens=seq_len_list,
        lrs=lrs,
        batch_sizes=batch_sizes,
        weight_decays=weight_decays,
        max_depths=max_depths,
        lambda_efficiency_list=lambda_efficiency_list,
        greedy_threshold_list=greedy_threshold_list,
        num_rollouts_list=num_rollouts_list,
        beta_entropy_list=beta_entropy_list,
        beta_policy_list=beta_policy_list,
        vocab_size=vocab_size,
        dataset=dataset,
    )

    # Handle epochs dimension
    epochs_list = []
    if args.epochs_list and args.epochs_list.strip():
        epochs_list = _parse_list_int(args.epochs_list)
    else:
        cfg_epochs_list = get_config_value(config, 'sweep', 'epochs_list', None)
        if cfg_epochs_list:
            epochs_list = [int(x) for x in cfg_epochs_list]
        else:
            epochs_list = [int(epochs)]

    # Expand matrix across epochs
    final_matrix: List[Dict[str, Any]] = []
    multi_epoch = len(epochs_list) > 1
    for m in matrix:
        for ep in epochs_list:
            cfg = dict(m)
            cfg["data_root"] = data_root
            cfg["epochs"] = int(ep)
            cfg["tag"] = m["tag"] + (f"_ep{ep}" if multi_epoch else "")
            final_matrix.append(cfg)

    # Summary
    k0_count = sum(1 for m in final_matrix if m.get("max_children", -1) == 0)
    kpos_count = len(final_matrix) - k0_count
    total_runs = len(final_matrix) * int(repeats)
    
    print(f"[matrix] BoeNet v1.0.0 Language Model Training Matrix")
    print(f"[matrix] Dataset: {dataset}")
    print(f"[matrix] Planned cells: {len(final_matrix)} | repeats: {repeats} | total: {total_runs}")
    print(f"[matrix] K=0 (dense): {k0_count} | K>0 (BFS): {kpos_count}")
    print(f"[matrix] Lambda sweep: {lambda_efficiency_list}")
    print(f"[matrix] Threshold sweep: {greedy_threshold_list}")
    print(f"[matrix] Seq len sweep: {seq_len_list}")
    print(f"[matrix] Embed dim sweep: {embed_dim_list}")

    # Prepare CSV/JSONL
    csv_path = out_root / "matrix_results.csv"
    jsonl_path = out_root / "matrix_results.jsonl"
    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    jsonl_f = jsonl_path.open("w", encoding="utf-8")

    # Column names (CHANGED: accuracy -> perplexity)
    base_cols = [
        "run_id", "tag", "epochs",
        "val_ppl_last", "val_loss_last", "train_ppl_last",
        "avg_nodes_last", "policy_loss_last",
        "val_ppl_best", "best_epoch",
        "total_training_time_sec", "avg_epoch_time_sec", "last_epoch_time_sec",
    ]
    keep_cols = [
        "batch_size", "hidden_dim", "embed_dim", "seq_len", "vocab_size",
        "max_depth", "max_children", "pooling_mode",
        "lr", "weight_decay", "lr_schedule", "grad_clip", "opt", "dataset",
        "num_rollouts", "lambda_efficiency", "beta_entropy", "beta_policy",
        "greedy_threshold", "seed",
    ]
    infer_cols = [
        "infer_val_ppl", "infer_val_loss", "infer_avg_nodes", "infer_sparsity_percent",
        "infer_latency_ms_mean", "infer_latency_ms_p50",
        "infer_latency_ms_p90", "infer_latency_ms_p99",
        "infer_device", "model_bytes", "checkpoint_path",
        "infer_mean_grow_prob", "infer_above_threshold_pct",
    ]
    fieldnames = base_cols + keep_cols + infer_cols
    csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Build run list
    all_runs: List[Tuple[Dict[str, Any], int]] = []
    for base_cfg in final_matrix:
        for rep in range(int(repeats)):
            all_runs.append((base_cfg, rep))

    # Progress
    run_counter = 0
    start_time = time.time()
    
    if _HAS_TQDM:
        progress_iter = tqdm(all_runs, desc="[matrix]", unit="run", ncols=120)
    else:
        progress_iter = all_runs
        print(f"\n[matrix] Starting {total_runs} runs...")

    try:
        for base_cfg, rep in progress_iter:
            run_cfg = dict(base_cfg)
            run_cfg["seed"] = int(seed0) + rep
            run_cfg["tag"] = f"{base_cfg['tag']}_rep{rep}"

            run_dir = out_root / run_cfg["tag"]
            run_dir.mkdir(parents=True, exist_ok=True)

            if _HAS_TQDM:
                short_tag = run_cfg["tag"]
                if len(short_tag) > 50:
                    short_tag = short_tag[:47] + "..."
                progress_iter.set_postfix_str(
                    f"lam={run_cfg.get('lambda_efficiency', '?')} "
                    f"thr={run_cfg.get('greedy_threshold', '?')}"
                )

            # Train
            recorder, train_log_path, ckpt_path, ok = run_training(
                train_script=train_script,
                python_exe=args.python,
                run_dir=run_dir,
                run_id=run_counter,
                cfg=run_cfg,
            )

            # Infer
            infer_obj = None
            if ok and os.path.isfile(ckpt_path):
                infer_obj, infer_json_path, infer_log_path = run_inference(
                    infer_script=infer_script,
                    python_exe=args.python,
                    run_dir=run_dir,
                    run_id=run_counter,
                    checkpoint=ckpt_path,
                    dataset=dataset,
                    infer_samples=int(infer_samples),
                    cpu_only=bool(cpu_only),
                    debug_policy=bool(debug_policy),
                )

            # Write row
            row = recorder.row_for_csv(static_cfg={**run_cfg, "save_path": ckpt_path}, infer=infer_obj)
            for k in fieldnames:
                row.setdefault(k, None)
            csv_writer.writerow(row)
            csv_f.flush()
            jsonl_f.write(json.dumps(row) + "\n")
            jsonl_f.flush()

            # Log (CHANGED: accuracy -> perplexity)
            print(f"[matrix] run {run_counter:03d} | lam={row.get('lambda_efficiency')} "
                  f"thr={row.get('greedy_threshold')} | "
                  f"best_ppl={row.get('val_ppl_best')} | "
                  f"infer_ppl={row.get('infer_val_ppl')} | "
                  f"nodes={row.get('infer_avg_nodes')} | "
                  f"mean_grow_p={row.get('infer_mean_grow_prob')}")

            run_counter += 1
            
    finally:
        try: csv_f.close()
        except Exception: pass
        try: jsonl_f.close()
        except Exception: pass

    total_elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("[matrix] COMPLETE")
    print("="*80)
    print(f"  Total runs:    {run_counter}")
    print(f"  Total time:    {_format_duration(total_elapsed)}")
    print(f"  CSV output:    {csv_path}")
    print(f"  JSONL output:  {jsonl_path}")
    print("="*80)


if __name__ == "__main__":
    main()