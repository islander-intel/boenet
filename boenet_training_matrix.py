#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet_training_matrix.py (v2.2.0 - Clean Progress Bars)

Run a *full factorial* matrix of BoeNet experiments on WikiText-2/Shakespeare/TinyStories.

v2.2.0 Changes (2026-01-03):
----------------------------
  MAJOR: Clean Console Progress Bars
  - Console shows clean progress bars with key metrics only
  - All verbose logs (rollout details, etc.) written to log files only
  - Per-epoch progress within each run
  - Summary metrics displayed after each epoch
  - Total matrix progress bar
  
  NEW FEATURES:
  - Subprocess output captured to file, not printed to console
  - Real-time epoch progress parsing from log file
  - Clean training header with model info
  - Duration estimates and ETA

v2.1.0 Critical Bug Fixes (2026-01-01):
---------------------------------------
  FIXED: CSV metric extraction from __SUMMARY__ JSON was completely broken
  - Previous version showed epochs=0 and empty metrics for all runs
  - Training was actually completing successfully but metrics weren't captured
  
  ROOT CAUSE:
    1. RE_JSON_ANY regex only matched lines starting with whitespace + JSON
    2. parse_summary_json() was being called line-by-line but couldn't find __SUMMARY__
    3. RunRecorder.epochs_total was never being set from summary
  
  FIXES APPLIED:
    1. Added RE_SUMMARY_JSON regex specifically for __SUMMARY__ prefix
    2. parse_summary_json() now properly extracts JSON from __SUMMARY__ line
    3. Row generation now falls back to trainer_summary if metrics_by_epoch is empty
    4. Added explicit extraction of epochs, best_val_loss, best_val_ppl from summary
    5. Added True BFS specific metrics: avg_nodes_per_position, avg_depth_reached

Author: BoeNet project
Version: 2.2.0
Date: 2026-01-03
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
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    print("[info] tqdm not installed. Install with: pip install tqdm")


# ------------------------------ Regex parsers ------------------------------ #

RE_EPOCH_V2_LM = re.compile(
    r"Train Loss:\s*([0-9.eE+-]+)\s*\(lm=([0-9.eE+-]+),\s*policy=([0-9.eE+-]+)\)\s*\|\s*"
    r"Val Loss:\s*([0-9.eE+-]+)\s*\n\s*"
    r"Train PPL:\s*([0-9.]+)\s*\|\s*Val PPL:\s*([0-9.]+)\s*\|\s*"
    r"Avg.*Nodes.*Position:\s*([0-9.]+)",
    re.MULTILINE
)

# v2.1.0 FIX: Regex for __SUMMARY__ JSON line
RE_SUMMARY_JSON = re.compile(r"__SUMMARY__\s*(\{.*\})\s*$")
RE_JSON_ANY = re.compile(r"^\s*(?:__SUMMARY__\s*)?(\{.*\})\s*$")
RE_INFER_PPL_TEXT = re.compile(r"Val PPL:\s*([0-9.]+)", re.IGNORECASE)
RE_EPOCH_NUM = re.compile(r"\[epoch\s*(\d+)/(\d+)\]", re.IGNORECASE)

# v2.2.0: Additional regex for progress parsing
RE_EPOCH_START = re.compile(r"\[epoch\s*(\d+)/(\d+)\]", re.IGNORECASE)
RE_TRAIN_LOSS = re.compile(r"Train Loss:\s*([0-9.eE+-]+)")
RE_VAL_LOSS = re.compile(r"Val Loss:\s*([0-9.eE+-]+)")
RE_VAL_PPL = re.compile(r"Val PPL:\s*([0-9.]+)")
RE_TRAIN_PPL = re.compile(r"Train PPL:\s*([0-9.]+)")
RE_STEP_PROGRESS = re.compile(r"\[step\s*(\d+)/(\d+)\]", re.IGNORECASE)
RE_BATCH_PROGRESS = re.compile(r"batch\s*(\d+)/(\d+)", re.IGNORECASE)


# --------------------------- Small parsing helpers -------------------------- #

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


def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in d.items()}


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
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


def _format_number(n: float, decimals: int = 2) -> str:
    """Format number with thousand separators."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    else:
        return f"{n:.{decimals}f}"


# --------------------------- Config file loading ---------------------------- #

def load_config(config_path: str) -> Dict[str, Any]:
    if not _HAS_YAML:
        print(f"[config] Warning: PyYAML not installed.")
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

def _exp_tag(k, pooling, hidden_dim, embed_dim, seq_len, lr, bs, wd, depth, 
             lambda_eff, greedy_thresh, num_rollouts, epochs) -> str:
    parts = [
        f"k{k}", f"pool{pooling}", f"hd{hidden_dim}", f"ed{embed_dim}",
        f"sl{seq_len}", f"lr{_slug(lr)}", f"bs{bs}", f"wd{_slug(wd)}",
        f"d{depth}", f"lam{_slug(lambda_eff)}", f"thr{_slug(greedy_thresh)}",
        f"roll{num_rollouts}", f"ep{epochs}",
    ]
    return "_".join(parts)


def build_factorial_matrix(
    k_values, poolings, hidden_dims, embed_dims, seq_lens, lrs, batch_sizes,
    weight_decays, max_depths, lambda_efficiency_list, greedy_threshold_list,
    num_rollouts_list, beta_entropy_list, beta_policy_list, epochs_list,
    vocab_size=256, dataset="wikitext2",
) -> List[Dict[str, Any]]:
    base_common = dict(
        lr_schedule="cosine", grad_clip=1.0, opt="adamw",
        vocab_size=vocab_size, dataset=dataset,
    )
    exps: List[Dict[str, Any]] = []

    for k in k_values:
        if k == 0:
            for hd in hidden_dims:
                for ed in embed_dims:
                    for sl in seq_lens:
                        for lr in lrs:
                            for bs in batch_sizes:
                                for wd in weight_decays:
                                    for ep in epochs_list:
                                        tag = _exp_tag(k, "mean", hd, ed, sl, lr, bs, wd, 1, 0.0, 0.5, 1, ep)
                                        cfg = _merge(
                                            base_common, tag=tag, batch_size=bs, hidden_dim=hd,
                                            embed_dim=ed, seq_len=sl, lr=lr, weight_decay=wd,
                                            max_depth=1, max_children=0, pooling_mode="mean",
                                            epochs=ep, num_rollouts=1, lambda_efficiency=0.0,
                                            beta_entropy=0.01, beta_policy=0.5, greedy_threshold=0.5,
                                        )
                                        exps.append(cfg)
        else:
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
                                                                for ep in epochs_list:
                                                                    tag = _exp_tag(
                                                                        k, pooling, hd, ed, sl, lr, bs, wd, depth,
                                                                        lambda_eff, greedy_thresh, num_roll, ep
                                                                    )
                                                                    cfg = _merge(
                                                                        base_common, tag=tag, batch_size=bs,
                                                                        hidden_dim=hd, embed_dim=ed, seq_len=sl,
                                                                        lr=lr, weight_decay=wd, max_depth=depth,
                                                                        max_children=k, pooling_mode=pooling,
                                                                        epochs=ep, num_rollouts=num_roll,
                                                                        lambda_efficiency=lambda_eff,
                                                                        beta_entropy=beta_ent, beta_policy=beta_pol,
                                                                        greedy_threshold=greedy_thresh,
                                                                    )
                                                                    exps.append(cfg)
    return exps


# =============================================================================
# v2.2.0: PROGRESS TRACKING CLASS
# =============================================================================

class TrainingProgress:
    """
    v2.2.0: Track training progress for clean console output.
    
    Parses log file in real-time to extract:
    - Current epoch
    - Current step/batch within epoch
    - Loss values
    - PPL values
    """
    
    def __init__(self, total_epochs: int, tag: str):
        self.total_epochs = total_epochs
        self.tag = tag
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 0
        self.train_loss: Optional[float] = None
        self.val_loss: Optional[float] = None
        self.train_ppl: Optional[float] = None
        self.val_ppl: Optional[float] = None
        self.best_val_ppl: float = float('inf')
        self.best_epoch: int = 0
        self.epoch_start_time: float = time.time()
        self.last_update_time: float = time.time()
        
    def parse_line(self, line: str):
        """Parse a single log line and update progress."""
        # Check for epoch start
        m = RE_EPOCH_START.search(line)
        if m:
            self.current_epoch = int(m.group(1))
            self.total_epochs = int(m.group(2))
            self.epoch_start_time = time.time()
            self.current_step = 0
            
        # Check for step/batch progress
        m = RE_STEP_PROGRESS.search(line)
        if m:
            self.current_step = int(m.group(1))
            self.total_steps = int(m.group(2))
        
        m = RE_BATCH_PROGRESS.search(line)
        if m:
            self.current_step = int(m.group(1))
            self.total_steps = int(m.group(2))
            
        # Check for train loss
        m = RE_TRAIN_LOSS.search(line)
        if m:
            self.train_loss = float(m.group(1))
            
        # Check for val loss
        m = RE_VAL_LOSS.search(line)
        if m:
            self.val_loss = float(m.group(1))
            
        # Check for train PPL
        m = RE_TRAIN_PPL.search(line)
        if m:
            self.train_ppl = float(m.group(1))
            
        # Check for val PPL
        m = RE_VAL_PPL.search(line)
        if m:
            self.val_ppl = float(m.group(1))
            if self.val_ppl < self.best_val_ppl:
                self.best_val_ppl = self.val_ppl
                self.best_epoch = self.current_epoch
                
        self.last_update_time = time.time()
    
    def get_epoch_progress_str(self) -> str:
        """Get formatted epoch progress string."""
        if self.total_steps > 0:
            pct = (self.current_step / self.total_steps) * 100
            bar_len = 20
            filled = int(bar_len * self.current_step / self.total_steps)
            bar = "━" * filled + "─" * (bar_len - filled)
            return f"[{bar}] {pct:5.1f}% {self.current_step}/{self.total_steps}"
        return ""
    
    def get_metrics_str(self) -> str:
        """Get formatted metrics string."""
        parts = []
        if self.train_loss is not None:
            parts.append(f"loss={self.train_loss:.4f}")
        if self.val_ppl is not None:
            parts.append(f"val_ppl={self.val_ppl:.1f}")
        if self.best_val_ppl < float('inf'):
            parts.append(f"best={self.best_val_ppl:.1f}")
        return " | ".join(parts) if parts else ""


class RunRecorder:
    """Collects metrics for one training run (v2.1.0)."""
    
    def __init__(self, run_id: int, tag: str):
        self.run_id = run_id
        self.tag = tag
        self.epochs_total: Optional[int] = None
        self.metrics_by_epoch: Dict[int, Dict[str, Any]] = {}
        self.best_val_ppl: float = float("inf")
        self.best_epoch: Optional[int] = None
        self.trainer_summary: Optional[Dict[str, Any]] = None
        self.last_seen_epoch: int = 0

    def parse_v2_epoch_lines(self, text: str):
        for m in RE_EPOCH_V2_LM.finditer(text):
            train_loss = float(m.group(1))
            lm_loss = float(m.group(2))
            policy_loss = float(m.group(3))
            val_loss = float(m.group(4))
            train_ppl = float(m.group(5))
            val_ppl = float(m.group(6))
            avg_nodes = float(m.group(7))
            
            e = len(self.metrics_by_epoch) + 1
            self.metrics_by_epoch[e] = dict(
                epoch=e, train_loss=train_loss, lm_loss=lm_loss,
                policy_loss=policy_loss, val_loss=val_loss,
                train_ppl=train_ppl, val_ppl=val_ppl, avg_nodes=avg_nodes,
            )
            if val_ppl < self.best_val_ppl:
                self.best_val_ppl = val_ppl
                self.best_epoch = e
        
        for m in RE_EPOCH_NUM.finditer(text):
            epoch_num = int(m.group(1))
            total_epochs = int(m.group(2))
            self.last_seen_epoch = max(self.last_seen_epoch, epoch_num)
            if self.epochs_total is None:
                self.epochs_total = total_epochs

    def parse_summary_json(self, text: str):
        """v2.1.0 FIX: Properly parse __SUMMARY__ JSON."""
        for line in text.split('\n'):
            line = line.strip()
            m = RE_SUMMARY_JSON.match(line)
            if m:
                try:
                    self.trainer_summary = json.loads(m.group(1))
                    return
                except json.JSONDecodeError as e:
                    continue
            if line.startswith("__SUMMARY__"):
                json_part = line[len("__SUMMARY__"):].strip()
                try:
                    self.trainer_summary = json.loads(json_part)
                    return
                except json.JSONDecodeError:
                    pass

    def row_for_csv(self, static_cfg: Dict[str, Any], infer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        last_e = max(self.metrics_by_epoch.keys()) if self.metrics_by_epoch else None
        last = self.metrics_by_epoch.get(last_e, {}) if last_e else {}

        # v2.1.0: Extract from trainer_summary
        epochs_from_summary = None
        best_val_loss_from_summary = None
        best_val_ppl_from_summary = None
        best_epoch_from_summary = None
        avg_nodes_from_summary = None
        avg_depth_from_summary = None
        sparsity_from_summary = None
        avg_epoch_time = None
        total_training_time = None
        last_epoch_time = None
        
        if self.trainer_summary is not None:
            run_data = self.trainer_summary.get("run", {})
            if isinstance(run_data, dict):
                epochs_from_summary = _safe_int(run_data.get("epochs"))
            
            results_data = self.trainer_summary.get("results", {})
            if isinstance(results_data, dict):
                best_val_loss_from_summary = _safe_float(results_data.get("best_val_loss"))
                best_val_ppl_from_summary = _safe_float(results_data.get("best_val_ppl"))
                best_epoch_from_summary = _safe_int(results_data.get("best_epoch"))
            
            true_bfs_data = self.trainer_summary.get("true_bfs", {})
            if isinstance(true_bfs_data, dict):
                avg_nodes_from_summary = _safe_float(true_bfs_data.get("avg_nodes_per_position"))
                avg_depth_from_summary = _safe_float(true_bfs_data.get("avg_depth_reached"))
                sparsity_from_summary = _safe_float(true_bfs_data.get("sparsity_pct"))
            
            time_data = self.trainer_summary.get("time", {})
            if isinstance(time_data, dict):
                avg_epoch_time = _safe_float(time_data.get("epoch_avg_s"))
                total_training_time = _safe_float(time_data.get("total_s"))
                last_epoch_time = _safe_float(time_data.get("last_epoch_s"))
            
            if epochs_from_summary is None:
                epochs_from_summary = _safe_int(self.trainer_summary.get("epochs"))
            if best_val_ppl_from_summary is None:
                best_val_ppl_from_summary = _safe_float(self.trainer_summary.get("best_val_ppl"))
            if best_epoch_from_summary is None:
                best_epoch_from_summary = _safe_int(self.trainer_summary.get("best_epoch"))

        final_epochs = epochs_from_summary or self.epochs_total or self.last_seen_epoch or len(self.metrics_by_epoch) or 0
        final_best_val_ppl = best_val_ppl_from_summary
        if final_best_val_ppl is None and self.best_val_ppl != float("inf"):
            final_best_val_ppl = self.best_val_ppl
        final_best_epoch = best_epoch_from_summary or self.best_epoch
        
        final_val_ppl_last = last.get("val_ppl")
        final_val_loss_last = last.get("val_loss")
        final_train_ppl_last = last.get("train_ppl")
        final_avg_nodes_last = last.get("avg_nodes")
        final_policy_loss_last = last.get("policy_loss")
        
        if final_val_ppl_last is None and final_best_val_ppl is not None:
            final_val_ppl_last = final_best_val_ppl
        if final_val_loss_last is None and best_val_loss_from_summary is not None:
            final_val_loss_last = best_val_loss_from_summary
        if final_avg_nodes_last is None and avg_nodes_from_summary is not None:
            final_avg_nodes_last = avg_nodes_from_summary

        row: Dict[str, Any] = dict(
            run_id=self.run_id, tag=self.tag, epochs=final_epochs,
            val_ppl_last=final_val_ppl_last, val_loss_last=final_val_loss_last,
            train_ppl_last=final_train_ppl_last, avg_nodes_last=final_avg_nodes_last,
            policy_loss_last=final_policy_loss_last, val_ppl_best=final_best_val_ppl,
            best_epoch=final_best_epoch, total_training_time_sec=total_training_time,
            avg_epoch_time_sec=avg_epoch_time, last_epoch_time_sec=last_epoch_time,
            avg_depth_reached=avg_depth_from_summary, sparsity_pct=sparsity_from_summary,
        )

        keep = [
            "batch_size", "hidden_dim", "embed_dim", "seq_len", "vocab_size",
            "max_depth", "max_children", "pooling_mode", "lr", "weight_decay", 
            "lr_schedule", "grad_clip", "opt", "dataset", "num_rollouts",
            "lambda_efficiency", "beta_entropy", "beta_policy", "greedy_threshold", "seed",
        ]
        for k in keep:
            if k in static_cfg:
                row[k] = static_cfg[k]

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
                "infer_mean_grow_prob": infer.get("debug_policy_mean_grow_prob"),
                "infer_above_threshold_pct": infer.get("debug_policy_above_threshold_pct"),
            })
        else:
            row.update({
                "infer_val_ppl": None, "infer_val_loss": None, "infer_avg_nodes": None,
                "infer_sparsity_percent": None, "infer_latency_ms_mean": None,
                "infer_latency_ms_p50": None, "infer_latency_ms_p90": None,
                "infer_latency_ms_p99": None, "infer_device": None, "model_bytes": None,
                "checkpoint_path": static_cfg.get("save_path"),
                "infer_mean_grow_prob": None, "infer_above_threshold_pct": None,
            })
        return row


# =============================================================================
# v2.2.0: CLEAN PROCESS EXECUTION WITH PROGRESS
# =============================================================================

def _run_process_with_progress(
    cmd: List[str], 
    log_file: Path,
    progress: TrainingProgress,
    env: Optional[Dict[str, str]] = None,
    update_interval: float = 0.5,
) -> Tuple[str, int]:
    """
    v2.2.0: Run subprocess, write output to log file, show clean progress.
    
    Console shows only progress bar and key metrics.
    All verbose output goes to log file only.
    """
    lines: List[str] = []
    
    with log_file.open("w", encoding="utf-8") as lf:
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True, env=(env or os.environ.copy()),
        ) as proc:
            assert proc.stdout is not None
            
            last_print_time = time.time()
            
            for line in proc.stdout:
                # Write to log file
                lf.write(line)
                lf.flush()
                lines.append(line)
                
                # Parse for progress updates
                progress.parse_line(line)
                
                # Update console periodically (not every line)
                current_time = time.time()
                if current_time - last_print_time >= update_interval:
                    _print_progress_line(progress)
                    last_print_time = current_time
            
            ret = proc.wait()
    
    # Final progress update
    _print_progress_line(progress, final=True)
    
    return "".join(lines), ret


def _print_progress_line(progress: TrainingProgress, final: bool = False):
    """Print a single progress line, overwriting the previous one."""
    if progress.total_epochs == 0:
        return
        
    # Build progress bar for epochs
    epoch_pct = (progress.current_epoch / progress.total_epochs) * 100 if progress.total_epochs > 0 else 0
    bar_len = 30
    filled = int(bar_len * progress.current_epoch / progress.total_epochs) if progress.total_epochs > 0 else 0
    bar = "━" * filled + "─" * (bar_len - filled)
    
    # Build metrics string
    metrics_parts = []
    if progress.train_loss is not None:
        metrics_parts.append(f"loss={progress.train_loss:.4f}")
    if progress.val_ppl is not None:
        metrics_parts.append(f"ppl={progress.val_ppl:.1f}")
    if progress.best_val_ppl < float('inf'):
        metrics_parts.append(f"best={progress.best_val_ppl:.1f}")
    
    metrics_str = " | ".join(metrics_parts) if metrics_parts else "starting..."
    
    # Calculate elapsed time
    elapsed = time.time() - progress.epoch_start_time
    elapsed_str = _format_duration(elapsed)
    
    # Build the line
    line = f"\r  Epoch {progress.current_epoch}/{progress.total_epochs} [{bar}] {metrics_str} [{elapsed_str}]"
    
    # Pad to overwrite previous content
    line = line.ljust(120)
    
    if final:
        print(line)
    else:
        print(line, end="", flush=True)


def _tee_process(cmd: List[str], log_file: Optional[Path] = None, 
                 env: Optional[Dict[str, str]] = None) -> Tuple[str, int]:
    """
    Original tee process for inference (keeps verbose output).
    v2.2.0: Still used for inference which doesn't need progress bars.
    """
    lines: List[str] = []
    lf = log_file.open("w", encoding="utf-8") if log_file else None

    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True, env=(env or os.environ.copy()),
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            # v2.2.0: Don't print to console for inference either
            lines.append(line)
            if lf:
                lf.write(line)
        ret = proc.wait()
    if lf:
        lf.flush()
        lf.close()
    return "".join(lines), ret


# =============================================================================
# TRAINING AND INFERENCE RUNNERS
# =============================================================================

def run_training(train_script: str, python_exe: str, run_dir: Path, 
                 run_id: int, cfg: Dict[str, Any], 
                 show_progress: bool = True) -> Tuple[RunRecorder, Path, str, bool]:
    """
    v2.2.0: Run training with clean progress output.
    """
    tag = cfg.get("tag", f"run_{run_id:03d}")
    log_path = run_dir / f"run_{run_id:03d}.log"
    ckpt_path = run_dir / f"{tag}.pt"

    args = dict(cfg)
    args["save_path"] = str(ckpt_path)
    cmd = [python_exe, "-u", train_script]
    skip_keys = {"tag", "vocab_size"}
    
    for k, v in args.items():
        if k in skip_keys:
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]

    recorder = RunRecorder(run_id=run_id, tag=tag)
    
    # v2.2.0: Use progress-aware process runner
    if show_progress:
        total_epochs = cfg.get("epochs", 10)
        progress = TrainingProgress(total_epochs=total_epochs, tag=tag)
        stdout_text, ret = _run_process_with_progress(cmd, log_path, progress)
    else:
        stdout_text, ret = _tee_process(cmd, log_file=log_path)
    
    recorder.parse_summary_json(stdout_text)
    recorder.parse_v2_epoch_lines(stdout_text)

    success = (ret == 0) and os.path.isfile(ckpt_path)

    return recorder, log_path, str(ckpt_path), success


def _parse_infer_json_from_stdout(text: str) -> Optional[Dict[str, Any]]:
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
    val_ppl = lk.get("val_ppl", lk.get("perplexity"))
    val_loss = lk.get("val_loss")
    lmean = lk.get("latency_ms_mean", lk.get("mean_ms", lk.get("latency_mean_ms")))
    lp50 = lk.get("latency_ms_p50", lk.get("p50_ms", lk.get("latency_p50_ms")))
    lp90 = lk.get("latency_ms_p90", lk.get("p90_ms", lk.get("latency_p90_ms")))
    lp99 = lk.get("latency_ms_p99", lk.get("p99_ms", lk.get("latency_p99_ms")))
    avg_nodes = lk.get("avg_nodes_per_position", lk.get("avg_nodes"))
    sparsity = lk.get("sparsity_percent")
    mean_grow = lk.get("debug_policy_mean_grow_prob", lk.get("debug_policy_mean"))
    above_thresh = lk.get("debug_policy_above_threshold_pct")
    device = lk.get("device")
    n_samples = lk.get("num_samples")
    model_b = lk.get("model_bytes")

    return {
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


def run_inference(
    infer_script: str, python_exe: str, run_dir: Path, run_id: int,
    checkpoint: str, dataset: str, infer_samples: int, cpu_only: bool, debug_policy: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path], Optional[Path]]:
    """Run inference on a checkpoint."""
    if not os.path.isfile(checkpoint):
        return None, None, None

    cmd = [python_exe, "-u", infer_script, "--ckpt", checkpoint,
           "--dataset", dataset, "--samples", str(infer_samples)]
    if cpu_only:
        cmd.append("--cpu")
    if debug_policy:
        cmd.append("--debug_policy")

    infer_log = run_dir / f"infer_{run_id:03d}.log"
    stdout_text, _ = _tee_process(cmd, log_file=infer_log)

    parsed = _parse_infer_json_from_stdout(stdout_text)
    if parsed is None:
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
            return None, None, infer_log
        parsed = {
            "val_ppl": val_ppl, "val_loss": None, "latency_ms_mean": None,
            "latency_ms_p50": None, "latency_ms_p90": None, "latency_ms_p99": None,
            "avg_nodes_per_position": None, "sparsity_percent": None,
            "device": "cpu" if cpu_only else None, "num_samples": infer_samples,
            "model_bytes": None, "debug_policy_mean_grow_prob": None,
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


# =============================================================================
# v2.2.0: HEADER AND SUMMARY PRINTING
# =============================================================================

def print_training_header(cfg: Dict[str, Any], run_id: int, total_runs: int):
    """v2.2.0: Print clean header before each training run."""
    tag = cfg.get("tag", f"run_{run_id:03d}")
    
    # Model info
    hidden_dim = cfg.get("hidden_dim", "?")
    embed_dim = cfg.get("embed_dim", "?")
    vocab_size = cfg.get("vocab_size", "?")
    max_depth = cfg.get("max_depth", "?")
    max_children = cfg.get("max_children", "?")
    epochs = cfg.get("epochs", "?")
    batch_size = cfg.get("batch_size", "?")
    lr = cfg.get("lr", "?")
    
    print()
    print("─" * 80)
    print(f"Run {run_id + 1}/{total_runs} | {tag}")
    print("─" * 80)
    print(f"  Model: hidden={hidden_dim}, embed={embed_dim}, vocab={_format_number(vocab_size) if isinstance(vocab_size, (int, float)) else vocab_size}")
    print(f"  Tree:  depth={max_depth}, children={max_children}")
    print(f"  Train: epochs={epochs}, batch={batch_size}, lr={lr}")
    print()


def print_run_summary(recorder: RunRecorder, infer_obj: Optional[Dict[str, Any]], duration: float):
    """v2.2.0: Print summary after each run."""
    best_ppl = recorder.best_val_ppl if recorder.best_val_ppl < float('inf') else None
    best_epoch = recorder.best_epoch
    infer_ppl = infer_obj.get("val_ppl") if infer_obj else None
    
    print()
    print(f"  ✓ Complete in {_format_duration(duration)}")
    if best_ppl is not None:
        print(f"    Train best: PPL={best_ppl:.2f} @ epoch {best_epoch}")
    if infer_ppl is not None:
        print(f"    Inference:  PPL={infer_ppl:.2f}")
    print()


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BoeNet v2.2.0 Training Matrix - Clean Progress")
    p.add_argument("--config", type=str, default="configs/experiment-config.yaml")
    p.add_argument("--train_script", type=str, default=None)
    p.add_argument("--infer_script", type=str, default=None)
    p.add_argument("--save_root", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--epochs_list", type=str, default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--seed0", type=int, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--vocab_size", type=int, default=None)
    p.add_argument("--seq_len_list", type=str, default=None)
    p.add_argument("--embed_dim_list", type=str, default=None)
    p.add_argument("--lambda_efficiency_list", type=str, default=None)
    p.add_argument("--greedy_threshold_list", type=str, default=None)
    p.add_argument("--num_rollouts_list", type=str, default=None)
    p.add_argument("--beta_entropy_list", type=str, default=None)
    p.add_argument("--beta_policy_list", type=str, default=None)
    p.add_argument("--k_values", type=str, default=None)
    p.add_argument("--poolings", type=str, default=None)
    p.add_argument("--hidden_dims", type=str, default=None)
    p.add_argument("--lrs", type=str, default=None)
    p.add_argument("--batch_sizes", type=str, default=None)
    p.add_argument("--weight_decays", type=str, default=None)
    p.add_argument("--max_depths", type=str, default=None)
    p.add_argument("--infer_samples", type=int, default=None)
    p.add_argument("--cpu_only", action="store_true", default=None)
    p.add_argument("--debug_policy", action="store_true", default=None)
    p.add_argument("--python", type=str, default=sys.executable)
    # v2.2.0: New options
    p.add_argument("--verbose", action="store_true", help="Show verbose output (disables clean progress)")
    return p


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = build_arg_parser().parse_args()
    
    # v2.2.0: Clean header
    print()
    print("═" * 80)
    print("  BoeNet Training Matrix v2.2.0 - Clean Progress")
    print("═" * 80)
    print()
    
    config = load_config(args.config)
    
    def get_val(cli_val, section, key, default):
        if cli_val is not None:
            return cli_val
        return get_config_value(config, section, key, None) or default
    
    def get_list_int_cfg(cli_str, section, key, default_str):
        if cli_str is not None:
            return _parse_list_int(cli_str)
        cfg_val = get_config_value(config, section, key, None)
        return [int(x) for x in cfg_val] if cfg_val else _parse_list_int(default_str)
    
    def get_list_float_cfg(cli_str, section, key, default_str):
        if cli_str is not None:
            return _parse_list_float(cli_str)
        cfg_val = get_config_value(config, section, key, None)
        return [float(x) for x in cfg_val] if cfg_val else _parse_list_float(default_str)
    
    def get_list_str_cfg(cli_str, section, key, default_str):
        if cli_str is not None:
            return [x.strip() for x in cli_str.split(',')]
        cfg_val = get_config_value(config, section, key, None)
        return [str(x) for x in cfg_val] if cfg_val else [x.strip() for x in default_str.split(',')]

    vocab_size = get_val(args.vocab_size, 'sweep', 'vocab_size', 256)
    seq_len_list = get_list_int_cfg(args.seq_len_list, 'sweep', 'seq_len_list', "128")
    embed_dim_list = get_list_int_cfg(args.embed_dim_list, 'sweep', 'embed_dim_list', "64")
    dataset = get_val(args.dataset, 'training', 'dataset', "wikitext2")
    lambda_efficiency_list = get_list_float_cfg(args.lambda_efficiency_list, 'sweep', 'lambda_efficiency_list', "0.05")
    greedy_threshold_list = get_list_float_cfg(args.greedy_threshold_list, 'sweep', 'greedy_threshold_list', "0.5")
    num_rollouts_list = get_list_int_cfg(args.num_rollouts_list, 'sweep', 'num_rollouts_list', "3")
    beta_entropy_list = get_list_float_cfg(args.beta_entropy_list, 'sweep', 'beta_entropy_list', "0.01")
    beta_policy_list = get_list_float_cfg(args.beta_policy_list, 'sweep', 'beta_policy_list', "0.5")
    k_values = get_list_int_cfg(args.k_values, 'sweep', 'k_values', "0,2,3")
    poolings = get_list_str_cfg(args.poolings, 'sweep', 'poolings', "mean")
    hidden_dims = get_list_int_cfg(args.hidden_dims, 'sweep', 'hidden_dims', "128")
    lrs = get_list_float_cfg(args.lrs, 'sweep', 'lrs', "0.001")
    batch_sizes = get_list_int_cfg(args.batch_sizes, 'sweep', 'batch_sizes', "64")
    weight_decays = get_list_float_cfg(args.weight_decays, 'sweep', 'weight_decays', "0.0")
    max_depths = get_list_int_cfg(args.max_depths, 'sweep', 'max_depths', "2")
    train_script = get_val(args.train_script, 'paths', 'train_script', "train_boenet.py")
    infer_script = get_val(args.infer_script, 'paths', 'infer_script', "infer_boenet.py")
    save_root = get_val(args.save_root, 'paths', 'save_root', "runs")
    epochs = get_val(args.epochs, 'training', 'epochs', 10)
    repeats = get_val(args.repeats, 'training', 'repeats', 1)
    seed0 = get_val(args.seed0, 'training', 'seed0', 42)
    infer_samples = get_val(args.infer_samples, 'inference', 'infer_samples', 1000)
    cpu_only = get_val(args.cpu_only, 'inference', 'cpu_only', True)
    debug_policy = get_val(args.debug_policy, 'inference', 'debug_policy', True)

    epochs_list = []
    if args.epochs_list and args.epochs_list.strip():
        epochs_list = _parse_list_int(args.epochs_list)
    else:
        cfg_epochs_list = get_config_value(config, 'sweep', 'epochs_list', None)
        epochs_list = [int(x) for x in cfg_epochs_list] if cfg_epochs_list else [int(epochs)]

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(save_root) / ts
    out_root.mkdir(parents=True, exist_ok=True)

    matrix = build_factorial_matrix(
        k_values=k_values, poolings=poolings, hidden_dims=hidden_dims,
        embed_dims=embed_dim_list, seq_lens=seq_len_list, lrs=lrs,
        batch_sizes=batch_sizes, weight_decays=weight_decays, max_depths=max_depths,
        lambda_efficiency_list=lambda_efficiency_list,
        greedy_threshold_list=greedy_threshold_list,
        num_rollouts_list=num_rollouts_list, beta_entropy_list=beta_entropy_list,
        beta_policy_list=beta_policy_list, epochs_list=epochs_list,
        vocab_size=vocab_size, dataset=dataset,
    )

    k0_count = sum(1 for m in matrix if m.get("max_children", -1) == 0)
    kpos_count = len(matrix) - k0_count
    total_runs = len(matrix) * int(repeats)
    
    # v2.2.0: Clean config summary
    print(f"  Dataset:    {dataset}")
    print(f"  Vocab:      {_format_number(vocab_size) if isinstance(vocab_size, (int, float)) else vocab_size}")
    print(f"  Configs:    {len(matrix)} (K=0: {k0_count}, K>0: {kpos_count})")
    print(f"  Repeats:    {repeats}")
    print(f"  Total runs: {total_runs}")
    print(f"  Output:     {out_root}")
    print()

    csv_path = out_root / "matrix_results.csv"
    jsonl_path = out_root / "matrix_results.jsonl"
    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    jsonl_f = jsonl_path.open("w", encoding="utf-8")

    base_cols = [
        "run_id", "tag", "epochs", "val_ppl_last", "val_loss_last", "train_ppl_last",
        "avg_nodes_last", "policy_loss_last", "val_ppl_best", "best_epoch",
        "total_training_time_sec", "avg_epoch_time_sec", "last_epoch_time_sec",
        "avg_depth_reached", "sparsity_pct",
    ]
    keep_cols = [
        "batch_size", "hidden_dim", "embed_dim", "seq_len", "vocab_size",
        "max_depth", "max_children", "pooling_mode", "lr", "weight_decay",
        "lr_schedule", "grad_clip", "opt", "dataset", "num_rollouts",
        "lambda_efficiency", "beta_entropy", "beta_policy", "greedy_threshold", "seed",
    ]
    infer_cols = [
        "infer_val_ppl", "infer_val_loss", "infer_avg_nodes", "infer_sparsity_percent",
        "infer_latency_ms_mean", "infer_latency_ms_p50", "infer_latency_ms_p90",
        "infer_latency_ms_p99", "infer_device", "model_bytes", "checkpoint_path",
        "infer_mean_grow_prob", "infer_above_threshold_pct",
    ]
    fieldnames = base_cols + keep_cols + infer_cols
    csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    csv_writer.writeheader()

    all_runs = [(base_cfg, rep) for base_cfg in matrix for rep in range(int(repeats))]
    run_counter = 0
    start_time = time.time()
    completed_runs = 0
    failed_runs = 0

    try:
        for base_cfg, rep in all_runs:
            run_cfg = dict(base_cfg)
            run_cfg["seed"] = int(seed0) + rep
            run_cfg["tag"] = f"{base_cfg['tag']}_rep{rep}"
            run_dir = out_root / run_cfg["tag"]
            run_dir.mkdir(parents=True, exist_ok=True)

            # v2.2.0: Print clean header
            print_training_header(run_cfg, run_counter, total_runs)
            
            run_start = time.time()
            
            # Run training with progress
            recorder, _, ckpt_path, ok = run_training(
                train_script=train_script, python_exe=args.python,
                run_dir=run_dir, run_id=run_counter, cfg=run_cfg,
                show_progress=not args.verbose,
            )

            # Run inference (silent)
            infer_obj = None
            if ok and os.path.isfile(ckpt_path):
                print("  Running inference...", end="", flush=True)
                infer_obj, _, _ = run_inference(
                    infer_script=infer_script, python_exe=args.python,
                    run_dir=run_dir, run_id=run_counter, checkpoint=ckpt_path,
                    dataset=dataset, infer_samples=int(infer_samples),
                    cpu_only=bool(cpu_only), debug_policy=bool(debug_policy),
                )
                print(" done")

            run_duration = time.time() - run_start
            
            # v2.2.0: Print summary
            if ok:
                completed_runs += 1
                print_run_summary(recorder, infer_obj, run_duration)
            else:
                failed_runs += 1
                print(f"\n  ✗ Failed (see log: {run_dir}/run_{run_counter:03d}.log)\n")

            # Write results
            row = recorder.row_for_csv({**run_cfg, "save_path": ckpt_path}, infer_obj)
            for k in fieldnames:
                row.setdefault(k, None)
            csv_writer.writerow(row)
            csv_f.flush()
            jsonl_f.write(json.dumps(row) + "\n")
            jsonl_f.flush()

            run_counter += 1
            
            # v2.2.0: Progress summary
            elapsed = time.time() - start_time
            avg_per_run = elapsed / run_counter
            remaining = avg_per_run * (total_runs - run_counter)
            print(f"  Progress: {run_counter}/{total_runs} | "
                  f"Elapsed: {_format_duration(elapsed)} | "
                  f"ETA: {_format_duration(remaining)}")
            
    finally:
        csv_f.close()
        jsonl_f.close()

    # v2.2.0: Final summary
    total_duration = time.time() - start_time
    print()
    print("═" * 80)
    print("  TRAINING MATRIX COMPLETE")
    print("═" * 80)
    print(f"  Total runs:  {run_counter}")
    print(f"  Completed:   {completed_runs}")
    print(f"  Failed:      {failed_runs}")
    print(f"  Duration:    {_format_duration(total_duration)}")
    print(f"  Results:     {csv_path}")
    print("═" * 80)
    print()


if __name__ == "__main__":
    main()