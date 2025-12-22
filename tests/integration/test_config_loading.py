# ============================================================================
# BFSNet Configuration Loading Tests
# ============================================================================
# Tests for configuration file parsing and validation.
#
# These tests verify:
#   - YAML config files can be loaded correctly
#   - Config values are parsed with correct types
#   - Default values are applied when missing
#   - CLI overrides work correctly
#   - Invalid configs are rejected with helpful errors
#   - Config inheritance and merging works
#
# Configuration files are YAML-based and control:
#   - K values to sweep (k_values)
#   - Model architecture (max_depth, hidden_dim, pooling)
#   - Temperature schedules (temp_schedules, temp_fixed, temp_anneal)
#   - Training parameters (epochs, lr, batch_size, weight_decay)
#   - Experiment settings (num_repeats, warmup_epochs)
#
# Run:
#   pytest tests/integration/test_config_loading.py -v
# ============================================================================

import pytest
import yaml
import tempfile
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Default Configuration Schema
# ============================================================================

DEFAULT_CONFIG = {
    # K values (branching factor)
    'k_values': [0, 1, 2, 3, 4, 5],
    
    # Model architecture
    'max_depths': [2],
    'hidden_dims': [128],
    'poolings': ['learned'],
    
    # Temperature schedules
    'temp_schedules': ['none', 'cosine'],
    'temp_fixed': [1.0],
    'temp_anneal': ['1.5->0.5'],
    
    # Training parameters
    'lrs': [0.001],
    'batch_sizes': [64],
    'weight_decays': [0.0],
    
    # Experiment settings
    'num_epochs': 15,
    'num_repeats': 3,
    'warmup_epochs': [3],
    
    # Device and seed
    'device': 'auto',
    'seed': 42,
}

REQUIRED_KEYS = ['k_values']

LIST_KEYS = [
    'k_values', 'max_depths', 'hidden_dims', 'poolings',
    'temp_schedules', 'temp_fixed', 'temp_anneal',
    'lrs', 'batch_sizes', 'weight_decays', 'warmup_epochs'
]

SCALAR_KEYS = ['num_epochs', 'num_repeats', 'device', 'seed']


# ============================================================================
# Helper Functions
# ============================================================================

def create_config_file(config: Dict[str, Any], filepath: Path) -> None:
    """Create a YAML config file."""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.debug(f"Created config file: {filepath}")


def load_config_file(filepath: Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    logger.debug(f"Loaded config from: {filepath}")
    return config if config is not None else {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configs, with override taking precedence."""
    result = base.copy()
    result.update(override)
    return result


def apply_defaults(config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default values for missing keys."""
    result = defaults.copy()
    result.update(config)
    return result


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required keys
    for key in REQUIRED_KEYS:
        if key not in config:
            errors.append(f"Missing required key: {key}")
    
    # Validate k_values
    if 'k_values' in config:
        k_values = config['k_values']
        if not isinstance(k_values, list):
            errors.append(f"k_values must be a list, got {type(k_values).__name__}")
        else:
            for k in k_values:
                if not isinstance(k, int) or k < 0:
                    errors.append(f"Invalid k value: {k} (must be non-negative integer)")
    
    # Validate list keys
    for key in LIST_KEYS:
        if key in config and not isinstance(config[key], list):
            errors.append(f"{key} must be a list, got {type(config[key]).__name__}")
    
    # Validate scalar keys
    if 'num_epochs' in config:
        if not isinstance(config['num_epochs'], int) or config['num_epochs'] < 1:
            errors.append(f"num_epochs must be positive integer, got {config['num_epochs']}")
    
    if 'num_repeats' in config:
        if not isinstance(config['num_repeats'], int) or config['num_repeats'] < 1:
            errors.append(f"num_repeats must be positive integer, got {config['num_repeats']}")
    
    # Validate pooling values
    if 'poolings' in config:
        valid_poolings = {'learned', 'sum', 'mean'}
        for p in config['poolings']:
            if p not in valid_poolings:
                errors.append(f"Invalid pooling: {p}. Must be one of {valid_poolings}")
    
    # Validate temp_schedules
    if 'temp_schedules' in config:
        valid_schedules = {'none', 'cosine', 'fixed', 'anneal'}
        for s in config['temp_schedules']:
            if s not in valid_schedules:
                errors.append(f"Invalid temp_schedule: {s}. Must be one of {valid_schedules}")
    
    # Validate learning rates
    if 'lrs' in config:
        for lr in config['lrs']:
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append(f"Invalid learning rate: {lr} (must be positive)")
    
    # Validate batch sizes
    if 'batch_sizes' in config:
        for bs in config['batch_sizes']:
            if not isinstance(bs, int) or bs < 1:
                errors.append(f"Invalid batch_size: {bs} (must be positive integer)")
    
    return errors


def parse_cli_overrides(args: List[str]) -> Dict[str, Any]:
    """
    Parse CLI arguments as config overrides.
    
    Supports format: --key value or --key=value
    """
    overrides = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--'):
            if '=' in arg:
                key, value = arg[2:].split('=', 1)
            else:
                key = arg[2:]
                i += 1
                if i < len(args):
                    value = args[i]
                else:
                    value = None
            
            # Parse value type
            if value is not None:
                # Try to parse as list
                if value.startswith('[') and value.endswith(']'):
                    try:
                        overrides[key] = yaml.safe_load(value)
                    except:
                        overrides[key] = value
                # Try to parse as number
                else:
                    try:
                        if '.' in value:
                            overrides[key] = float(value)
                        else:
                            overrides[key] = int(value)
                    except ValueError:
                        overrides[key] = value
        i += 1
    
    return overrides


def count_experiment_combinations(config: Dict[str, Any]) -> int:
    """Count total number of experiment combinations from config."""
    count = 1
    
    for key in LIST_KEYS:
        if key in config and isinstance(config[key], list):
            count *= len(config[key])
    
    if 'num_repeats' in config:
        count *= config['num_repeats']
    
    return count


# ============================================================================
# Basic Config Loading Tests
# ============================================================================

class TestBasicConfigLoading:
    """Basic tests for config file loading."""
    
    @pytest.mark.integration
    def test_load_empty_config(self, tmp_path):
        """Test loading an empty config file."""
        config_path = tmp_path / "config.yaml"
        
        with open(config_path, 'w') as f:
            f.write("")
        
        config = load_config_file(config_path)
        
        assert config == {} or config is None or config == {}
        logger.info("Empty config loaded successfully")
    
    @pytest.mark.integration
    def test_load_minimal_config(self, tmp_path):
        """Test loading a minimal config with required keys only."""
        config_path = tmp_path / "config.yaml"
        
        minimal_config = {'k_values': [0, 2]}
        create_config_file(minimal_config, config_path)
        
        config = load_config_file(config_path)
        
        assert config['k_values'] == [0, 2]
        
        errors = validate_config(config)
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        
        logger.info("Minimal config loaded successfully")
    
    @pytest.mark.integration
    def test_load_full_config(self, tmp_path):
        """Test loading a full config with all options."""
        config_path = tmp_path / "config.yaml"
        
        full_config = {
            'k_values': [0, 1, 2, 3],
            'max_depths': [2, 3],
            'hidden_dims': [64, 128],
            'poolings': ['learned', 'mean'],
            'temp_schedules': ['none', 'cosine'],
            'temp_fixed': [1.0, 1.2],
            'temp_anneal': ['1.5->0.5'],
            'lrs': [0.001, 0.01],
            'batch_sizes': [32, 64],
            'weight_decays': [0.0, 0.01],
            'num_epochs': 15,
            'num_repeats': 3,
            'warmup_epochs': [3, 5],
            'device': 'auto',
            'seed': 42,
        }
        
        create_config_file(full_config, config_path)
        
        config = load_config_file(config_path)
        
        assert config['k_values'] == [0, 1, 2, 3]
        assert config['num_epochs'] == 15
        assert config['seed'] == 42
        
        errors = validate_config(config)
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        
        logger.info("Full config loaded successfully")
    
    @pytest.mark.integration
    def test_config_preserves_types(self, tmp_path):
        """Test that config values preserve their types."""
        config_path = tmp_path / "config.yaml"
        
        config = {
            'k_values': [0, 1, 2],  # List of ints
            'lrs': [0.001, 0.01],   # List of floats
            'poolings': ['learned', 'mean'],  # List of strings
            'num_epochs': 15,       # Scalar int
            'device': 'cuda',       # Scalar string
        }
        
        create_config_file(config, config_path)
        loaded = load_config_file(config_path)
        
        assert isinstance(loaded['k_values'], list)
        assert all(isinstance(k, int) for k in loaded['k_values'])
        
        assert isinstance(loaded['lrs'], list)
        assert all(isinstance(lr, float) for lr in loaded['lrs'])
        
        assert isinstance(loaded['poolings'], list)
        assert all(isinstance(p, str) for p in loaded['poolings'])
        
        assert isinstance(loaded['num_epochs'], int)
        assert isinstance(loaded['device'], str)
        
        logger.info("Config types preserved correctly")


# ============================================================================
# Config Validation Tests
# ============================================================================

class TestConfigValidation:
    """Tests for config validation."""
    
    @pytest.mark.integration
    def test_missing_required_key(self, tmp_path):
        """Test that missing required keys are detected."""
        config_path = tmp_path / "config.yaml"
        
        # Missing k_values
        config = {'num_epochs': 10}
        create_config_file(config, config_path)
        
        loaded = load_config_file(config_path)
        errors = validate_config(loaded)
        
        assert any('k_values' in e for e in errors), "Should detect missing k_values"
        
        logger.info("Missing required key detected")
    
    @pytest.mark.integration
    def test_invalid_k_values(self, tmp_path):
        """Test that invalid K values are detected."""
        config_path = tmp_path / "config.yaml"
        
        # Negative K value
        config = {'k_values': [-1, 0, 2]}
        create_config_file(config, config_path)
        
        loaded = load_config_file(config_path)
        errors = validate_config(loaded)
        
        assert any('-1' in e or 'Invalid k' in e for e in errors), \
            "Should detect negative K value"
        
        logger.info("Invalid K values detected")
    
    @pytest.mark.integration
    def test_invalid_pooling(self, tmp_path):
        """Test that invalid pooling values are detected."""
        config_path = tmp_path / "config.yaml"
        
        config = {
            'k_values': [0, 2],
            'poolings': ['learned', 'invalid_pooling']
        }
        create_config_file(config, config_path)
        
        loaded = load_config_file(config_path)
        errors = validate_config(loaded)
        
        assert any('invalid_pooling' in e or 'Invalid pooling' in e for e in errors), \
            "Should detect invalid pooling"
        
        logger.info("Invalid pooling detected")
    
    @pytest.mark.integration
    def test_invalid_temp_schedule(self, tmp_path):
        """Test that invalid temperature schedules are detected."""
        config_path = tmp_path / "config.yaml"
        
        config = {
            'k_values': [0, 2],
            'temp_schedules': ['none', 'invalid_schedule']
        }
        create_config_file(config, config_path)
        
        loaded = load_config_file(config_path)
        errors = validate_config(loaded)
        
        assert any('invalid_schedule' in e or 'Invalid temp_schedule' in e for e in errors), \
            "Should detect invalid temp_schedule"
        
        logger.info("Invalid temp_schedule detected")
    
    @pytest.mark.integration
    def test_wrong_type_for_list_key(self, tmp_path):
        """Test that wrong types for list keys are detected."""
        config_path = tmp_path / "config.yaml"
        
        config = {
            'k_values': 2,  # Should be list, not int
        }
        create_config_file(config, config_path)
        
        loaded = load_config_file(config_path)
        errors = validate_config(loaded)
        
        assert any('list' in e.lower() for e in errors), \
            "Should detect wrong type for k_values"
        
        logger.info("Wrong type for list key detected")
    
    @pytest.mark.integration
    def test_invalid_num_epochs(self, tmp_path):
        """Test that invalid num_epochs is detected."""
        config_path = tmp_path / "config.yaml"
        
        config = {
            'k_values': [0, 2],
            'num_epochs': -5
        }
        create_config_file(config, config_path)
        
        loaded = load_config_file(config_path)
        errors = validate_config(loaded)
        
        assert any('num_epochs' in e for e in errors), \
            "Should detect invalid num_epochs"
        
        logger.info("Invalid num_epochs detected")


# ============================================================================
# Default Value Tests
# ============================================================================

class TestDefaultValues:
    """Tests for default value application."""
    
    @pytest.mark.integration
    def test_defaults_applied_to_minimal_config(self, tmp_path):
        """Test that defaults are applied to minimal config."""
        config_path = tmp_path / "config.yaml"
        
        minimal = {'k_values': [0, 2]}
        create_config_file(minimal, config_path)
        
        loaded = load_config_file(config_path)
        full = apply_defaults(loaded, DEFAULT_CONFIG)
        
        assert full['k_values'] == [0, 2]  # From config
        assert 'num_epochs' in full  # From defaults
        assert 'poolings' in full
        
        logger.info("Defaults applied correctly")
    
    @pytest.mark.integration
    def test_config_overrides_defaults(self, tmp_path):
        """Test that config values override defaults."""
        config_path = tmp_path / "config.yaml"
        
        config = {
            'k_values': [0, 2],
            'num_epochs': 30,  # Override default
            'lrs': [0.1],      # Override default
        }
        create_config_file(config, config_path)
        
        loaded = load_config_file(config_path)
        full = apply_defaults(loaded, DEFAULT_CONFIG)
        
        assert full['num_epochs'] == 30
        assert full['lrs'] == [0.1]
        
        logger.info("Config correctly overrides defaults")
    
    @pytest.mark.integration
    def test_all_defaults_have_valid_types(self):
        """Test that all default values are valid."""
        errors = validate_config(DEFAULT_CONFIG)
        
        assert len(errors) == 0, f"Default config has errors: {errors}"
        
        logger.info("Default config is valid")


# ============================================================================
# CLI Override Tests
# ============================================================================

class TestCLIOverrides:
    """Tests for CLI argument overrides."""
    
    @pytest.mark.integration
    def test_parse_single_override(self):
        """Test parsing a single CLI override."""
        args = ['--num_epochs', '20']
        overrides = parse_cli_overrides(args)
        
        assert overrides['num_epochs'] == 20
        
        logger.info("Single CLI override parsed")
    
    @pytest.mark.integration
    def test_parse_multiple_overrides(self):
        """Test parsing multiple CLI overrides."""
        args = ['--num_epochs', '20', '--seed', '123', '--device', 'cpu']
        overrides = parse_cli_overrides(args)
        
        assert overrides['num_epochs'] == 20
        assert overrides['seed'] == 123
        assert overrides['device'] == 'cpu'
        
        logger.info("Multiple CLI overrides parsed")
    
    @pytest.mark.integration
    def test_parse_equals_syntax(self):
        """Test parsing --key=value syntax."""
        args = ['--num_epochs=20', '--device=cuda']
        overrides = parse_cli_overrides(args)
        
        assert overrides['num_epochs'] == 20
        assert overrides['device'] == 'cuda'
        
        logger.info("Equals syntax parsed correctly")
    
    @pytest.mark.integration
    def test_parse_list_override(self):
        """Test parsing list values from CLI."""
        args = ['--k_values', '[0,2,4]']
        overrides = parse_cli_overrides(args)
        
        assert overrides['k_values'] == [0, 2, 4]
        
        logger.info("List override parsed correctly")
    
    @pytest.mark.integration
    def test_cli_overrides_config_file(self, tmp_path):
        """Test that CLI overrides take precedence over config file."""
        config_path = tmp_path / "config.yaml"
        
        config = {
            'k_values': [0, 2],
            'num_epochs': 15,
        }
        create_config_file(config, config_path)
        
        loaded = load_config_file(config_path)
        
        # CLI override
        cli_args = ['--num_epochs', '30']
        overrides = parse_cli_overrides(cli_args)
        
        # Merge with CLI taking precedence
        final = merge_configs(loaded, overrides)
        
        assert final['k_values'] == [0, 2]  # From file
        assert final['num_epochs'] == 30    # From CLI
        
        logger.info("CLI overrides take precedence")


# ============================================================================
# Config Merging Tests
# ============================================================================

class TestConfigMerging:
    """Tests for config merging functionality."""
    
    @pytest.mark.integration
    def test_merge_disjoint_configs(self):
        """Test merging configs with no overlapping keys."""
        base = {'k_values': [0, 2], 'num_epochs': 10}
        override = {'lrs': [0.01], 'seed': 123}
        
        merged = merge_configs(base, override)
        
        assert merged['k_values'] == [0, 2]
        assert merged['num_epochs'] == 10
        assert merged['lrs'] == [0.01]
        assert merged['seed'] == 123
        
        logger.info("Disjoint configs merged correctly")
    
    @pytest.mark.integration
    def test_merge_overlapping_configs(self):
        """Test merging configs with overlapping keys."""
        base = {'k_values': [0, 2], 'num_epochs': 10}
        override = {'k_values': [0, 1, 2, 3], 'seed': 123}
        
        merged = merge_configs(base, override)
        
        # Override takes precedence
        assert merged['k_values'] == [0, 1, 2, 3]
        assert merged['num_epochs'] == 10
        assert merged['seed'] == 123
        
        logger.info("Overlapping configs merged correctly")
    
    @pytest.mark.integration
    def test_three_way_merge(self, tmp_path):
        """Test merging defaults + file + CLI overrides."""
        config_path = tmp_path / "config.yaml"
        
        # File config
        file_config = {
            'k_values': [0, 2],
            'num_epochs': 20,
        }
        create_config_file(file_config, config_path)
        loaded = load_config_file(config_path)
        
        # CLI overrides
        cli_overrides = {'num_epochs': 30, 'seed': 999}
        
        # Three-way merge: defaults + file + CLI
        step1 = apply_defaults(loaded, DEFAULT_CONFIG)
        final = merge_configs(step1, cli_overrides)
        
        # Check precedence
        assert final['k_values'] == [0, 2]    # From file
        assert final['num_epochs'] == 30       # From CLI
        assert final['seed'] == 999            # From CLI
        assert 'poolings' in final             # From defaults
        
        logger.info("Three-way merge successful")


# ============================================================================
# Experiment Counting Tests
# ============================================================================

class TestExperimentCounting:
    """Tests for counting experiment combinations."""
    
    @pytest.mark.integration
    def test_count_minimal_config(self):
        """Test counting experiments from minimal config."""
        config = {
            'k_values': [0, 2],
            'num_repeats': 1,
        }
        
        count = count_experiment_combinations(config)
        
        # 2 k_values * 1 repeat = 2
        assert count >= 2
        
        logger.info(f"Minimal config: {count} experiments")
    
    @pytest.mark.integration
    def test_count_multi_parameter_config(self):
        """Test counting experiments from multi-parameter config."""
        config = {
            'k_values': [0, 1, 2],        # 3
            'max_depths': [2, 3],         # 2
            'hidden_dims': [64, 128],     # 2
            'lrs': [0.001, 0.01],         # 2
            'num_repeats': 3,             # 3
        }
        
        count = count_experiment_combinations(config)
        
        # 3 * 2 * 2 * 2 * 3 = 72
        expected = 3 * 2 * 2 * 2 * 3
        assert count == expected, f"Expected {expected}, got {count}"
        
        logger.info(f"Multi-parameter config: {count} experiments")
    
    @pytest.mark.integration
    def test_count_large_config(self):
        """Test counting experiments from large config."""
        config = {
            'k_values': [0, 1, 2, 3, 4, 5],  # 6
            'max_depths': [2, 3],             # 2
            'hidden_dims': [64, 128, 256],    # 3
            'poolings': ['learned', 'mean'],  # 2
            'temp_schedules': ['none', 'cosine'],  # 2
            'lrs': [0.001, 0.005, 0.01],      # 3
            'batch_sizes': [32, 64, 128],     # 3
            'weight_decays': [0.0, 0.01],     # 2
            'warmup_epochs': [3, 5],          # 2
            'num_repeats': 3,                 # 3
        }
        
        count = count_experiment_combinations(config)
        
        # Should be a large number
        expected = 6 * 2 * 3 * 2 * 2 * 3 * 3 * 2 * 2 * 3
        assert count == expected, f"Expected {expected}, got {count}"
        
        logger.info(f"Large config: {count} experiments (expected {expected})")


# ============================================================================
# Config File Path Tests
# ============================================================================

class TestConfigFilePaths:
    """Tests for config file path handling."""
    
    @pytest.mark.integration
    def test_load_from_absolute_path(self, tmp_path):
        """Test loading config from absolute path."""
        config_path = tmp_path / "config.yaml"
        config = {'k_values': [0, 2]}
        create_config_file(config, config_path)
        
        # Use absolute path
        loaded = load_config_file(config_path.absolute())
        
        assert loaded['k_values'] == [0, 2]
        
        logger.info("Config loaded from absolute path")
    
    @pytest.mark.integration
    def test_load_from_nested_directory(self, tmp_path):
        """Test loading config from nested directory."""
        nested_dir = tmp_path / "configs" / "examples"
        nested_dir.mkdir(parents=True)
        
        config_path = nested_dir / "test-config.yaml"
        config = {'k_values': [0, 2]}
        create_config_file(config, config_path)
        
        loaded = load_config_file(config_path)
        
        assert loaded['k_values'] == [0, 2]
        
        logger.info("Config loaded from nested directory")
    
    @pytest.mark.integration
    def test_nonexistent_file_raises(self, tmp_path):
        """Test that loading nonexistent file raises error."""
        config_path = tmp_path / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError):
            load_config_file(config_path)
        
        logger.info("Nonexistent file raises FileNotFoundError")


# ============================================================================
# YAML Format Tests
# ============================================================================

class TestYAMLFormat:
    """Tests for YAML format handling."""
    
    @pytest.mark.integration
    def test_yaml_flow_style(self, tmp_path):
        """Test parsing YAML with flow style (inline lists)."""
        config_path = tmp_path / "config.yaml"
        
        yaml_content = """
k_values: [0, 1, 2, 3]
lrs: [0.001, 0.01]
"""
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        loaded = load_config_file(config_path)
        
        assert loaded['k_values'] == [0, 1, 2, 3]
        assert loaded['lrs'] == [0.001, 0.01]
        
        logger.info("Flow style YAML parsed correctly")
    
    @pytest.mark.integration
    def test_yaml_block_style(self, tmp_path):
        """Test parsing YAML with block style (multiline lists)."""
        config_path = tmp_path / "config.yaml"
        
        yaml_content = """
k_values:
  - 0
  - 1
  - 2
  - 3
lrs:
  - 0.001
  - 0.01
"""
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        loaded = load_config_file(config_path)
        
        assert loaded['k_values'] == [0, 1, 2, 3]
        assert loaded['lrs'] == [0.001, 0.01]
        
        logger.info("Block style YAML parsed correctly")
    
    @pytest.mark.integration
    def test_yaml_comments_ignored(self, tmp_path):
        """Test that YAML comments are ignored."""
        config_path = tmp_path / "config.yaml"
        
        yaml_content = """
# This is a comment
k_values: [0, 2]  # Inline comment
# Another comment
num_epochs: 15
"""
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        loaded = load_config_file(config_path)
        
        assert loaded['k_values'] == [0, 2]
        assert loaded['num_epochs'] == 15
        assert '# This is a comment' not in str(loaded)
        
        logger.info("YAML comments ignored correctly")
    
    @pytest.mark.integration
    def test_yaml_with_anchors(self, tmp_path):
        """Test parsing YAML with anchors and aliases."""
        config_path = tmp_path / "config.yaml"
        
        yaml_content = """
common_lrs: &lrs
  - 0.001
  - 0.01

k_values: [0, 2]
lrs: *lrs
"""
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        loaded = load_config_file(config_path)
        
        assert loaded['lrs'] == [0.001, 0.01]
        
        logger.info("YAML anchors/aliases parsed correctly")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])