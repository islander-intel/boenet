# ============================================================================
# BFSNet CSV Output Validation Tests
# ============================================================================
# Tests for CSV output generation and format validation.
#
# These tests verify:
#   - CSV files are created correctly
#   - Required columns are present
#   - Data types are correct
#   - Values are within expected ranges
#   - Multiple runs append correctly
#   - CSV can be loaded and parsed
#
# The training matrix (bfs_training_matrix.py) generates CSV files with
# experiment results. These tests ensure the output format is correct
# and consistent.
#
# Run:
#   pytest tests/integration/test_csv_output.py -v
# ============================================================================

import pytest
import torch
import csv
import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import io

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Expected CSV Schema
# ============================================================================

# Required columns that must be present in every CSV output
REQUIRED_COLUMNS = [
    'k',
    'max_depth',
    'hidden_dim',
    'pooling',
    'lr',
    'batch_size',
    'weight_decay',
    'temp_schedule',
    'warmup_epochs',
    'repeat',
    'final_train_loss',
    'final_train_acc',
    'final_val_loss',
    'final_val_acc',
    'best_val_acc',
    'best_epoch',
    'total_time_sec',
]

# Optional columns that may be present
OPTIONAL_COLUMNS = [
    'temp_fixed',
    'temp_anneal',
    'temp_start',
    'temp_end',
    'num_epochs',
    'timestamp',
    'device',
    'seed',
]

# Column data types for validation
COLUMN_TYPES = {
    'k': int,
    'max_depth': int,
    'hidden_dim': int,
    'pooling': str,
    'lr': float,
    'batch_size': int,
    'weight_decay': float,
    'temp_schedule': str,
    'temp_fixed': float,
    'temp_anneal': str,
    'warmup_epochs': int,
    'repeat': int,
    'final_train_loss': float,
    'final_train_acc': float,
    'final_val_loss': float,
    'final_val_acc': float,
    'best_val_acc': float,
    'best_epoch': int,
    'total_time_sec': float,
    'num_epochs': int,
    'timestamp': str,
    'device': str,
    'seed': int,
}

# Value ranges for validation
VALUE_RANGES = {
    'k': (0, 10),
    'max_depth': (1, 10),
    'hidden_dim': (1, 2048),
    'lr': (1e-8, 10.0),
    'batch_size': (1, 4096),
    'weight_decay': (0.0, 1.0),
    'warmup_epochs': (0, 100),
    'repeat': (0, 100),
    'final_train_loss': (0.0, 100.0),
    'final_train_acc': (0.0, 1.0),
    'final_val_loss': (0.0, 100.0),
    'final_val_acc': (0.0, 1.0),
    'best_val_acc': (0.0, 1.0),
    'best_epoch': (0, 1000),
    'total_time_sec': (0.0, 86400.0),  # Max 1 day
}


# ============================================================================
# Helper Functions
# ============================================================================

def create_sample_csv_row() -> Dict[str, Any]:
    """Create a sample CSV row with valid data."""
    return {
        'k': 2,
        'max_depth': 2,
        'hidden_dim': 64,
        'pooling': 'learned',
        'lr': 0.001,
        'batch_size': 64,
        'weight_decay': 0.01,
        'temp_schedule': 'cosine',
        'temp_fixed': 1.0,
        'temp_anneal': '1.5->0.5',
        'warmup_epochs': 3,
        'repeat': 0,
        'final_train_loss': 0.5432,
        'final_train_acc': 0.8234,
        'final_val_loss': 0.6123,
        'final_val_acc': 0.7890,
        'best_val_acc': 0.8012,
        'best_epoch': 12,
        'total_time_sec': 45.67,
        'num_epochs': 15,
        'timestamp': datetime.now().isoformat(),
        'device': 'cuda:0',
        'seed': 42,
    }


def write_csv_file(
    filepath: Path,
    rows: List[Dict[str, Any]],
    columns: Optional[List[str]] = None
) -> None:
    """Write rows to CSV file."""
    if columns is None:
        columns = list(rows[0].keys()) if rows else REQUIRED_COLUMNS
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k in columns})
    
    logger.debug(f"Wrote CSV with {len(rows)} rows to {filepath}")


def read_csv_file(filepath: Path) -> List[Dict[str, str]]:
    """Read CSV file and return list of row dictionaries."""
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    logger.debug(f"Read CSV with {len(rows)} rows from {filepath}")
    return rows


def validate_csv_row(
    row: Dict[str, str],
    required_columns: List[str] = REQUIRED_COLUMNS
) -> List[str]:
    """
    Validate a single CSV row.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required columns
    for col in required_columns:
        if col not in row:
            errors.append(f"Missing required column: {col}")
    
    # Check data types and ranges
    for col, value in row.items():
        if col in COLUMN_TYPES:
            expected_type = COLUMN_TYPES[col]
            try:
                if expected_type == int:
                    parsed = int(float(value))  # Handle "2.0" as int
                elif expected_type == float:
                    parsed = float(value)
                else:
                    parsed = str(value)
                
                # Check range
                if col in VALUE_RANGES:
                    min_val, max_val = VALUE_RANGES[col]
                    if not (min_val <= parsed <= max_val):
                        errors.append(f"Column '{col}' value {parsed} out of range [{min_val}, {max_val}]")
            
            except (ValueError, TypeError) as e:
                errors.append(f"Column '{col}' has invalid type: {value} (expected {expected_type.__name__})")
    
    return errors


def append_to_csv(filepath: Path, row: Dict[str, Any]) -> None:
    """Append a row to existing CSV file."""
    file_exists = filepath.exists()
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================================
# CSV File Creation Tests
# ============================================================================

class TestCSVFileCreation:
    """Tests for CSV file creation."""
    
    @pytest.mark.integration
    def test_create_empty_csv(self, tmp_path):
        """Test creating an empty CSV with headers."""
        csv_path = tmp_path / "results.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
        
        assert csv_path.exists()
        
        # Read and verify headers
        rows = read_csv_file(csv_path)
        assert len(rows) == 0
        
        with open(csv_path, 'r') as f:
            header = f.readline().strip()
        
        for col in REQUIRED_COLUMNS:
            assert col in header, f"Missing column: {col}"
        
        logger.info("Empty CSV created with correct headers")
    
    @pytest.mark.integration
    def test_create_csv_with_single_row(self, tmp_path):
        """Test creating CSV with a single row."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        write_csv_file(csv_path, [row])
        
        assert csv_path.exists()
        
        rows = read_csv_file(csv_path)
        assert len(rows) == 1
        
        logger.info("CSV created with single row")
    
    @pytest.mark.integration
    def test_create_csv_with_multiple_rows(self, tmp_path):
        """Test creating CSV with multiple rows."""
        csv_path = tmp_path / "results.csv"
        
        rows = []
        for i in range(10):
            row = create_sample_csv_row()
            row['repeat'] = i
            row['k'] = i % 6
            rows.append(row)
        
        write_csv_file(csv_path, rows)
        
        read_rows = read_csv_file(csv_path)
        assert len(read_rows) == 10
        
        logger.info("CSV created with 10 rows")
    
    @pytest.mark.integration
    def test_append_to_csv(self, tmp_path):
        """Test appending rows to existing CSV."""
        csv_path = tmp_path / "results.csv"
        
        # Create initial file
        row1 = create_sample_csv_row()
        row1['repeat'] = 0
        write_csv_file(csv_path, [row1])
        
        # Append more rows
        for i in range(1, 5):
            row = create_sample_csv_row()
            row['repeat'] = i
            append_to_csv(csv_path, row)
        
        # Verify
        rows = read_csv_file(csv_path)
        assert len(rows) == 5
        
        for i, row in enumerate(rows):
            assert int(row['repeat']) == i
        
        logger.info("CSV appending works correctly")


# ============================================================================
# CSV Column Validation Tests
# ============================================================================

class TestCSVColumnValidation:
    """Tests for CSV column validation."""
    
    @pytest.mark.integration
    def test_all_required_columns_present(self, tmp_path):
        """Test that all required columns are present."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        write_csv_file(csv_path, [row], columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
        
        rows = read_csv_file(csv_path)
        assert len(rows) == 1
        
        for col in REQUIRED_COLUMNS:
            assert col in rows[0], f"Missing required column: {col}"
        
        logger.info("All required columns present")
    
    @pytest.mark.integration
    def test_missing_column_detected(self, tmp_path):
        """Test that missing columns are detected."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        # Remove a required column
        del row['best_val_acc']
        
        # Write with subset of columns
        columns = [c for c in REQUIRED_COLUMNS if c != 'best_val_acc']
        write_csv_file(csv_path, [row], columns=columns)
        
        rows = read_csv_file(csv_path)
        errors = validate_csv_row(rows[0])
        
        assert any('best_val_acc' in e for e in errors), "Should detect missing column"
        
        logger.info("Missing column correctly detected")
    
    @pytest.mark.integration
    def test_extra_columns_allowed(self, tmp_path):
        """Test that extra columns don't cause errors."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        # Add extra column
        row['extra_metric'] = 0.999
        
        write_csv_file(csv_path, [row])
        
        rows = read_csv_file(csv_path)
        assert 'extra_metric' in rows[0]
        
        # Validation should still pass
        errors = validate_csv_row(rows[0])
        assert len(errors) == 0
        
        logger.info("Extra columns handled correctly")


# ============================================================================
# CSV Data Type Validation Tests
# ============================================================================

class TestCSVDataTypeValidation:
    """Tests for CSV data type validation."""
    
    @pytest.mark.integration
    def test_integer_columns(self, tmp_path):
        """Test that integer columns contain valid integers."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        write_csv_file(csv_path, [row])
        
        rows = read_csv_file(csv_path)
        
        int_columns = ['k', 'max_depth', 'hidden_dim', 'batch_size', 
                       'warmup_epochs', 'repeat', 'best_epoch']
        
        for col in int_columns:
            if col in rows[0]:
                value = rows[0][col]
                try:
                    int(float(value))
                except ValueError:
                    pytest.fail(f"Column {col} not a valid integer: {value}")
        
        logger.info("Integer columns validated")
    
    @pytest.mark.integration
    def test_float_columns(self, tmp_path):
        """Test that float columns contain valid floats."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        write_csv_file(csv_path, [row])
        
        rows = read_csv_file(csv_path)
        
        float_columns = ['lr', 'weight_decay', 'final_train_loss', 'final_train_acc',
                        'final_val_loss', 'final_val_acc', 'best_val_acc', 'total_time_sec']
        
        for col in float_columns:
            if col in rows[0]:
                value = rows[0][col]
                try:
                    float(value)
                except ValueError:
                    pytest.fail(f"Column {col} not a valid float: {value}")
        
        logger.info("Float columns validated")
    
    @pytest.mark.integration
    def test_string_columns(self, tmp_path):
        """Test that string columns contain valid strings."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        write_csv_file(csv_path, [row])
        
        rows = read_csv_file(csv_path)
        
        string_columns = ['pooling', 'temp_schedule']
        
        for col in string_columns:
            if col in rows[0]:
                value = rows[0][col]
                assert isinstance(value, str), f"Column {col} not a string: {value}"
                assert len(value) > 0, f"Column {col} is empty"
        
        logger.info("String columns validated")
    
    @pytest.mark.integration
    def test_invalid_type_detected(self, tmp_path):
        """Test that invalid types are detected."""
        csv_path = tmp_path / "results.csv"
        
        # Create row with invalid type
        with open(csv_path, 'w', newline='') as f:
            f.write("k,max_depth,hidden_dim,pooling,lr,batch_size,weight_decay,")
            f.write("temp_schedule,warmup_epochs,repeat,final_train_loss,final_train_acc,")
            f.write("final_val_loss,final_val_acc,best_val_acc,best_epoch,total_time_sec\n")
            f.write("invalid,2,64,learned,0.001,64,0.01,cosine,3,0,0.5,0.8,0.6,0.7,0.8,10,45.0\n")
        
        rows = read_csv_file(csv_path)
        errors = validate_csv_row(rows[0])
        
        assert any('k' in e for e in errors), "Should detect invalid type for 'k'"
        
        logger.info("Invalid type correctly detected")


# ============================================================================
# CSV Value Range Validation Tests
# ============================================================================

class TestCSVValueRangeValidation:
    """Tests for CSV value range validation."""
    
    @pytest.mark.integration
    def test_accuracy_in_valid_range(self, tmp_path):
        """Test that accuracy values are in [0, 1]."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        # Set valid accuracy values
        row['final_train_acc'] = 0.85
        row['final_val_acc'] = 0.80
        row['best_val_acc'] = 0.82
        
        write_csv_file(csv_path, [row])
        
        rows = read_csv_file(csv_path)
        errors = validate_csv_row(rows[0])
        
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        
        logger.info("Accuracy values in valid range")
    
    @pytest.mark.integration
    def test_accuracy_out_of_range_detected(self, tmp_path):
        """Test that out-of-range accuracy is detected."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        # Set invalid accuracy (> 1.0)
        row['final_val_acc'] = 1.5
        
        write_csv_file(csv_path, [row])
        
        rows = read_csv_file(csv_path)
        errors = validate_csv_row(rows[0])
        
        assert any('final_val_acc' in e for e in errors), "Should detect out-of-range accuracy"
        
        logger.info("Out-of-range accuracy detected")
    
    @pytest.mark.integration
    def test_k_value_range(self, tmp_path):
        """Test that K values are in expected range."""
        csv_path = tmp_path / "results.csv"
        
        rows = []
        for k in range(6):  # K=0 to K=5
            row = create_sample_csv_row()
            row['k'] = k
            rows.append(row)
        
        write_csv_file(csv_path, rows)
        
        read_rows = read_csv_file(csv_path)
        
        for r in read_rows:
            errors = validate_csv_row(r)
            assert len(errors) == 0, f"Unexpected errors for K={r['k']}: {errors}"
        
        logger.info("All K values in valid range")
    
    @pytest.mark.integration
    def test_negative_time_detected(self, tmp_path):
        """Test that negative time is detected as invalid."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        row['total_time_sec'] = -10.0
        
        write_csv_file(csv_path, [row])
        
        rows = read_csv_file(csv_path)
        errors = validate_csv_row(rows[0])
        
        assert any('total_time_sec' in e for e in errors), "Should detect negative time"
        
        logger.info("Negative time correctly detected")


# ============================================================================
# CSV Format Tests
# ============================================================================

class TestCSVFormat:
    """Tests for CSV format correctness."""
    
    @pytest.mark.integration
    def test_csv_comma_delimiter(self, tmp_path):
        """Test that CSV uses comma delimiter."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        write_csv_file(csv_path, [row])
        
        with open(csv_path, 'r') as f:
            content = f.read()
        
        # Should have commas (not tabs or semicolons)
        assert ',' in content
        assert '\t' not in content.replace('\n', '')
        
        logger.info("CSV uses comma delimiter")
    
    @pytest.mark.integration
    def test_csv_header_row(self, tmp_path):
        """Test that CSV has header row."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        write_csv_file(csv_path, [row])
        
        with open(csv_path, 'r') as f:
            first_line = f.readline().strip()
        
        # First line should be headers (not data)
        assert 'k' in first_line
        assert 'final_val_acc' in first_line
        
        # Should not be a numeric value
        try:
            float(first_line.split(',')[0])
            pytest.fail("First row should be header, not data")
        except ValueError:
            pass  # Expected - header is not numeric
        
        logger.info("CSV has correct header row")
    
    @pytest.mark.integration
    def test_csv_no_trailing_comma(self, tmp_path):
        """Test that CSV rows don't have trailing commas."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        write_csv_file(csv_path, [row])
        
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                assert not line.endswith(','), f"Trailing comma in: {line[:50]}..."
        
        logger.info("No trailing commas in CSV")
    
    @pytest.mark.integration
    def test_csv_consistent_columns(self, tmp_path):
        """Test that all rows have consistent columns."""
        csv_path = tmp_path / "results.csv"
        
        rows = []
        for i in range(5):
            row = create_sample_csv_row()
            row['repeat'] = i
            rows.append(row)
        
        write_csv_file(csv_path, rows)
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            num_columns = len(header)
            
            for i, data_row in enumerate(reader):
                assert len(data_row) == num_columns, \
                    f"Row {i} has {len(data_row)} columns, expected {num_columns}"
        
        logger.info("All rows have consistent columns")
    
    @pytest.mark.integration
    def test_csv_handles_special_characters(self, tmp_path):
        """Test that CSV handles special characters correctly."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        # Add values with special characters
        row['temp_anneal'] = '1.5->0.5,cosine'  # Contains comma
        
        write_csv_file(csv_path, [row])
        
        # Should be readable
        rows = read_csv_file(csv_path)
        assert len(rows) == 1
        
        logger.info("CSV handles special characters correctly")


# ============================================================================
# CSV Parsing Tests
# ============================================================================

class TestCSVParsing:
    """Tests for CSV parsing and loading."""
    
    @pytest.mark.integration
    def test_parse_with_pandas(self, tmp_path):
        """Test that CSV can be parsed with pandas."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")
        
        csv_path = tmp_path / "results.csv"
        rows = [create_sample_csv_row() for _ in range(5)]
        write_csv_file(csv_path, rows)
        
        df = pd.read_csv(csv_path)
        
        assert len(df) == 5
        assert 'k' in df.columns
        assert 'final_val_acc' in df.columns
        
        logger.info("CSV parsed successfully with pandas")
    
    @pytest.mark.integration
    def test_parse_with_csv_module(self, tmp_path):
        """Test that CSV can be parsed with csv module."""
        csv_path = tmp_path / "results.csv"
        rows = [create_sample_csv_row() for _ in range(5)]
        write_csv_file(csv_path, rows)
        
        parsed_rows = read_csv_file(csv_path)
        
        assert len(parsed_rows) == 5
        assert all('k' in r for r in parsed_rows)
        
        logger.info("CSV parsed successfully with csv module")
    
    @pytest.mark.integration
    def test_parse_large_csv(self, tmp_path):
        """Test parsing a larger CSV file."""
        csv_path = tmp_path / "results.csv"
        
        rows = []
        for i in range(100):
            row = create_sample_csv_row()
            row['repeat'] = i
            rows.append(row)
        
        write_csv_file(csv_path, rows)
        
        parsed_rows = read_csv_file(csv_path)
        assert len(parsed_rows) == 100
        
        logger.info("Large CSV (100 rows) parsed successfully")


# ============================================================================
# CSV Content Validation Tests
# ============================================================================

class TestCSVContentValidation:
    """Tests for validating CSV content correctness."""
    
    @pytest.mark.integration
    def test_pooling_values(self, tmp_path):
        """Test that pooling values are valid."""
        csv_path = tmp_path / "results.csv"
        
        valid_poolings = ['learned', 'sum', 'mean']
        
        rows = []
        for pooling in valid_poolings:
            row = create_sample_csv_row()
            row['pooling'] = pooling
            rows.append(row)
        
        write_csv_file(csv_path, rows)
        
        parsed_rows = read_csv_file(csv_path)
        
        for r in parsed_rows:
            assert r['pooling'] in valid_poolings, f"Invalid pooling: {r['pooling']}"
        
        logger.info("Pooling values validated")
    
    @pytest.mark.integration
    def test_temp_schedule_values(self, tmp_path):
        """Test that temperature schedule values are valid."""
        csv_path = tmp_path / "results.csv"
        
        valid_schedules = ['none', 'cosine', 'fixed', 'anneal']
        
        rows = []
        for schedule in valid_schedules:
            row = create_sample_csv_row()
            row['temp_schedule'] = schedule
            rows.append(row)
        
        write_csv_file(csv_path, rows)
        
        parsed_rows = read_csv_file(csv_path)
        
        for r in parsed_rows:
            assert r['temp_schedule'] in valid_schedules, \
                f"Invalid temp_schedule: {r['temp_schedule']}"
        
        logger.info("Temperature schedule values validated")
    
    @pytest.mark.integration
    def test_metrics_consistency(self, tmp_path):
        """Test that metrics are internally consistent."""
        csv_path = tmp_path / "results.csv"
        row = create_sample_csv_row()
        
        # best_val_acc should be >= final_val_acc (or close)
        row['final_val_acc'] = 0.75
        row['best_val_acc'] = 0.80
        
        write_csv_file(csv_path, [row])
        
        parsed_rows = read_csv_file(csv_path)
        r = parsed_rows[0]
        
        final_val = float(r['final_val_acc'])
        best_val = float(r['best_val_acc'])
        
        # Best should be >= final (allowing for some numerical tolerance)
        assert best_val >= final_val - 0.01, \
            f"best_val_acc ({best_val}) < final_val_acc ({final_val})"
        
        logger.info("Metrics consistency validated")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])