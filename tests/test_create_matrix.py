import pytest
import numpy as np
import pandas as pd
from strainminer.matrix import create_matrix



def test_empty_input():
        """Test with empty dictionary input."""
        matrix, reads = create_matrix({})
        
        assert matrix.shape == (0,)
        assert reads == []
    
def test_single_position():
        """Test with single genomic position."""
        data = {1000: {'read1': 1, 'read2': 0, 'read3': 1}}
        matrix, reads = create_matrix(data, min_coverage_threshold=0.5)
        
        # With only one column, thirds filtering shouldn't apply
        assert len(reads) == 3
        assert matrix.shape == (3, 1)
        np.testing.assert_array_equal(matrix.flatten(), [1, 0, 1])
    
def test_normal_case():
        """Test with normal genomic data."""
        data = {
            1000: {'read1': 1, 'read2': 0, 'read3': 1},
            1001: {'read1': 0, 'read2': 1, 'read3': 0},
            1002: {'read1': 1, 'read2': 1, 'read3': 1}
        }
        matrix, reads = create_matrix(data, min_coverage_threshold=0.6)
        
        assert len(reads) == 3
        assert matrix.shape == (3, 3)
        assert set(reads) == {'read1', 'read2', 'read3'}
    
def test_coverage_filtering():
        """Test column filtering based on coverage threshold."""
        data = {
            1000: {'read1': 1, 'read2': 0},  # 100% coverage
            1001: {'read1': None, 'read2': 1},  # 50% coverage
            1002: {'read1': 1, 'read2': 1}  # 100% coverage
        }
        matrix, reads = create_matrix(data, min_coverage_threshold=0.8)
        
        # Only positions with >=80% coverage should remain
        assert matrix.shape[1] == 2  # Only positions 1000 and 1002
        assert len(reads) == 2
    
def test_thirds_filtering():
        """Test filtering of reads based on thirds coverage."""
        # Create data where some reads don't span the full window
        data = {}
        positions = list(range(1000, 1010))  # 10 positions
        
        # read1: missing data in first third
        # read2: missing data in last third  
        # read3: spans entire window
        for i, pos in enumerate(positions):
            data[pos] = {}
            if i >= 3:  # read1 missing first third
                data[pos]['read1'] = 1
            if i < 7:   # read2 missing last third
                data[pos]['read2'] = 1
            data[pos]['read3'] = 1  # read3 spans all
        
        matrix, reads = create_matrix(data, min_coverage_threshold=0.1)
        
        # Only read3 should remain as it spans the entire window
        assert 'read3' in reads
        assert len(reads) >= 1
    
def test_all_reads_filtered_out():
        """Test case where all reads are filtered out."""
        data = {
            1000: {'read1': 1},
            1001: {'read2': 1},  # Different reads at different positions
            1002: {'read3': 1}
        }
        matrix, reads = create_matrix(data, min_coverage_threshold=0.9)
        
        # High coverage threshold should filter out all columns
        assert matrix.shape == (0,)
        assert reads == []
    
def test_min_coverage_threshold_validation():
        """Test different coverage threshold values."""
        data = {
            1000: {'read1': 1, 'read2': None, 'read3': 1},  # 67% coverage
            1001: {'read1': 1, 'read2': 1, 'read3': 1}      # 100% coverage
        }
        
        # With 60% threshold, both columns should pass
        matrix1, reads1 = create_matrix(data, min_coverage_threshold=0.6)
        assert matrix1.shape[1] == 2
        
        # With 80% threshold, only second column should pass
        matrix2, reads2 = create_matrix(data, min_coverage_threshold=0.8)
        assert matrix2.shape[1] == 1
    
def test_data_types():
        """Test that function handles different data types correctly."""
        data = {
            1000: {'read1': 1, 'read2': 0},
            1001: {'read1': 0, 'read2': 1}
        }
        matrix, reads = create_matrix(data)
        
        assert isinstance(matrix, np.ndarray)
        assert isinstance(reads, list)
        assert all(isinstance(read, str) for read in reads)
    
def test_large_dataset():
        """Test with larger dataset to ensure performance."""
        # Create larger test dataset
        positions = list(range(1000, 1100))  # 100 positions
        read_names = [f'read_{i}' for i in range(50)]  # 50 reads
        
        data = {}
        for pos in positions:
            data[pos] = {}
            for read in read_names:
                # Random binary values
                data[pos][read] = np.random.choice([0, 1])
        
        matrix, reads = create_matrix(data, min_coverage_threshold=0.5)
        
        assert matrix.shape[0] <= len(read_names)
        assert matrix.shape[1] <= len(positions)
        assert len(reads) == matrix.shape[0]


