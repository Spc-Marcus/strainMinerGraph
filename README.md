# StrainMiner

A modern tool for separating microbial strains in metagenomic assemblies using Integer Linear Programming (ILP).

*Based on the original work by Tam Khac Min Truong and Roland Faure, refactored and enhanced by Marcus Foin.*

## Overview

StrainMiner identifies and separates closely related microbial strains from metagenomic data by analyzing variant patterns in mapped sequencing reads. It uses advanced ILP optimization to find optimal strain groupings while accounting for sequencing errors and assembly artifacts.

### Key Features

- **ILP-based optimization** for robust strain separation
- **Modern Python architecture** with comprehensive error handling
- **Scalable processing** with memory-efficient algorithms
- **Comprehensive validation** of input data and parameters
- **Detailed logging** and progress tracking
- **Flexible output formats** for downstream analysis

## Installation

### Prerequisites

#### System Dependencies
Install these tools in your PATH (recommended via conda):
```bash
conda install -c bioconda minimap2 samtools racon
```

#### Python Dependencies
```bash
# Core scientific computing
pip install numpy pandas scikit-learn

# Bioinformatics
pip install pysam

# Optimization (requires valid Gurobi license)
pip install gurobipy
```

### Gurobi License Setup

StrainMiner requires a valid Gurobi license for ILP optimization. 

1. **Academic License**: Get a free academic license at [gurobi.com/academia](https://www.gurobi.com/academia/)
2. **Commercial License**: Contact Gurobi for commercial licensing

After obtaining your license, update the credentials in `src/strainminer/cluster.py`:
```python
options = {
    "WLSACCESSID": "your-access-id",
    "WLSSECRET": "your-secret-key", 
    "LICENSEID": your-license-id
}
```

### Build and Install

#### Development Installation
```bash
# Clone the repository
git clone https://github.com/Spc-Marcus/strainMinerGraph.git
cd strainMiner

# Create conda environment
conda env create -f strainminer.yml
conda activate strainminer

# Build C extensions
mkdir -p build
cd build
cmake ..
make
cd ..

# Install in development mode
pip install -e .
```

### Verify Installation
```bash
# Test the installation
python -m strainminer --help
```

## Usage

### Basic Command Line

```bash
python -m strainminer -a ASSEMBLY -b BAM -r READS -o OUTPUT_DIR [OPTIONS]
```

### Required Arguments

- `-a, --assembly ASSEMBLY`: Assembly file in GFA format
- `-b, --bam BAM`: Indexed BAM alignment file  
- `-r, --reads READS`: Sequencing reads (FASTQ/FASTA format)
- `-o, --out-folder OUTPUT_DIR`: Output directory for results

### Optional Arguments

- `-e, --error_rate ERROR_RATE`: Expected sequencing error rate (default: 0.025)
- `--window WINDOW`: Analysis window size in bp (default: 5000)
- `--min-coverage MIN_COV`: Minimum coverage for variant calling (default: 5)
- `--min-quality MIN_QUAL`: Minimum base quality score (default: 10)
- `-v, --verbose`: Enable verbose logging
- `-h, --help`: Show help message

### Examples

#### Basic Strain Separation
```bash
python -m strainminer \
    -a assembly.gfa \
    -b aligned_reads.bam \
    -r raw_reads.fastq \
    -o strain_results
```

#### High-Error Rate Data
```bash
python -m strainminer \
    -a assembly.gfa \
    -b aligned_reads.bam \
    -r raw_reads.fastq \
    -o strain_results \
    --error_rate 0.05 \
    --min-coverage 10
```

#### Large Dataset Processing
```bash
python -m strainminer \
    -a large_assembly.gfa \
    -b large_alignment.bam \
    -r large_reads.fastq \
    -o large_results \
    --window 10000 \
    --verbose
```

## Input File Requirements

### Assembly File (GFA)
- **Format**: GFA 1.0 or 2.0
- **Content**: Assembled contigs/scaffolds
- **Requirements**: Must contain segment records with sequence data

### Alignment File (BAM)
- **Format**: Binary Alignment Map (BAM)
- **Requirements**: 
  - Must be sorted and indexed (`.bai` file)
  - Reads aligned to the assembly sequences
  - Generated with tools like `minimap2` or `bwa`

### Reads File (FASTQ/FASTA)
- **Format**: FASTQ or FASTA
- **Content**: Raw sequencing reads used for alignment
- **Requirements**: Must match reads in BAM file

### Preparing Input Files

#### 1. Create BAM Alignment
```bash
# Using minimap2 for long reads
minimap2 -ax map-ont assembly.gfa reads.fastq | \
samtools sort -o alignment.bam
samtools index alignment.bam

# Using minimap2 for short reads  
minimap2 -ax sr assembly.gfa reads.fastq | \
samtools sort -o alignment.bam
samtools index alignment.bam
```

#### 2. Validate File Formats
```bash
# Check GFA format
head -n 5 assembly.gfa

# Verify BAM is sorted and indexed
samtools quickcheck alignment.bam
ls -la alignment.bam.bai

# Check reads format
head -n 4 reads.fastq
```

## Output Structure

StrainMiner creates a comprehensive output directory:

```
output_directory/
├── logs/
│   ├── strainminer.log          # Detailed execution log
├── tmp/                         # Temporary processing file
├── supercontigs.txt  # Text file containing information about assembled supercontigs
├── strainminer_final_assembly.gfa  # Final assembly in GFA (Graphical Fragment Assembly)
├── strainminer_final_assembly.fasta # Final assembled sequences in FASTA format

```


## Algorithm Overview

### 1. Variant Detection
- Pileup analysis of aligned reads
- Quality filtering and coverage thresholds
- Identification of polymorphic sites

### 2. Matrix Construction
- Binary encoding of variant patterns
- Missing data imputation using k-NN
- Feature clustering for computational efficiency

### 3. ILP Optimization
- Quasi-biclique detection in variant matrix
- Error-aware objective function
- Iterative refinement of strain assignments

### 4. Post-processing
- Strain validation and merging
- Quality assessment and filtering
- Output generation and formatting

## Performance Considerations

### Optimization Tips

1. **Reduce window size** for memory-constrained systems
2. **Increase error rate** for noisy data
3. **Filter low-quality regions** before analysis

## Troubleshooting

### Common Issues

#### Gurobi License Errors
```bash
# Error: Invalid license
# Solution: Verify license credentials in source code
```

#### Memory Errors
```bash
# Error: Out of memory during clustering
# Solution: Reduce window size or increase system RAM
```

#### BAM Index Missing
```bash
# Error: BAM file not indexed
# Solution: Create index with samtools
samtools index alignment.bam
```

#### Empty Results
```bash
# Check input data quality
samtools flagstat alignment.bam
samtools coverage alignment.bam
```

### Getting Help

1. **Check logs**: Detailed error information
2. **Validate inputs**: Use provided validation functions
3. **Adjust parameters**: Try different error rates or window sizes
4. **Contact support**: Open an issue on GitHub

## Citation

If you use StrainMiner in your research, please cite:

```bibtex
@article{truong2024strainminer,
  title={StrainMiner: Integer Linear Programming for Strain Separation in Metagenomic Assemblies},
  author={Truong, Tam Khac Min and Faure, Roland and Andonov, Rumen},
  journal={HAL Preprint},
  year={2024},
  url={https://inria.hal.science/hal-04349675}
}
```


## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see [LICENSE](LICENSE) file for details.

**Important License Notice**: 
- This is a copyleft license that requires derivative works to be licensed under the same terms
- If you modify this software and provide it as a network service (e.g., web application, API), you must make the complete source code available to users of that service
- Commercial use is permitted, but all modifications must remain open source under AGPL-3.0

## Acknowledgments

- **Tam Khac Min Truong**: Original algorithm development and implementation
- **Roland Faure**: Integration and pipeline development  
- **Riccardo Vicedomini**: Scientific supervision
- **Rumen Andonov**: Scientific supervision
- **Marcus Foin**: Code refactoring and modernization