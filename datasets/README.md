# Datasets

This folder contains benchmark datasets for evaluating Restricted Decision Trees (RDT) and related algorithms.

## Quick Start

### 1. Download Datasets

```bash
cd datasets
python download_datasets.py
```

**Requirements**: `pip install requests pandas openpyxl scipy`

This will download 18 datasets from the UCI ML Repository and package them as ZIP files.

### 2. Validate Datasets

```bash
python validate_datasets.py
```

This validates all ZIP files and generates:
- `DATASETS_CATALOG.md` - Human-readable catalog
- `datasets_summary.csv` - Machine-readable summary

## Dataset List

The following 21 datasets are included (Heart Disease already present):

| Dataset | Samples | Features | Classes | Type |
|---------|---------|----------|---------|------|
| Skin Segmentation | ~245K | 3 | 2 | Binary |
| Avila | ~21K | 10 | 12 | Multi-class |
| Occupancy Detection | ~20K | 5 | 2 | Binary |
| Magic Gamma Telescope | ~19K | 10 | 2 | Binary |
| HTRU2 (Pulsar) | ~18K | 8 | 2 | Binary |
| EEG Eye State | ~15K | 14 | 2 | Binary |
| Dry Bean | ~14K | 16 | 7 | Multi-class |
| Room Occupancy | ~8K | 5 | 2 | Binary |
| Online Bidding | ~7K | 9 | 2 | Binary |
| Satellite | ~6K | 36 | 6 | Multi-class |
| Page Blocks | ~5K | 10 | 5 | Multi-class |
| Wall-Following Robot | ~5K | 24 | 4 | Multi-class |
| Spambase | ~4.6K | 57 | 2 | Binary |
| Segment | ~2.3K | 19 | 7 | Multi-class |
| Steel Plates Faults | ~2K | 27 | 7 | Multi-class |
| QSAR Biodegradation | ~1K | 41 | 2 | Binary |
| Vehicle | ~850 | 18 | 4 | Multi-class |
| Pima Indians Diabetes | 768 | 8 | 2 | Binary |
| Breast Cancer (WDBC) | 569 | 30 | 2 | Binary |
| Ionosphere | 351 | 34 | 2 | Binary |
| Heart Disease | ~1K | 13 | 2 | Binary |

**Note**: Some datasets may require manual handling (see `download_datasets.py` for details).

## Dataset Format

All datasets follow a standardized format:

- **File**: `Dataset Name.zip`
- **Contents**: `Dataset Name.csv`
- **Structure**: 
  - Features in first N columns
  - Target variable in **last column**
  - Headers included
  - No preprocessing (raw data)

## Using Datasets in Benchmarks

The datasets integrate seamlessly with the benchmarking scripts:

```python
# In revised_benchmark/run_benchmark.py
python run_benchmark.py --dataset "Skin Segmentation"
python run_benchmark.py --dataset "HTRU2"
python run_benchmark.py --dataset "Dry Bean"
```

To add new datasets to the benchmark configuration, update `DATASET_PATHS` in `revised_benchmark/run_benchmark.py`:

```python
DATASET_PATHS = {
    'Adult Income': '../datasets/Adult Income.zip',
    'Skin Segmentation': '../datasets/Skin Segmentation.zip',
    'HTRU2': '../datasets/HTRU2.zip',
    # ... add more datasets
}
```

## Data Sources

Most datasets are from:
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle](https://www.kaggle.com/datasets)
- Original research publications

## Dataset Characteristics

### Binary Classification (13 datasets)
Best for testing depth-constrained methods on balanced/imbalanced problems.

### Multi-class Classification (8 datasets)
Tests generalization across multiple classes.

### Size Range
- Small (<1K): Good for quick testing
- Medium (1K-10K): Standard benchmarks
- Large (>10K): Computational stress tests

## Citation

If you use these datasets, please cite:

```
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
School of Information and Computer Science.
```

Individual datasets may have specific citation requirements - check the UCI repository for details.

## Troubleshooting

### Download Issues

**Problem**: Network timeout or 404 errors

**Solution**: 
```bash
# Install required packages
pip install requests pandas openpyxl scipy

# Retry download (script has automatic retry logic)
python download_datasets.py
```

**Problem**: ARFF parsing fails for EEG Eye State

**Solution**: Install scipy
```bash
pip install scipy
```

**Problem**: Excel file fails for Dry Bean

**Solution**: Install openpyxl
```bash
pip install openpyxl
```

### Validation Issues

**Problem**: "No CSV file found in ZIP"

**Solution**: Re-download the dataset or check the ZIP file manually

**Problem**: Encoding errors

**Solution**: The validation script handles most encodings automatically. For persistent issues, check the CSV file manually.

## Dataset-Specific Notes

### Segment
- First 5 rows are header comments (automatically skipped)
- Target class moved to last column during processing

### Vehicle
- Combined from 9 separate files
- Download script handles merging automatically

### Dry Bean
- Source format is Excel (.xlsx)
- Requires `openpyxl` package

### EEG Eye State
- Source format is ARFF
- Requires `scipy` package

### Occupancy Detection & Room Occupancy
- Both from same source ZIP
- Different files within the archive

## Maintenance

### Adding New Datasets

1. Add configuration to `download_datasets.py`:
```python
DATASETS = {
    'Your Dataset': {
        'url': 'https://...',
        'separator': ',',
        'header': None,
        'names': ['col1', 'col2', ..., 'Class'],
        'target_col': 'Class'
    }
}
```

2. Run download script
3. Validate with `validate_datasets.py`
4. Update `DATASET_PATHS` in benchmark scripts

### Updating Existing Datasets

1. Delete old ZIP file
2. Re-run download script
3. Validate changes

## File Structure

```
datasets/
├── README.md                    # This file
├── download_datasets.py         # Download script
├── validate_datasets.py         # Validation script
├── audit_datasets.py            # Audit script (for specific datasets)
├── DATASETS_CATALOG.md          # Generated catalog (after validation)
├── datasets_summary.csv         # Generated summary (after validation)
├── Adult Income.zip             # Dataset files
├── Bank Marketing.zip
├── Skin Segmentation.zip
└── ...
```

## Support

For issues with:
- **Download script**: Check UCI repository availability
- **Validation**: Ensure pandas and numpy are installed
- **Integration**: Check benchmark configuration

For dataset-specific questions, refer to the original UCI documentation.
