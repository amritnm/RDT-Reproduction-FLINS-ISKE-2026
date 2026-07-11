"""
Dataset Validation Script
Validates all datasets in the datasets folder and generates a catalog

Usage:
    python validate_datasets.py
"""

import os
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def validate_dataset(zip_path):
    """
    Validate a single dataset ZIP file.
    
    Returns:
        Dictionary with validation results
    """
    dataset_name = Path(zip_path).stem
    print(f"\n{'='*80}")
    print(f"Validating: {dataset_name}")
    print(f"{'='*80}")
    
    results = {
        'name': dataset_name,
        'valid': False,
        'error': None,
        'samples': 0,
        'features': 0,
        'classes': 0,
        'target_col': None,
        'has_missing': False,
        'has_duplicates': False,
        'categorical_features': 0,
        'numeric_features': 0,
        'file_size_mb': 0
    }
    
    try:
        # Check file exists
        if not os.path.exists(zip_path):
            results['error'] = 'File not found'
            return results
        
        # Get file size
        results['file_size_mb'] = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"  File size: {results['file_size_mb']:.2f} MB")
        
        # Extract and read CSV
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                results['error'] = 'No CSV file found in ZIP'
                print(f"  ERROR: {results['error']}")
                return results
            
            csv_file = csv_files[0]
            with zip_ref.open(csv_file) as f:
                df = pd.read_csv(f)
        
        # Basic information
        results['samples'] = len(df)
        results['features'] = len(df.columns) - 1  # Exclude target
        results['target_col'] = df.columns[-1]
        
        print(f"  Samples: {results['samples']:,}")
        print(f"  Features: {results['features']}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Target: '{results['target_col']}'")
        
        # Class information
        target_col = df.columns[-1]
        results['classes'] = df[target_col].nunique()
        print(f"  Classes: {results['classes']}")
        
        # Class distribution
        class_dist = df[target_col].value_counts().sort_index()
        print(f"  Class distribution:")
        for cls, count in class_dist.items():
            print(f"    {cls}: {count:,} ({count/len(df)*100:.1f}%)")
        
        # Check for missing values
        missing = df.isnull().sum()
        results['has_missing'] = missing.any()
        if results['has_missing']:
            print(f"  Missing values:")
            for col, count in missing[missing > 0].items():
                print(f"    '{col}': {count:,} ({count/len(df)*100:.1f}%)")
        else:
            print(f"  Missing values: None")
        
        # Check for duplicates
        n_duplicates = df.duplicated().sum()
        results['has_duplicates'] = n_duplicates > 0
        if results['has_duplicates']:
            print(f"  Duplicates: {n_duplicates:,} ({n_duplicates/len(df)*100:.1f}%)")
        else:
            print(f"  Duplicates: None")
        
        # Feature types
        X = df.iloc[:, :-1]
        results['numeric_features'] = len(X.select_dtypes(include=[np.number]).columns)
        results['categorical_features'] = len(X.select_dtypes(exclude=[np.number]).columns)
        print(f"  Feature types:")
        print(f"    Numeric: {results['numeric_features']}")
        print(f"    Categorical: {results['categorical_features']}")
        
        # Data types
        print(f"  Column data types:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"    {dtype}: {count}")
        
        # Sample data (first row)
        print(f"  Sample (first row):")
        for col in df.columns[:5]:  # Show first 5 columns
            val = df[col].iloc[0]
            print(f"    {col}: {val}")
        if len(df.columns) > 5:
            print(f"    ... ({len(df.columns) - 5} more columns)")
        
        # Validation passed
        results['valid'] = True
        print(f"  ✓ VALID")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"  ✗ ERROR: {results['error']}")
    
    return results


def generate_catalog(results_list, output_file='DATASETS_CATALOG.md'):
    """Generate markdown catalog of all datasets."""
    
    # Sort by number of samples
    results_list = sorted(results_list, key=lambda x: x['samples'], reverse=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Dataset Catalog\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Summary statistics
        valid_datasets = [r for r in results_list if r['valid']]
        invalid_datasets = [r for r in results_list if not r['valid']]
        
        f.write("## Summary\n\n")
        f.write(f"- **Total datasets**: {len(results_list)}\n")
        f.write(f"- **Valid datasets**: {len(valid_datasets)}\n")
        f.write(f"- **Invalid datasets**: {len(invalid_datasets)}\n")
        f.write(f"- **Total samples**: {sum(r['samples'] for r in valid_datasets):,}\n")
        f.write(f"- **Total features**: {sum(r['features'] for r in valid_datasets):,}\n\n")
        
        # Dataset table
        f.write("## Dataset Overview\n\n")
        f.write("| Dataset | Samples | Features | Classes | Type | Size (MB) |\n")
        f.write("|---------|---------|----------|---------|------|----------|\n")
        
        for r in valid_datasets:
            dataset_type = "Binary" if r['classes'] == 2 else "Multi-class"
            f.write(f"| {r['name']} | {r['samples']:,} | {r['features']} | {r['classes']} | {dataset_type} | {r['file_size_mb']:.2f} |\n")
        
        f.write("\n---\n\n")
        
        # Detailed information
        f.write("## Detailed Dataset Information\n\n")
        
        for r in valid_datasets:
            f.write(f"### {r['name']}\n\n")
            f.write(f"- **Samples**: {r['samples']:,}\n")
            f.write(f"- **Features**: {r['features']}\n")
            f.write(f"- **Classes**: {r['classes']}\n")
            f.write(f"- **Target column**: `{r['target_col']}`\n")
            f.write(f"- **Feature types**:\n")
            f.write(f"  - Numeric: {r['numeric_features']}\n")
            f.write(f"  - Categorical: {r['categorical_features']}\n")
            f.write(f"- **Missing values**: {'Yes' if r['has_missing'] else 'No'}\n")
            f.write(f"- **Duplicates**: {'Yes' if r['has_duplicates'] else 'No'}\n")
            f.write(f"- **File size**: {r['file_size_mb']:.2f} MB\n\n")
        
        # Invalid datasets
        if invalid_datasets:
            f.write("---\n\n")
            f.write("## Invalid Datasets\n\n")
            for r in invalid_datasets:
                f.write(f"### {r['name']}\n\n")
                f.write(f"- **Error**: {r['error']}\n\n")
        
        # Data sources
        f.write("---\n\n")
        f.write("## Data Sources\n\n")
        f.write("Most datasets are from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).\n\n")
        f.write("### Citation\n\n")
        f.write("If you use these datasets in your research, please cite:\n\n")
        f.write("```\n")
        f.write("Dua, D. and Graff, C. (2019). UCI Machine Learning Repository\n")
        f.write("[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,\n")
        f.write("School of Information and Computer Science.\n")
        f.write("```\n\n")


def generate_csv_summary(results_list, output_file='datasets_summary.csv'):
    """Generate CSV summary of all datasets."""
    df = pd.DataFrame(results_list)
    
    # Reorder columns
    cols = ['name', 'valid', 'samples', 'features', 'classes', 
            'numeric_features', 'categorical_features', 
            'has_missing', 'has_duplicates', 'file_size_mb', 'target_col', 'error']
    df = df[cols]
    
    # Sort by samples
    df = df.sort_values('samples', ascending=False)
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")


def main():
    """Main execution."""
    print("="*80)
    print("DATASET VALIDATION SCRIPT")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Find all ZIP files in current directory
    zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
    print(f"Found {len(zip_files)} ZIP files")
    
    if not zip_files:
        print("  No ZIP files found in current directory")
        print("  Make sure to run this script from the datasets/ folder")
        return
    
    # Validate each dataset
    results = []
    for zip_file in sorted(zip_files):
        result = validate_dataset(zip_file)
        results.append(result)
    
    # Generate outputs
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80)
    
    generate_catalog(results, 'DATASETS_CATALOG.md')
    print("  Generated: DATASETS_CATALOG.md")
    
    generate_csv_summary(results, 'datasets_summary.csv')
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    valid = [r for r in results if r['valid']]
    invalid = [r for r in results if not r['valid']]
    
    print(f"Valid datasets: {len(valid)}")
    for r in valid:
        print(f"  ✓ {r['name']} ({r['samples']:,} samples, {r['features']} features, {r['classes']} classes)")
    
    if invalid:
        print(f"\nInvalid datasets: {len(invalid)}")
        for r in invalid:
            print(f"  ✗ {r['name']}: {r['error']}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")


if __name__ == '__main__':
    main()
