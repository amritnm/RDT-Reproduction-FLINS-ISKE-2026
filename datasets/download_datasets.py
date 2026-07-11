"""
Dataset Download Script
Downloads 20 datasets from UCI ML Repository and packages them as ZIP files

Usage:
    python download_datasets.py
    
Requirements:
    pip install requests pandas
"""

import os
import zipfile
import requests
import pandas as pd
import time
from io import StringIO
from datetime import datetime


# Dataset configurations
DATASETS = {
    'Skin Segmentation': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt',
        'separator': '\t',
        'header': None,
        'names': ['B', 'G', 'R', 'Class'],
        'target_col': 'Class'
    },
    'Avila': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila-tr.txt',
        'separator': ',',
        'header': None,
        'names': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'Class'],
        'target_col': 'Class'
    },
    'Occupancy Detection': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip',
        'is_zip': True,
        'csv_file': 'datatraining.txt',
        'separator': ',',
        'header': 0,
        'target_col': 'Occupancy'
    },
    'Magic Gamma Telescope': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data',
        'separator': ',',
        'header': None,
        'names': ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'Class'],
        'target_col': 'Class'
    },
    'HTRU2': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip',
        'is_zip': True,
        'csv_file': 'HTRU_2.csv',
        'separator': ',',
        'header': None,
        'names': ['Mean_IP', 'SD_IP', 'EK_IP', 'Skew_IP', 'Mean_DMSNR', 'SD_DMSNR', 'EK_DMSNR', 'Skew_DMSNR', 'Class'],
        'target_col': 'Class'
    },
    'EEG Eye State': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff',
        'format': 'arff',
        'target_col': 'Class'
    },
    'Dry Bean': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip',
        'is_zip': True,
        'csv_file': 'DryBeanDataset/Dry_Bean_Dataset.xlsx',
        'format': 'excel',
        'target_col': 'Class'
    },
    'Page Blocks': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/page-blocks/page-blocks.data',
        'separator': r'\s+',
        'header': None,
        'names': ['height', 'length', 'area', 'eccen', 'p_black', 'p_and', 'mean_tr', 'blackpix', 'blackand', 'wb_trans', 'Class'],
        'target_col': 'Class'
    },
    'Spambase': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
        'separator': ',',
        'header': None,
        'names': [f'word_freq_{i}' for i in range(48)] + 
                 [f'char_freq_{i}' for i in range(6)] + 
                 ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'Class'],
        'target_col': 'Class'
    },
    'Satellite': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn',
        'separator': r'\s+',
        'header': None,
        'names': [f'feature_{i}' for i in range(36)] + ['Class'],
        'target_col': 'Class'
    },
    'Wall-Following Robot': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00194/sensor_readings_24.data',
        'separator': ',',
        'header': None,
        'names': [f'US_{i}' for i in range(24)] + ['Class'],
        'target_col': 'Class'
    },
    'Segment': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data',
        'separator': ',',
        'header': None,
        'skiprows': 5,
        'names': ['Class', 'region-centroid-col', 'region-centroid-row', 'region-pixel-count', 
                 'short-line-density-5', 'short-line-density-2', 'vedge-mean', 'vegde-sd',
                 'hedge-mean', 'hedge-sd', 'intensity-mean', 'rawred-mean', 'rawblue-mean',
                 'rawgreen-mean', 'exred-mean', 'exblue-mean', 'exgreen-mean', 'value-mean',
                 'saturation-mean', 'hue-mean'],
        'target_col': 'Class',
        'move_target_to_end': True
    },
    'Steel Plates Faults': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA',
        'separator': '\t',
        'header': None,
        'names': [f'feature_{i}' for i in range(27)] + ['Class'],
        'target_col': 'Class'
    },
    'QSAR Biodegradation': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv',
        'separator': ';',
        'header': 0,
        'target_col': 'class'
    },
    'Pima Indians Diabetes': {
        'url': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
        'separator': ',',
        'header': None,
        'names': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Class'],
        'target_col': 'Class'
    },
    'Breast Cancer': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
        'separator': ',',
        'header': None,
        'names': ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(30)],
        'drop_cols': ['ID'],
        'target_col': 'Diagnosis'
    },
    'Ionosphere': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data',
        'separator': ',',
        'header': None,
        'names': [f'feature_{i}' for i in range(34)] + ['Class'],
        'target_col': 'Class'
    },
    'Vehicle': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaa.dat',
        'separator': r'\s+',
        'header': None,
        'names': [f'feature_{i}' for i in range(18)] + ['Class'],
        'target_col': 'Class',
        'combine_files': [
            'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaa.dat',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xab.dat',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xac.dat',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xad.dat',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xae.dat',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaf.dat',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xag.dat',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xah.dat',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xai.dat'
        ]
    }
}

# Manual additions for datasets requiring special handling
MANUAL_DATASETS = {
    'Room Occupancy': {
        'note': 'Same as Occupancy Detection - use different file from same ZIP'
    },
    'Online Bidding': {
        'note': 'Requires manual download or alternative source'
    }
}


def download_file(url, retries=3):
    """Download file with retry logic."""
    for attempt in range(retries):
        try:
            print(f"  Downloading from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise
    return None


def parse_arff(content):
    """Parse ARFF file format to DataFrame."""
    try:
        from scipy.io import arff
        import io
        data, meta = arff.loadarff(io.BytesIO(content))
        df = pd.DataFrame(data)
        # Decode byte strings
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].str.decode('utf-8')
                except:
                    pass
        return df
    except ImportError:
        print("  WARNING: scipy not installed, cannot parse ARFF. Install with: pip install scipy")
        return None


def download_dataset(name, config):
    """Download and process a single dataset."""
    print(f"\n{'='*80}")
    print(f"Processing: {name}")
    print(f"{'='*80}")
    
    try:
        # Handle multiple files (like Vehicle dataset)
        if 'combine_files' in config:
            print(f"  Combining {len(config['combine_files'])} files...")
            dfs = []
            for url in config['combine_files']:
                content = download_file(url)
                df_part = pd.read_csv(
                    StringIO(content.decode('utf-8')),
                    sep=config.get('separator', ','),
                    header=config.get('header'),
                    names=config.get('names'),
                    skiprows=config.get('skiprows', 0)
                )
                dfs.append(df_part)
            df = pd.concat(dfs, ignore_index=True)
        
        # Handle ZIP files
        elif config.get('is_zip', False):
            content = download_file(config['url'])
            
            # Save temporarily
            temp_zip = 'temp_download.zip'
            with open(temp_zip, 'wb') as f:
                f.write(content)
            
            # Extract
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall('temp_extract')
            
            # Find CSV file
            csv_path = os.path.join('temp_extract', config['csv_file'])
            
            if config.get('format') == 'excel':
                df = pd.read_excel(csv_path)
            else:
                df = pd.read_csv(
                    csv_path,
                    sep=config.get('separator', ','),
                    header=config.get('header'),
                    names=config.get('names'),
                    skiprows=config.get('skiprows', 0)
                )
            
            # Cleanup
            os.remove(temp_zip)
            import shutil
            shutil.rmtree('temp_extract')
        
        # Handle ARFF format
        elif config.get('format') == 'arff':
            content = download_file(config['url'])
            df = parse_arff(content)
            if df is None:
                print(f"  SKIPPED: Cannot parse ARFF without scipy")
                return False
        
        # Handle regular CSV/TXT
        else:
            content = download_file(config['url'])
            df = pd.read_csv(
                StringIO(content.decode('utf-8')),
                sep=config.get('separator', ','),
                header=config.get('header'),
                names=config.get('names'),
                skiprows=config.get('skiprows', 0),
                engine='python'  # More flexible parser
            )
        
        # Drop unwanted columns
        if 'drop_cols' in config:
            df = df.drop(columns=config['drop_cols'])
        
        # Move target to last column if needed
        target_col = config['target_col']
        if config.get('move_target_to_end', False) or df.columns[-1] != target_col:
            cols = [col for col in df.columns if col != target_col] + [target_col]
            df = df[cols]
        
        # Ensure target is last column
        if df.columns[-1] != target_col:
            cols = [col for col in df.columns if col != target_col] + [target_col]
            df = df[cols]
        
        print(f"  Shape: {df.shape[0]} samples, {df.shape[1]} columns")
        print(f"  Target: '{target_col}' (at position {len(df.columns)-1})")
        print(f"  Classes: {df[target_col].nunique()}")
        
        # Save as CSV
        csv_filename = f"{name}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"  Saved: {csv_filename}")
        
        # Create ZIP
        zip_filename = f"{name}.zip"
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_filename)
        print(f"  Packaged: {zip_filename}")
        
        # Cleanup CSV
        os.remove(csv_filename)
        
        print(f"  SUCCESS!")
        return True
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution."""
    print("="*80)
    print("DATASET DOWNLOAD SCRIPT")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets to download: {len(DATASETS)}")
    print("")
    
    # Track results
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    # Process each dataset
    for name, config in DATASETS.items():
        success = download_dataset(name, config)
        
        if success:
            results['success'].append(name)
        else:
            results['failed'].append(name)
        
        # Brief pause between downloads
        time.sleep(1)
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"Successful: {len(results['success'])}")
    for name in results['success']:
        print(f"  ✓ {name}")
    
    if results['failed']:
        print(f"\nFailed: {len(results['failed'])}")
        for name in results['failed']:
            print(f"  ✗ {name}")
    
    if results['skipped']:
        print(f"\nSkipped: {len(results['skipped'])}")
        for name in results['skipped']:
            print(f"  - {name}")
    
    # Manual datasets note
    if MANUAL_DATASETS:
        print(f"\nManual datasets (require special handling): {len(MANUAL_DATASETS)}")
        for name, info in MANUAL_DATASETS.items():
            print(f"  - {name}: {info['note']}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")


if __name__ == '__main__':
    main()
