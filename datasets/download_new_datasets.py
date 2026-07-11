"""
Download and package three new high-dimensional binary classification datasets:
1. Phishing Websites (30 features, ~11K samples)
2. MADELON (500 features, 2000 samples)
3. Musk v2 (166 features, ~6.6K samples)

Each is saved as a zip file containing a CSV with the target as the last column.
"""

import os
import zipfile
import requests
import gzip
import io
import pandas as pd
import numpy as np
from scipy.io import arff
from datetime import datetime

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def download_bytes(url, timeout=60):
    print(f"  Downloading: {url}")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def save_to_zip(df, zip_name, csv_name):
    zip_path = os.path.join(OUTPUT_DIR, zip_name)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, csv_bytes)
    print(f"  Saved: {zip_path}")
    print(f"    Rows: {len(df)}, Columns: {len(df.columns)}, Target: '{df.columns[-1]}'")
    print(f"    Classes: {df.iloc[:, -1].nunique()} -> {sorted(df.iloc[:, -1].unique())}")
    print(f"    Features: {len(df.columns) - 1}")


# ============================================================
# 1. PHISHING WEBSITES
# ============================================================
def download_phishing():
    print("\n" + "="*60)
    print("1. Phishing Websites Dataset")
    print("="*60)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
    try:
        content = download_bytes(url)
        # Decode bytes to string first, then wrap in StringIO for loadarff
        content_str = content.decode('utf-8', errors='replace')
        data, meta = arff.loadarff(io.StringIO(content_str))
        df = pd.DataFrame(data)

        # Decode byte strings (ARFF sometimes returns bytes)
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].str.decode('utf-8')
                except Exception:
                    pass

        # The last column is 'Result' with values -1 (phishing) and 1 (legitimate)
        target_col = df.columns[-1]
        print(f"  Target column: '{target_col}', unique values: {df[target_col].unique()}")

        # Convert all columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Map target: -1 -> 0 (phishing), 1 -> 1 (legitimate)
        df[target_col] = df[target_col].map({-1: 0, 1: 1})
        df = df.dropna()
        df = df.astype(int)

        # Rename target to Class and move to last
        df = df.rename(columns={target_col: 'Class'})
        cols = [c for c in df.columns if c != 'Class'] + ['Class']
        df = df[cols]

        save_to_zip(df, 'Phishing Websites.zip', 'phishing_websites.csv')
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# 2. MADELON
# ============================================================
def download_madelon():
    print("\n" + "="*60)
    print("2. MADELON Dataset")
    print("="*60)

    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data"
    label_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels"

    try:
        data_content = download_bytes(data_url)
        label_content = download_bytes(label_url)

        X = pd.read_csv(io.StringIO(data_content.decode('utf-8')), sep=' ', header=None)
        # Remove any all-NaN columns (trailing spaces create empty col)
        X = X.dropna(axis=1, how='all')

        y = pd.read_csv(io.StringIO(label_content.decode('utf-8')), header=None, names=['Class'])

        # Labels are -1 and 1 -> map to 0 and 1
        y['Class'] = y['Class'].map({-1: 0, 1: 1})

        # Name feature columns
        X.columns = [f'feature_{i}' for i in range(X.shape[1])]

        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        df = df.dropna()

        print(f"  Data shape: {X.shape}, Labels shape: {y.shape}")
        save_to_zip(df, 'Madelon.zip', 'madelon.csv')
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        # Try alternative URL
        try:
            print("  Trying alternative source...")
            # NIPS2003 alternative
            data_url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data"
            label_url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.labels"
            data_content = download_bytes(data_url2)
            label_content = download_bytes(label_url2)

            X = pd.read_csv(io.StringIO(data_content.decode('utf-8')), sep=' ', header=None)
            X = X.dropna(axis=1, how='all')
            y = pd.read_csv(io.StringIO(label_content.decode('utf-8')), header=None, names=['Class'])
            y['Class'] = y['Class'].map({-1: 0, 1: 1})
            X.columns = [f'feature_{i}' for i in range(X.shape[1])]
            df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            df = df.dropna()
            save_to_zip(df, 'Madelon.zip', 'madelon.csv')
            return True
        except Exception as e2:
            print(f"  Alternative also failed: {e2}")
            return False


# ============================================================
# 3. MUSK v2
# ============================================================
def download_musk_v2():
    print("\n" + "="*60)
    print("3. Musk (Version 2) Dataset")
    print("="*60)

    def process_musk_df(df):
        """Process raw musk dataframe: drop name cols, name features, return clean df."""
        n_cols = df.shape[1]
        print(f"  Total columns in file: {n_cols}")
        # Drop first 2 (molecule_name, conformation_name), last col is class
        df = df.iloc[:, 2:]
        n_features = df.shape[1] - 1
        df.columns = [f'feature_{i}' for i in range(n_features)] + ['Class']
        print(f"  Class values: {sorted(df['Class'].unique())}")
        return df

    try:
        print("  Trying UCI static zip (clean2.data.Z, LZW compressed)...")
        from unlzw3 import unlzw
        content = download_bytes("https://archive.ics.uci.edu/static/public/75/musk+version+2.zip")

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            print(f"  Zip contents: {zf.namelist()}")
            # The .Z file is LZW (Unix compress) format
            z_files = [f for f in zf.namelist() if f.endswith('.Z')]
            if not z_files:
                raise ValueError("No .Z file found in zip")
            print(f"  Decompressing: {z_files[0]}")
            compressed_bytes = zf.read(z_files[0])

        # Decompress LZW
        raw_bytes = unlzw(compressed_bytes)
        raw_text = raw_bytes.decode('utf-8', errors='replace')

        df = pd.read_csv(io.StringIO(raw_text), header=None)
        df = process_musk_df(df)
        save_to_zip(df, 'Musk v2.zip', 'musk_v2.csv')
        return True
    except Exception as e:
        print(f"  UCI zip (.Z) failed: {e}")

    # OpenML fallback - try different dataset IDs
    openml_urls = [
        "https://api.openml.org/data/v1/download/1590406",
        "https://www.openml.org/data/get_csv/21552979/musk.arff",
        "https://api.openml.org/data/v1/download/22103439",
    ]
    for ourl in openml_urls:
        try:
            print(f"  Trying OpenML: {ourl}")
            content = download_bytes(ourl)
            text = content.decode('utf-8', errors='replace')
            # Try as CSV first
            try:
                df = pd.read_csv(io.StringIO(text))
            except Exception:
                # Try as ARFF
                data, meta = arff.loadarff(io.StringIO(text))
                df = pd.DataFrame(data)
                for col in df.columns:
                    if df[col].dtype == object:
                        try:
                            df[col] = df[col].str.decode('utf-8')
                        except Exception:
                            pass
            print(f"  Shape: {df.shape}, columns[-3:]: {list(df.columns[-3:])}")
            df = df.rename(columns={df.columns[-1]: 'Class'})
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            save_to_zip(df, 'Musk v2.zip', 'musk_v2.csv')
            return True
        except Exception as e2:
            print(f"  Failed: {e2}")

    print("  All Musk v2 URLs failed.")
    return False


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print(f"Dataset Download Script")
    print(f"Timestamp: {datetime.now()}")
    print(f"Output directory: {OUTPUT_DIR}")

    results = {}
    results['Phishing Websites'] = download_phishing()
    results['Madelon'] = download_madelon()
    results['Musk v2'] = download_musk_v2()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  [{status}]: {name}")
