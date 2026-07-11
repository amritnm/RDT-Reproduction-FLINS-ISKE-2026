# Dataset Download Instructions

## Manual Execution Guide

Run these commands in your terminal to download and validate the datasets.

### Step 1: Install Dependencies

```bash
pip install requests pandas openpyxl scipy
```

### Step 2: Navigate to Datasets Folder

```bash
cd c:\Users\amrit\Documents\project_rdt\datasets
```

### Step 3: Run Download Script

```bash
python download_datasets.py
```

**Expected behavior:**
- Downloads 18 datasets from UCI ML Repository
- Some may fail due to URL changes (404 errors) - this is normal
- Script will continue with remaining datasets
- Each successful dataset creates a `.zip` file

**Note:** The script may take 5-10 minutes and some datasets may fail. This is expected.

### Step 4: Validate Downloaded Datasets

```bash
python validate_datasets.py
```

**This generates:**
- `DATASETS_CATALOG.md` - Human-readable catalog
- `datasets_summary.csv` - Machine-readable summary

---

## Known Issues & Fixes

### Issue 1: Some Datasets Return 404 Errors

**Affected datasets:** Avila, potentially others

**Why:** UCI repository URLs change over time

**Solution:** These datasets can be manually downloaded if needed, or use alternative sources. The script will continue with other datasets.

### Issue 2: EEG Eye State ARFF Parsing Error

**Error:** `TypeError: cannot use a string pattern on a bytes-like object`

**Why:** scipy ARFF parser compatibility issue

**Solution:** Skip this dataset or manually convert ARFF to CSV

### Issue 3: Dry Bean Excel File

**Requires:** `openpyxl` package

**If it fails:**
```bash
pip install openpyxl
```

---

## Alternative: Download Selected Datasets Only

If you want to download specific datasets, edit `download_datasets.py` and comment out datasets you don't need:

```python
DATASETS = {
    'Skin Segmentation': {...},  # Keep this one
    # 'Avila': {...},  # Comment out this one
    'Magic Gamma Telescope': {...},  # Keep this one
    # ... etc
}
```

---

## Manual Download URLs (for failed datasets)

If automated download fails, here are direct links:

### Avila
- URL: https://archive.ics.uci.edu/dataset/459/avila
- Download manually and place in datasets folder

### EEG Eye State  
- URL: https://archive.ics.uci.edu/dataset/264/eeg+eye+state
- Download manually and place in datasets folder

### Alternative: Use UCI's New Interface
The UCI ML Repository recently changed their URL structure. Visit:
https://archive.ics.uci.edu/datasets

Search for the dataset name and download manually.

---

## Quick Verification

Check which datasets were successfully downloaded:

```bash
dir *.zip
```

Or on Linux/Mac:
```bash
ls *.zip
```

You should see multiple `.zip` files, each containing a dataset.

---

## Next Steps After Download

1. **Validate datasets:**
   ```bash
   python validate_datasets.py
   ```

2. **Check the catalog:**
   ```bash
   type DATASETS_CATALOG.md
   ```
   (Or `cat DATASETS_CATALOG.md` on Linux/Mac)

3. **View summary:**
   ```bash
   type datasets_summary.csv
   ```

4. **Update benchmark script** (see below)

---

## Updating Benchmark Configuration

After downloading, update `revised_benchmark/run_benchmark.py` to include new datasets:

```python
DATASET_PATHS = {
    # Existing datasets
    'Adult Income': '../datasets/Adult Income.zip',
    'Bank Marketing': '../datasets/Bank Marketing.zip',
    'Cover Type': '../datasets/Cover Type.zip',
    'German Credit': '../datasets/German Credit.zip',
    'HELOC': '../datasets/HELOC.zip',
    'Heart Disease': '../datasets/Heart Disease.zip',
    
    # New datasets (add those that downloaded successfully)
    'Skin Segmentation': '../datasets/Skin Segmentation.zip',
    'Magic Gamma Telescope': '../datasets/Magic Gamma Telescope.zip',
    'HTRU2': '../datasets/HTRU2.zip',
    'Occupancy Detection': '../datasets/Occupancy Detection.zip',
    'Page Blocks': '../datasets/Page Blocks.zip',
    'Spambase': '../datasets/Spambase.zip',
    'Satellite': '../datasets/Satellite.zip',
    'Wall-Following Robot': '../datasets/Wall-Following Robot.zip',
    'Segment': '../datasets/Segment.zip',
    'Steel Plates Faults': '../datasets/Steel Plates Faults.zip',
    'QSAR Biodegradation': '../datasets/QSAR Biodegradation.zip',
    'Pima Indians Diabetes': '../datasets/Pima Indians Diabetes.zip',
    'Breast Cancer': '../datasets/Breast Cancer.zip',
    'Ionosphere': '../datasets/Ionosphere.zip',
    'Vehicle': '../datasets/Vehicle.zip',
    'Dry Bean': '../datasets/Dry Bean.zip',
}
```

---

## Testing with Benchmark

Once datasets are downloaded and validated, test with:

```bash
cd c:\Users\amrit\Documents\project_rdt\revised_benchmark
python run_benchmark.py --dataset "Skin Segmentation"
```

---

## Troubleshooting

### Script hangs or takes too long
- Press `Ctrl+C` to stop
- Comment out large datasets (Skin Segmentation is 245K rows)
- Run validation on what was downloaded

### Permission errors
- Run terminal as Administrator
- Or use `--user` flag with pip

### Import errors
- Make sure you're using the correct Python environment
- Check: `python --version` (should be Python 3.8+)
- Verify packages: `pip list | findstr "requests pandas"`

### Network issues
- Check internet connection
- Some university/corporate networks block archive downloads
- Try using a VPN or different network

---

## Summary

**Minimum commands to run:**

```bash
# 1. Install dependencies
pip install requests pandas openpyxl scipy

# 2. Navigate to datasets folder
cd c:\Users\amrit\Documents\project_rdt\datasets

# 3. Download datasets
python download_datasets.py

# 4. Validate what was downloaded
python validate_datasets.py

# 5. Check results
dir *.zip
type DATASETS_CATALOG.md
```

That's it! Some datasets may fail to download due to URL changes, but most should work. The validation script will show you exactly what was successful.
