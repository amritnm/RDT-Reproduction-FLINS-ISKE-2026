"""Standalone Musk v2 downloader with log output."""
import sys, io, zipfile, requests, pandas as pd, os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(OUTPUT_DIR, 'musk_download.log'), 'w', encoding='utf-8')

def log(msg):
    print(msg)
    LOG.write(msg + '\n')
    LOG.flush()

log('=== Musk v2 Download ===')

try:
    from unlzw3 import unlzw
    log('unlzw3 imported OK')

    log('Downloading UCI zip...')
    r = requests.get('https://archive.ics.uci.edu/static/public/75/musk+version+2.zip', timeout=120)
    log(f'Status: {r.status_code}, Size: {len(r.content)} bytes')

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
        log(f'Zip contents: {names}')
        z_files = [f for f in names if f.endswith('.Z')]
        log(f'.Z files: {z_files}')
        if not z_files:
            log('ERROR: No .Z file found!')
            sys.exit(1)
        compressed = zf.read(z_files[0])
        log(f'Compressed size: {len(compressed)}')

    log('Decompressing LZW...')
    raw = unlzw(compressed)
    log(f'Decompressed size: {len(raw)}')
    text = raw.decode('utf-8', errors='replace')
    log(f'First 300 chars:\n{repr(text[:300])}')

    log('Parsing CSV...')
    df = pd.read_csv(io.StringIO(text), header=None)
    log(f'Raw shape: {df.shape}')
    log(f'First row: {list(df.iloc[0][:5])}')

    # Drop first 2 name cols, last is class
    df = df.iloc[:, 2:]
    n_features = df.shape[1] - 1
    df.columns = [f'feature_{i}' for i in range(n_features)] + ['Class']
    log(f'Processed shape: {df.shape}')
    log(f'Class values: {sorted(df["Class"].unique())}')

    # Save to zip
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    zip_path = os.path.join(OUTPUT_DIR, 'Musk v2.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as out_zf:
        out_zf.writestr('musk_v2.csv', csv_bytes)
    log(f'Saved: {zip_path}')
    log('SUCCESS')

except Exception as e:
    import traceback
    log(f'ERROR: {e}')
    traceback.print_exc(file=LOG)
    log('FAILED')

LOG.close()
