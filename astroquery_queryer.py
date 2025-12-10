import os
import shutil
import requests
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
from concurrent.futures import ThreadPoolExecutor

# --- CONFIG ---
OUTPUT_DIR = "fits2"
TEMP_DIR = "temp_frames"  # Created in current dir (D:)
CROP_SIZE = 256
MAX_WORKERS = 4
BAND = 'r'
RERUN = '301' # Standard SDSS rerun for legacy imaging

# --- FILTERING CONFIG ---
# Set FILTER_FEATURE to None to download all galaxies
# Set FILTER_FEATURE to a column name and FILTER_VALUE to filter
FILTER_FEATURE = "edgeon"  # e.g., "edgeon", "morphology", None
FILTER_VALUE = "yes"       # e.g., "yes", "smooth", "spiral", etc.
# ----------------

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def get_field_info(objid):
    objid = int(objid)
    run = (objid >> 32) & 0xFFFF
    camcol = (objid >> 29) & 0x7
    field = (objid >> 16) & 0xFFF
    return run, camcol, field

def download_file(url, local_path):
    """Downloads a file to the specified path."""
    try:
        with requests.get(url, stream=True, timeout=15) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Download error {url}: {e}")
        return False

def process_field_group(run, camcol, field, group):
    temp_file_path = None
    try:
        # --- OPTIMIZATION: Check if we even need this field ---
        all_done = True
        for row in group.itertuples():
            save_path = os.path.join(OUTPUT_DIR, str(row.objectID), f"SDSS{BAND}_{row.objectID}.fits")
            if not os.path.exists(save_path):
                all_done = False
                break
        
        if all_done:
            return # Skip download entirely
        
        # --- 1. Construct SDSS SAS URL Manually ---
        # Pattern: https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/[run]/[camcol]/frame-[band]-[run6]-[camcol]-[field4].fits.bz2
        # Run is zero-padded to 6 digits, Field to 4 digits in filename.
        filename = f"frame-{BAND}-{run:06d}-{camcol}-{field:04d}.fits.bz2"
        # Try DR17 path (standard for current data)
        url = f"https://data.sdss.org/sas/dr17/eboss/photoObj/frames/{RERUN}/{run}/{camcol}/{filename}"
        
        temp_file_path = os.path.join(TEMP_DIR, filename)
        
        # --- 2. Download to D: Drive (Temp) ---
        if not os.path.exists(temp_file_path):
            success = download_file(url, temp_file_path)
            if not success:
                # Fallback: Try DR16 path if DR17 fails (rare, but happens for some plates)
                url = f"https://data.sdss.org/sas/dr16/eboss/photoObj/frames/{RERUN}/{run}/{camcol}/{filename}"
                print(f"Retrying with DR16: {run}-{camcol}-{field}")
                if not download_file(url, temp_file_path):
                    return # Give up on this field

        # --- 3. Open, Crop, and Save ---
        # fits.open handles .bz2 transparently
        try:
            with fits.open(temp_file_path) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                wcs = WCS(header)
                
                for row in group.itertuples():
                    obj_id = row.objectID
                    
                    save_dir = os.path.join(OUTPUT_DIR, str(obj_id))
                    save_path = os.path.join(save_dir, f"SDSS{BAND}_{obj_id}.fits")
                    
                    if os.path.exists(save_path):
                        continue
                    
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                        
                    position = SkyCoord(row.ra, row.dec, unit=u.deg)
                    
                    try:
                        cutout = Cutout2D(data, position, (CROP_SIZE, CROP_SIZE), wcs=wcs, mode='partial', fill_value=0)
                        hdu_cutout = fits.PrimaryHDU(data=cutout.data, header=cutout.wcs.to_header())
                        hdu_cutout.writeto(save_path, overwrite=True)
                    except Exception as e:
                        print(f"Cutout error {obj_id}: {e}")
                        
            print(f"Processed Field {run}-{camcol}-{field}")

        except Exception as e:
            print(f"Corrupt FITS/Processing error {filename}: {e}")

    except Exception as e:
        print(f"Fatal field error {run}-{camcol}-{field}: {e}")
    finally:
        # --- 4. Delete Temp File Immediately ---
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

def main():
    df = pd.read_csv("datasets/galaxies_categorical.csv")
    
    # --- Apply Filter if Configured ---
    original_count = len(df)
    if FILTER_FEATURE is not None and FILTER_FEATURE in df.columns:
        df = df[df[FILTER_FEATURE] == FILTER_VALUE]
        filtered_count = len(df)
        print(f"Filter applied: {FILTER_FEATURE} = {FILTER_VALUE}")
        print(f"Galaxies: {filtered_count} / {original_count} ({filtered_count/original_count*100:.1f}%)")
        
        if filtered_count == 0:
            print("No galaxies match the filter criteria. Exiting.")
            return
    else:
        if FILTER_FEATURE is not None:
            print(f"Warning: Filter feature '{FILTER_FEATURE}' not found in CSV. Downloading all galaxies.")
        else:
            print("No filter applied. Downloading all galaxies.")
        print(f"Total galaxies: {original_count}")
    
    # Pre-calculate Run/Camcol/Field from ObjectID
    print("Grouping galaxies...")
    df['run'], df['camcol'], df['field'] = zip(*df['objectID'].apply(get_field_info))
    
    groups = df.groupby(['run', 'camcol', 'field'])
    
    tasks = []
    for (run, camcol, field), group in groups:
        tasks.append((run, camcol, field, group))
        
    print(f"Processing {len(tasks)} unique fields...")
    print(f"Using temp directory: {os.path.abspath(TEMP_DIR)}")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_field_group, *task) for task in tasks]
        # Wait for all
        for future in futures:
            future.result()

    # Final cleanup
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except:
            pass
    print("Done!")

if __name__ == "__main__":
    main()