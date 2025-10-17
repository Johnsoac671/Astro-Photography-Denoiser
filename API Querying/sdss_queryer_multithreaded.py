import os
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

MAX_FAILS = 20
IMAGE_COUNT = 2500
IMG_PIXELS = "64,64"
OUTPUT_DIR = "./fitssmall"
# CHANNELS = ["SDSSg", "SDSSr", "SDSSi"]

CHANNELS = ["DSS1 Red"]

MAX_WORKERS = 6
RATE_LIMIT_DELAY = 0.5

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

data = pd.read_csv("./datasets/galaxies_categorical.csv")
coordinates = data[["objectID", "ra", "dec"]]

BASE_URL = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"

failed_requests = []
failed_lock = Lock()
last_request_time = time.time()
time_lock = Lock()

def download_fits(i, id, ra, dec, channel):
    global last_request_time
    
    cutout_dir = os.path.join(OUTPUT_DIR, f"{id}")
    if not os.path.exists(cutout_dir):
        os.makedirs(cutout_dir, exist_ok=True)
    
    filename = os.path.join(cutout_dir, f"{channel}_{id}.fits")
    if os.path.exists(filename):
        print(f"({i+1}/{len(coordinates)}) {channel} already exists for {id}, skipping")
        return True
    
    with time_lock:
        elapsed = time.time() - last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        last_request_time = time.time()
    
    query_params = {
        'Survey': channel,
        'Position': f'{ra},{dec}',
        'Pixels': IMG_PIXELS,
        'Return': 'FITS'
    }
    
    try:
        response = requests.post(BASE_URL, data=query_params)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"({i+1}/{len(coordinates)}) {channel} downloaded for {id}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Request failed for ({ra}, {dec}) channel {channel}. Error: {e}")
        with failed_lock:
            failed_requests.append(f"{ra},{dec},{channel}")
        return False

tasks = []
for i, id, ra, dec in coordinates.itertuples():
    if i >= IMAGE_COUNT:
        break
    for channel in CHANNELS:
        tasks.append((i, id, ra, dec, channel))

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(download_fits, *task): task for task in tasks}
    
    for future in as_completed(futures):
        if len(failed_requests) > MAX_FAILS:
            print("Ending early due to too many fails")
            executor.shutdown(wait=False, cancel_futures=True)
            break

if len(failed_requests) > 0:
    with open("failed.txt", "w") as output:
        for req in failed_requests:
            output.write(req + "\n")

print("\nTest download complete!")