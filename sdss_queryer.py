import os
import time
import requests
import pandas as pd

MAX_FAILS = 10
IMG_PIXELS = "256,256"
OUTPUT_DIR = "fits"
CHANNELS = ["SDSSu", "SDSSg", "SDSSr", "SDSSi", "SDSSz"]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

data = pd.read_csv("datasets/galaxies_categorical.csv")
coordinates = data[["objectID", "ra", "dec"]]

BASE_URL = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
failed_requests = []

for i, id, ra, dec in coordinates.itertuples():
    if i > 100:
        break

    cutout_dir = os.path.join(OUTPUT_DIR, f"{id}_cutout_{i+1:02d}")
    if not os.path.exists(cutout_dir):
        os.makedirs(cutout_dir)

    for channel in CHANNELS:
        filename = os.path.join(cutout_dir, f"{channel}_{ra:.4f}_{dec:.4f}.fits")
        if os.path.exists(filename):
            print(f"({i+1}/{len(coordinates)}) {channel} already exists for {id}, skipping")
            continue

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

        except requests.exceptions.RequestException as e:
            print(f"Request failed for ({ra}, {dec}) channel {channel}. Error: {e}")
            failed_requests.append(f"{ra},{dec},{channel}")

            if len(failed_requests) > MAX_FAILS:
                print("Ending early due to too many fails")
                break

        time.sleep(0.5)

if len(failed_requests) > 0:
    with open("failed.txt", "w") as output:
        for req in failed_requests:
            output.write(req + "\n")

print("\nTest download complete!")
