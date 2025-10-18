import os
import cv2 as cv
import numpy
import astropy.io.fits as fits
import torch
import torch.nn as nn
import torch.nn.functional as F

CUTOUT_DIR_SMALL = "fitssmall"
CUTOUT_DIR_BIG = "fits"
FAILED_LOG = "failed_upscale.txt"


def get_galaxy_ids():
    
    small_dirs = set(os.listdir(CUTOUT_DIR_SMALL))
    big_dirs = set(os.listdir(CUTOUT_DIR_BIG))
    
    return list(small_dirs.intersection(big_dirs))

def get_fits(dir, galaxy_id):
    galaxy_dir = os.path.join(dir, f"{galaxy_id}")
    
    bands = []
    

    path = os.path.join(galaxy_dir, f"{"SDSSr" if dir == CUTOUT_DIR_BIG else "DSS1 Red"}_{galaxy_id}.fits")
    
    if not os.path.exists(path) or os.path.getsize(path) < 6500:
        return None

    
        
    try:
        bands.append(fits.open(path)[0].data)
        stacked = numpy.stack(bands)
        stacked = numpy.moveaxis(stacked, 0, -1)
    except:
        return None

    return stacked


def build_dataset(ids):
    low_res_fits = []
    high_res_fits = []

    index = 0
    fails = 0

    with open(FAILED_LOG, "w") as log:
        for galaxy_id in ids:
            
            if fails > 10000:
                print("breaking due to fails")
                break
            
            low_res = get_fits(CUTOUT_DIR_SMALL, galaxy_id)
            high_res = get_fits(CUTOUT_DIR_BIG, galaxy_id)
            
            if low_res is None or high_res is None:
                fails += 1
                log.write(f"{galaxy_id}\n")
                continue

            low_res_fits.append(low_res)
            high_res_fits.append(high_res)

            index += 1
            
            if index % 100 == 0:
                print(f"Done with: {index}")

    low_res_data = numpy.stack(low_res_fits)
    high_res_data = numpy.stack(high_res_fits)

    return low_res_data, high_res_data

low_res, high_res = build_dataset(get_galaxy_ids())

print(low_res.shape)
print(high_res.shape)

cv.imshow("low res example", cv.normalize(low_res[0], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
cv.waitKey(0)

cv.imshow("high res example", cv.normalize(high_res[0], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
cv.waitKey(0)

cv.imshow("low to high res example", cv.normalize(cv.resize(low_res[0], (256, 256)), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
cv.waitKey(0)

cv.imshow("high to low res example", cv.normalize(cv.resize(high_res[0], (64, 64)), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
cv.waitKey(0)

cv.imshow("high to low to high res example", cv.normalize(cv.resize(cv.resize(high_res[0], (64, 64)), (256, 256)), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
cv.waitKey(0)

cv.destroyAllWindows()