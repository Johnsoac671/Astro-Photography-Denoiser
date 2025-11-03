import os
import cv2 as cv
import numpy
import astropy.io.fits as fits
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.amp import autocast, GradScaler


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
        
        stacked = numpy.moveaxis(stacked, -1, 0)
    except:
        return None

    return stacked

def format_time(seconds):
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def normalize_image(img):
    low, high = numpy.percentile(img, [1, 99])
    img_clipped = numpy.clip(img, low, high)
    return (img_clipped - img_clipped.min()) / (img_clipped.max() - img_clipped.min() + 1e-8)


def build_dataset():
    ids = get_galaxy_ids()
    
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
    
    for i in range(len(low_res_fits)):
        low_res_fits[i] = normalize_image(low_res_fits[i])
        high_res_fits[i] = normalize_image(high_res_fits[i])

    low_res_data = numpy.stack(low_res_fits)
    high_res_data = numpy.stack(high_res_fits)

    return low_res_data, high_res_data


class CnnUpscaler(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.b1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.b5 = nn.BatchNorm2d(64)
        
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(32, scale_factor**2, kernel_size=3, padding=1)
        
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.b1(self.conv1(x)))
        x = self.relu(self.b2(self.conv2(x)))
        x = self.relu(self.b3(self.conv3(x)))
        x = self.relu(self.b4(self.conv4(x)))
        x = self.relu(self.b5(self.conv5(x)))
        x = self.relu(self.conv6(x))
        x = self.conv7(x)
        x = self.pixel_shuffle(x)
        
        return x
    

class ImageDataset(Dataset):
    def __init__(self, low_res_images, high_res_images):
        self.low_res = low_res_images 
        self.high_res = high_res_images 
    
    def __len__(self):
        return len(self.low_res)
    
    def __getitem__(self, index):
        low = self.low_res[index]
        high = self.high_res[index]
        return torch.from_numpy(low).float(), torch.from_numpy(high).float()


if __name__ == "__main__":
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    compute_device = torch.device(device_type)
    
    
    model = CnnUpscaler().to(compute_device)
    loss_model = nn.MSELoss()
    optimizer_model = Adam(model.parameters(), lr=0.001)

    dataset = ImageDataset(*build_dataset())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
    
    scaler = GradScaler()
    
    start = time.time()
    iterations = 50
    
    for iteration in range(iterations):
        model.train()
        total_loss = 0
        iteration_time = time.time()
        
        for low, high in dataloader:
            low_res = low.to(compute_device)
            high_res = high.to(compute_device)
            
            optimizer_model.zero_grad()
            
            with autocast(device_type):
                output = model(low_res)
                loss = loss_model(output, high_res)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer_model)
            scaler.update()
        
            total_loss += loss.item()
            
        end = time.time()

        avg_time = (end - start) / (iteration + 1)
        remaining = avg_time * (iterations - iteration - 1)

        print(f'Iteration [{iteration + 1}/{iterations}], '
            f'Loss: {total_loss/len(dataloader):.4f}, '
            f'Iteration Time: {end - iteration_time:.2f}s, '
            f'Total Time: {format_time(end - start)}, '
            f'Estimated Remaining: {format_time(remaining)}')
        
        if (iteration + 1) % 5 == 0:
            torch.save({'epoch': iteration,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer_model.state_dict(),
                        'loss': total_loss/len(dataloader), },
                       f"checkpoint_epoch_{iteration + 1}.pth")
    
    torch.save(model.state_dict(), "upscale_model.pth")