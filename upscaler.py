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
    path = os.path.join(galaxy_dir, f"{"SDSSr" if dir == CUTOUT_DIR_BIG else "DSS1 Red"}_{galaxy_id}.fits")
    
    if not os.path.exists(path) or os.path.getsize(path) < 6500:
        return None

    try:
        bands = [fits.open(path)[0].data]
        stacked = numpy.stack(bands)
        stacked = numpy.moveaxis(stacked, 0, -1)
        stacked = numpy.moveaxis(stacked, -1, 0)
        return stacked
    except:
        return None


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def normalize_image(img, low_val, high_val):
    img_clipped = numpy.clip(img, low_val, high_val)
    return (img_clipped - low_val) / (high_val - low_val + 1e-8)


def downsample_fits(high_res, downsample_factor=4, add_noise=False, noise_level=0.01):
    h, w = high_res.shape[1], high_res.shape[2]
    new_h, new_w = h // downsample_factor, w // downsample_factor
    
    low_res = cv.GaussianBlur(high_res[0], (5, 5), 1.0)
    low_res = cv.resize(low_res, (new_w, new_h), interpolation=cv.INTER_AREA)
    
    if add_noise:
        noise = numpy.random.normal(0, noise_level, low_res.shape)
        low_res = numpy.clip(low_res + noise, 0, None)
    
    return low_res[numpy.newaxis, :, :]


def build_dataset(artifical_downsample=True):
    ids = get_galaxy_ids()
    low_res_fits = []
    high_res_fits = []
    fails = 0

    with open(FAILED_LOG, "w") as log:
        for idx, galaxy_id in enumerate(ids):
            if fails > 10000:
                print("Breaking due to excessive fails")
                break
            
            high_res = get_fits(CUTOUT_DIR_BIG, galaxy_id)
            
            if not artifical_downsample:
                low_res = get_fits(CUTOUT_DIR_SMALL, galaxy_id)
            else:
                low_res = downsample_fits(high_res, 4, True, 0.02) if high_res is not None else None

            if low_res is None or high_res is None:
                fails += 1
                log.write(f"{galaxy_id}\n")
                continue

            low_res_fits.append(low_res)
            high_res_fits.append(high_res)

            if (idx + 1) % 100 == 0:
                print(f"Loaded: {idx + 1} galaxies")
    
    all_low = numpy.concatenate([img.flatten() for img in low_res_fits])
    all_high = numpy.concatenate([img.flatten() for img in high_res_fits])
    
    low_min, low_max = numpy.percentile(all_low, [1, 99.5])
    high_min, high_max = numpy.percentile(all_high, [1, 99.5])
    
    print(f"Low-res range:  [{low_min:.6f}, {low_max:.6f}]")
    print(f"High-res range: [{high_min:.6f}, {high_max:.6f}]")
    
    low_normalized = [normalize_image(img, low_min, low_max) for img in low_res_fits]
    high_normalized = [normalize_image(img, high_min, high_max) for img in high_res_fits]

    stats = {'low_min': low_min, 'low_max': low_max, 'high_min': high_min, 'high_max': high_max}
    numpy.save('normalization_stats.npy', stats)

    return numpy.stack(low_normalized), numpy.stack(high_normalized)


class CnnUpscaler(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.b1 = nn.BatchNorm2d(64)
        
        self.layers = nn.ModuleList([self.make_layer(64) for _ in range(12)])
        
        self.conv_up = nn.Conv2d(64, 64 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.upsampler = nn.PixelShuffle(scale_factor)
        
        self.conv_recon1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_recon2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv_final = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def make_layer(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        x_upscaled = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        
        x = self.relu(self.b1(self.conv1(x)))
        identity = x
        
        for layer in self.layers:
            x = x + layer(x)
            x = self.relu(x)
        
        x = x + identity
        x = self.upsampler(self.conv_up(x))
        x = self.relu(self.conv_recon1(x))
        x = self.relu(self.conv_recon2(x))
        x = self.conv_final(x)
        
        return x + x_upscaled


class ImageDataset(Dataset):
    def __init__(self, low_res_images, high_res_images, augment=True):
        self.low_res = low_res_images 
        self.high_res = high_res_images
        self.augment = augment
    
    def __len__(self):
        return len(self.low_res)
    
    def __getitem__(self, index):
        low = self.low_res[index]
        high = self.high_res[index]
        
        if self.augment:
            if numpy.random.rand() > 0.5:
                low = numpy.flip(low, axis=2).copy()
                high = numpy.flip(high, axis=2).copy()
            
            if numpy.random.rand() > 0.5:
                low = numpy.flip(low, axis=1).copy()
                high = numpy.flip(high, axis=1).copy()
            
            k = numpy.random.randint(0, 4)
            if k > 0:
                low = numpy.rot90(low, k, axes=(1, 2)).copy()
                high = numpy.rot90(high, k, axes=(1, 2)).copy()
        
        return torch.from_numpy(low).float(), torch.from_numpy(high).float()


class FancyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
    
    def forward(self, output, target):
        if output.shape != target.shape:
            min_h = min(output.shape[2], target.shape[2])
            min_w = min(output.shape[3], target.shape[3])
            output = output[:, :, :min_h, :min_w]
            target = target[:, :, :min_h, :min_w]
        
        loss_l1 = self.l1(output, target)
        loss_mse = self.mse(output, target)
        
        def gradient(img):
            dx = img[:, :, :, 1:] - img[:, :, :, :-1]
            dy = img[:, :, 1:, :] - img[:, :, :-1, :]
            return dx, dy
        
        out_dx, out_dy = gradient(output)
        tar_dx, tar_dy = gradient(target)
        loss_grad = self.l1(out_dx, tar_dx) + self.l1(out_dy, tar_dy)
        
        return 0.6 * loss_l1 + 0.2 * loss_mse + 0.2 * loss_grad


def save_data_check(dataset):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(5, 2, figsize=(8, 16))
    for i in range(5):
        axes[i, 0].imshow(dataset.low_res[i, 0], cmap='gray')
        axes[i, 0].set_title(f'Low Res {i}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(dataset.high_res[i, 0], cmap='gray')
        axes[i, 1].set_title(f'High Res {i}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('data_check.png')
    print("Saved data_check.png")


def train():
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    compute_device = torch.device(device_type)
    
    model = CnnUpscaler().to(compute_device)
    loss_model = FancyLoss()
    optimizer = Adam(model.parameters(), lr=0.0005)
    
    dataset = ImageDataset(*build_dataset())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
    
    save_data_check(dataset)
    
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    start = time.time()
    iterations = 50
    
    for iteration in range(iterations):
        model.train()
        total_loss = 0
        iteration_time = time.time()
        
        for low, high in dataloader:
            low_res = low.to(compute_device)
            high_res = high.to(compute_device)
            
            optimizer.zero_grad()
            
            with autocast(device_type):
                output = model(low_res)
                loss = loss_model(output, high_res)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        end = time.time()
        avg_time = (end - start) / (iteration + 1)
        remaining = avg_time * (iterations - iteration - 1)

        print(f'Iteration [{iteration + 1}/{iterations}], '
              f'Loss: {avg_loss:.4f}, '
              f'LR: {current_lr:.6f}, '
              f'Time: {end - iteration_time:.2f}s, '
              f'Total: {format_time(end - start)}, '
              f'Remaining: {format_time(remaining)}')
        
        if (iteration + 1) % 5 == 0:
            torch.save({
                'epoch': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"checkpoint_iteration_{iteration + 1}.pth")
    
    torch.save(model.state_dict(), "upscale_model.pth")


if __name__ == "__main__":
    train()