import os
import cv2 as cv
import numpy
import astropy.io.fits as fits
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.amp import autocast, GradScaler


CUTOUT_DIR_SMALL = "fitssmall" # directory for the 64x64 cutouts
CUTOUT_DIR_BIG = "fits2"    # directory for the 256x256 cutouts
ARTIFICIAL_DOWNSAMPLE = False # If True, downsample from big cutouts instead of loading small cutouts

BATCH_SIZE = 32 # Batch size for training
NUM_WORKERS = 6 # DataLoader workers
TRAINING_ITERATIONS = 50 # Number of training iterations (epochs)


NOISE_CORRUPTION_RATE = 0.0  # Fraction of pixels to corrupt (0.0 to 1.0)
NOISE_AMOUNT = 0.0  # Magnitude of noise relative to normalized range (0.0 to 1.0+)

AUTO_ALIGN_IMAGES = True  # Enable automatic alignment


def get_galaxy_ids(require_pairs=True):
    big_dirs = set(os.listdir(CUTOUT_DIR_BIG))
    
    if require_pairs:
        small_dirs = set(os.listdir(CUTOUT_DIR_SMALL))
        intersection = list(small_dirs.intersection(big_dirs))
        print(f"Found {len(intersection)} paired galaxies.")
        return intersection
    
    print(f"Found {len(big_dirs)} galaxies in {CUTOUT_DIR_BIG}.")
    return list(big_dirs)

def get_fits(base_dir, galaxy_id):
    galaxy_dir = os.path.join(base_dir, f"{galaxy_id}")

    filename = f"SDSSr_{galaxy_id}.fits" if base_dir == CUTOUT_DIR_BIG else f"DSS1 Red_{galaxy_id}.fits"
    path = os.path.join(galaxy_dir, filename)
    
    if not os.path.exists(path):
        return None

    try:
        with fits.open(path) as hdul:
            data = hdul[0].data

            if len(data.shape) == 2:
                data = data[numpy.newaxis, :, :] 
            
            if base_dir == CUTOUT_DIR_BIG:
                data = numpy.rot90(data, k=1, axes=(1, 2)) 
            
            return data.astype(numpy.float32)
    except Exception as e:
        return None

def normalize_image(img):
    low_val = numpy.percentile(img, 1)
    high_val = numpy.percentile(img, 99.5)
    
    img_clipped = numpy.clip(img, low_val, high_val)
    normalized = (img_clipped - low_val) / (high_val - low_val + 1e-8)
    
    return normalized

def add_corruption_noise(img, corruption_rate=0.3, noise_amount=0.5):
    if corruption_rate <= 0:
        return img
    
    noisy = img.copy()
    
    mask = numpy.random.rand(*img.shape) < corruption_rate
    
    noise = numpy.random.uniform(-noise_amount, noise_amount, img.shape)
    noisy[mask] = noisy[mask] + noise[mask]
    
    noisy = numpy.clip(noisy, 0, 1)
    
    return noisy

def downsample_fits(high_res, downsample_factor=4, add_noise=False, 
                   corruption_rate=0.3, noise_amount=0.5):
    c, h, w = high_res.shape
    new_h, new_w = h // downsample_factor, w // downsample_factor
    
    img_2d = high_res[0]
    
    low_res = cv.resize(img_2d, (new_w, new_h), interpolation=cv.INTER_AREA)
    
    if add_noise:
        low_res = add_corruption_noise(low_res, corruption_rate, noise_amount)
    
    low_res = normalize_image(low_res)
    
    return low_res[numpy.newaxis, :, :]


def align_images_phase_correlation(low_res, high_res):
    try:
        low_2d = low_res[0] if len(low_res.shape) == 3 else low_res
        high_2d = high_res[0] if len(high_res.shape) == 3 else high_res
        

        h, w = low_2d.shape
        high_downsampled = cv.resize(high_2d, (w, h), interpolation=cv.INTER_AREA)
        

        low_uint8 = (low_2d * 255).astype(numpy.uint8)
        high_uint8 = (high_downsampled * 255).astype(numpy.uint8)
        

        best_response = -1
        best_angle = 0
        best_shift = (0, 0)
        
        for angle in [0, 90, 180, 270]:

            if angle == 90:
                rotated = cv.rotate(high_uint8, cv.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv.rotate(high_uint8, cv.ROTATE_180)
            elif angle == 270:
                rotated = cv.rotate(high_uint8, cv.ROTATE_90_COUNTERCLOCKWISE)
            else:
                rotated = high_uint8
            
            shift, response = cv.phaseCorrelate(
                numpy.float32(low_uint8),
                numpy.float32(rotated)
            )
            
            if response > best_response:
                best_response = response
                best_angle = angle
                best_shift = shift
        

        if best_angle == 90:
            aligned = numpy.rot90(high_2d, k=3, axes=(0, 1))  
        elif best_angle == 180:
            aligned = numpy.rot90(high_2d, k=2, axes=(0, 1))
        elif best_angle == 270:
            aligned = numpy.rot90(high_2d, k=1, axes=(0, 1))
        else:
            aligned = high_2d
        

        shift_x, shift_y = best_shift
        if abs(shift_x) > 2 or abs(shift_y) > 2:  

            scale_factor = high_2d.shape[0] / low_2d.shape[0]
            shift_x *= scale_factor
            shift_y *= scale_factor
            
            M = numpy.float32([[1, 0, shift_x], [0, 1, shift_y]])
            aligned = cv.warpAffine(aligned, M, (aligned.shape[1], aligned.shape[0]))
        
        aligned = aligned[numpy.newaxis, :, :]
        
        return aligned, best_response > 0.1 
        
    except Exception as e:
        print(f"Phase correlation alignment failed: {e}")
        return high_res, False


def align_images(low_res, high_res, method='phase'):
    if method == 'phase':
        return align_images_phase_correlation(low_res, high_res)


def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100
    return 10 * torch.log10(1.0 / mse)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

class GalaxyDataset(Dataset):
    def __init__(self, galaxy_ids, augment=True, align=AUTO_ALIGN_IMAGES):
        self.galaxy_ids = galaxy_ids
        self.augment = augment
        self.align = align
        
    def __len__(self):
        return len(self.galaxy_ids)
    
    def __getitem__(self, index):
        attempts = 0
        while attempts < 3:
            try:
                galaxy_id = self.galaxy_ids[index]
                
                high_res = get_fits(CUTOUT_DIR_BIG, galaxy_id)
                if high_res is None: raise FileNotFoundError
                
                high_res = normalize_image(high_res)
                
                if ARTIFICIAL_DOWNSAMPLE:
                    low_res = downsample_fits(high_res, 4, True, 
                                            NOISE_CORRUPTION_RATE, NOISE_AMOUNT)
                else:
                    low_res = get_fits(CUTOUT_DIR_SMALL, galaxy_id)
                    if low_res is None: raise FileNotFoundError
                    low_res = normalize_image(low_res)
                    
                    if self.align:
                        aligned_high_res, success = align_images(
                            low_res, high_res, method="phase"
                        )
                        if success:
                            high_res = aligned_high_res
                            
                if self.augment:
                    if numpy.random.rand() > 0.5:
                        low_res = numpy.flip(low_res, axis=2).copy()
                        high_res = numpy.flip(high_res, axis=2).copy()
                    
                    if numpy.random.rand() > 0.5:
                        low_res = numpy.flip(low_res, axis=1).copy()
                        high_res = numpy.flip(high_res, axis=1).copy()
                    
                    k = numpy.random.randint(0, 4)
                    if k > 0:
                        low_res = numpy.rot90(low_res, k, axes=(1, 2)).copy()
                        high_res = numpy.rot90(high_res, k, axes=(1, 2)).copy()
                
                return torch.from_numpy(low_res).float(), torch.from_numpy(high_res).float()

            except (FileNotFoundError, Exception) as e:
                index = random.randint(0, len(self.galaxy_ids) - 1)
                attempts += 1
        
        return torch.zeros((1, 64, 64)), torch.zeros((1, 256, 256))

class CnnUpscaler(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
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
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        x_upscaled = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        x = self.relu(self.conv1(x))
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
        
        return 0.6 * loss_l1 + 0.4 * loss_mse 

def save_data_check(dataset):
    import matplotlib.pyplot as plt
    try:
        num_pairs = min(5, len(dataset))
        
        fig, axes = plt.subplots(num_pairs, 3, figsize=(12, 4 * num_pairs))
        if num_pairs == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_pairs):
            low, high = dataset[i]
            low = low.numpy()
            high = high.numpy()
            
            high_downsampled = cv.resize(high[0], (low.shape[2], low.shape[1]), 
                                        interpolation=cv.INTER_AREA)
            
            axes[i, 0].imshow(low[0], cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(f'Low Res - Pair {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(high[0], cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(f'High Res (Aligned) - Pair {i+1}')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(high_downsampled, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f'High Res Downsampled - Pair {i+1}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('data_check.png', dpi=150, bbox_inches='tight')
        print("Saved data_check.png")
        print(f"Low res range: [{low.min():.3f}, {low.max():.3f}]")
        print(f"High res range: [{high.min():.3f}, {high.max():.3f}]")
        print(f"Alignment: {'ENABLED' if AUTO_ALIGN_IMAGES else 'DISABLED'}")
    except Exception as e:
        print(f"Could not save data check image: {e}")
        import traceback
        traceback.print_exc()

def train():
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    compute_device = torch.device(device_type)
    print(f"Training on: {device_type}")
    print(f"Noise settings: Corruption rate={NOISE_CORRUPTION_RATE}, Amount={NOISE_AMOUNT}")
    print(f"Alignment: {'ENABLED' if AUTO_ALIGN_IMAGES else 'DISABLED'}")

    ids = get_galaxy_ids(require_pairs=not ARTIFICIAL_DOWNSAMPLE)
    random.shuffle(ids)
    
    dataset = GalaxyDataset(ids)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True,
        num_workers=NUM_WORKERS, 
        prefetch_factor=2
    )
    
    save_data_check(dataset)
    
    model = CnnUpscaler().to(compute_device)
    loss_model = FancyLoss().to(compute_device)
    optimizer = Adam(model.parameters(), lr=0.0005)
    
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAINING_ITERATIONS, eta_min=1e-6)
    
    start = time.time()
    
    print("Starting training...")

    for iteration in range(TRAINING_ITERATIONS):
        model.train()
        total_loss = 0
        total_psnr = 0
        iteration_time = time.time()
        
        for i, (low, high) in enumerate(dataloader):
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
            
            with torch.no_grad():
                psnr = calculate_psnr(output, high_res)
                total_psnr += psnr
        
        avg_loss = total_loss / len(dataloader)
        avg_psnr = total_psnr / len(dataloader)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        end = time.time()
        avg_time = (end - start) / (iteration + 1)
        remaining = avg_time * (TRAINING_ITERATIONS - iteration - 1)

        print(f'Epoch [{iteration + 1}/{TRAINING_ITERATIONS}], '
              f'Loss: {avg_loss:.4f}, '
              f'PSNR: {avg_psnr:.2f}dB, '
              f'LR: {current_lr:.6f}, '
              f'Time: {end - iteration_time:.2f}s, '
              f'Remaining: {format_time(remaining)}')
        
        if (iteration + 1) % 5 == 0:
            torch.save({
                'epoch': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'psnr': avg_psnr,
            }, f"checkpoint_iteration_{iteration + 1}.pth")
    
    torch.save(model.state_dict(), "upscale_model.pth")

if __name__ == "__main__":
    train()