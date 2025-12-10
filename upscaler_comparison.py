import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import astropy.io.fits as fits
import os
import random
from upscaler import CnnUpscaler, normalize_image, downsample_fits, align_images

USE_ARTIFICIAL_DOWNSAMPLING = False 

HIGH_RES_DIR = "fits"
LOW_RES_DIR = "fitssmall"

CHECKPOINT_PATH = "checkpoints/downsample_noiseAdded_50.pth"

ADD_NOISE = False
NOISE_CORRUPTION_RATE = 0.5 
NOISE_AMOUNT = 0.15  
SCALE_FACTOR = 4

USE_RANDOM_GALAXY = False  
SPECIFIC_GALAXY_ID = "587722982299402394"  
RANDOM_SEED = None  

OUTPUT_DIR = "upscaler_outputs"


def get_available_galaxies():
    if not os.path.exists(HIGH_RES_DIR):
        raise Exception(f"Directory {HIGH_RES_DIR} does not exist")
    
    galaxy_ids = os.listdir(HIGH_RES_DIR)
    
    galaxy_ids = [gid for gid in galaxy_ids if os.path.isdir(os.path.join(HIGH_RES_DIR, gid))]
    
    return galaxy_ids

def select_galaxy():
    if USE_RANDOM_GALAXY:
        available = get_available_galaxies()
        
        if len(available) == 0:
            raise Exception(f"No galaxies found in {HIGH_RES_DIR}")
        
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
        
        selected = random.choice(available)
        print(f"Randomly selected galaxy: {selected}")
        print(f"(Selected from {len(available)} available galaxies)")
        return selected
    else:
        print(f"Using specified galaxy: {SPECIFIC_GALAXY_ID}")
        return SPECIFIC_GALAXY_ID

def get_fits_local(directory, galaxy_id):
    prefix = "SDSSr" if directory == HIGH_RES_DIR else "DSS1 Red"
    
    galaxy_dir = os.path.join(directory, f"{galaxy_id}")
    filename = f"{prefix}_{galaxy_id}.fits"
    path = os.path.join(galaxy_dir, filename)
    
    if not os.path.exists(path):
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return None

    try:
        with fits.open(path) as hdul:
            data = hdul[0].data
            
        if data is None:
            return None
        
        data = data.astype(np.float32)
        
        if len(data.shape) == 2:
            data = data[np.newaxis, :, :]
        elif len(data.shape) == 3 and data.shape[0] > 4:
            data = np.moveaxis(data, -1, 0)
            
        return data
        
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def upscale_nearest(low_res, scale_factor=4):
    img = low_res[0] if low_res.ndim == 3 else low_res
    h, w = img.shape
    upscaled = cv.resize(img, (w * scale_factor, h * scale_factor), interpolation=cv.INTER_NEAREST)
    return upscaled[np.newaxis, :, :]

def upscale_bilinear(low_res, scale_factor=4):
    img = low_res[0] if low_res.ndim == 3 else low_res
    h, w = img.shape
    upscaled = cv.resize(img, (w * scale_factor, h * scale_factor), interpolation=cv.INTER_LINEAR)
    return upscaled[np.newaxis, :, :]

def upscale_bicubic(low_res, scale_factor=4):
    img = low_res[0] if low_res.ndim == 3 else low_res
    h, w = img.shape
    upscaled = cv.resize(img, (w * scale_factor, h * scale_factor), interpolation=cv.INTER_CUBIC)
    return upscaled[np.newaxis, :, :]

def upscale_lanczos(low_res, scale_factor=4):
    img = low_res[0] if low_res.ndim == 3 else low_res
    h, w = img.shape
    upscaled = cv.resize(img, (w * scale_factor, h * scale_factor), interpolation=cv.INTER_LANCZOS4)
    return upscaled[np.newaxis, :, :]

def upscale_cnn(low_res, model, device='cpu'):
    low_res_tensor = torch.from_numpy(low_res).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(low_res_tensor)
    return output.squeeze(0).cpu().numpy()


def calculate_metrics(upscaled, ground_truth):
    if upscaled.shape != ground_truth.shape:
        min_h = min(upscaled.shape[1], ground_truth.shape[1])
        min_w = min(upscaled.shape[2], ground_truth.shape[2])
        upscaled = upscaled[:, :min_h, :min_w]
        ground_truth = ground_truth[:, :min_h, :min_w]
    
    mse = np.mean((upscaled - ground_truth) ** 2)
    
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    return psnr, mse

def load_and_prepare(galaxy_id):
    high_res = get_fits_local(HIGH_RES_DIR, galaxy_id)
    if high_res is None:
        print(f"Could not load High Res for {galaxy_id}")
        return None, None
    
    high_res_norm = normalize_image(high_res)

    if USE_ARTIFICIAL_DOWNSAMPLING:
        low_res_norm = downsample_fits(
            high_res_norm, 
            downsample_factor=SCALE_FACTOR, 
            add_noise=ADD_NOISE,
            corruption_rate=NOISE_CORRUPTION_RATE,
            noise_amount=NOISE_AMOUNT
        )
    else:

        low_res = get_fits_local(LOW_RES_DIR, galaxy_id)
        if low_res is None:
            print(f"Could not load Low Res for {galaxy_id}")
            return None, None
        low_res_norm = normalize_image(low_res)
    
    high_res_norm, _ = align_images(low_res_norm, high_res_norm)
    return low_res_norm, high_res_norm

def save_individual_image(img, method_name, galaxy_id, metrics=None, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    
    img_display = img[0] if img.ndim == 3 else img
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_display, cmap='gray', vmin=0, vmax=1)
    title = method_name

    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    safe_method_name = method_name.replace(" ", "_").lower()
    filename = os.path.join(output_dir, f"{galaxy_id}_{safe_method_name}.png")
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return filename

def compare_upscalers(galaxy_id, model_path, save_fits=False, 
                     device='cpu', include_cnn=True, scale_factor=4,
                     save_individual=True):

    model = None
    if include_cnn:
        print(f"Loading CNN model from {model_path}...")
        model = CnnUpscaler(scale_factor=scale_factor).to(device)
        
        loaded_content = torch.load(model_path, map_location=device, weights_only=True)
        
        if isinstance(loaded_content, dict) and 'model_state_dict' in loaded_content:
            print("Detected checkpoint dictionary, extracting state_dict...")
            model.load_state_dict(loaded_content['model_state_dict'])
        else:
            model.load_state_dict(loaded_content)
            
        model.eval()
    
    print(f"\nLoading galaxy {galaxy_id}...")
    print(f"Noise settings: Corruption={NOISE_CORRUPTION_RATE*100:.0f}%, Amount=±{NOISE_AMOUNT}")
    
    low_res_norm, high_res_norm = load_and_prepare(galaxy_id)
    
    if low_res_norm is None:
        return None, None
    
    print(f"Low-res shape: {low_res_norm.shape}")
    print(f"High-res shape: {high_res_norm.shape}")
    print(f"Low-res range: [{low_res_norm.min():.3f}, {low_res_norm.max():.3f}]")
    print(f"High-res range: [{high_res_norm.min():.3f}, {high_res_norm.max():.3f}]")
    
    print("\nApplying upscaling methods...")
    upscalers = {
        'Low Resolution Image': low_res_norm,
        'Nearest Neighbor': upscale_nearest(low_res_norm, scale_factor),
        'Bilinear': upscale_bilinear(low_res_norm, scale_factor),
        'Bicubic': upscale_bicubic(low_res_norm, scale_factor),
        'Lanczos': upscale_lanczos(low_res_norm, scale_factor),
        'High Resolution Image': high_res_norm
    }
    
    if include_cnn and model is not None:
        upscalers['CNN Upscaler'] = upscale_cnn(low_res_norm, model, device)
    
    metrics = {}
    print("\n" + "="*60)
    print("METRICS (vs High Resolution Image)")
    print("="*60)
    
    for name, img in upscalers.items():
        if name != 'High Resolution Image':
            psnr, mse = calculate_metrics(img, high_res_norm)
            metrics[name] = {'PSNR': psnr, 'MSE': mse}
    
    if save_individual:
        print("\nSaving individual images...")
        for name, img in upscalers.items():
            method_metrics = metrics.get(name, None)
            filename = save_individual_image(img, name, galaxy_id, method_metrics)
            print(f"  Saved {filename}")
    
    print("\nCreating comparison plot...")
    
    plot_order = ['Low Resolution Image', 'Nearest Neighbor', 'Bilinear', 'Bicubic', 'Lanczos']
    if include_cnn and 'CNN Upscaler' in upscalers:
        plot_order.append('CNN Upscaler')
    plot_order.append('High Resolution Image')
    
    num_images = len(plot_order)
    nrows = 2
    ncols = 3 if num_images <= 6 else 4
    fig = plt.figure(figsize=(18, 10) if num_images <= 6 else (20, 12))
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, name in enumerate(plot_order):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[row, col])
        
        img = upscalers[name]
        img_display = img[0] if img.ndim == 3 else img
        
        ax.imshow(img_display, cmap='gray', vmin=0, vmax=1)
        
        if name in metrics:
            title = name
        else:
            title = name
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    output_filename = f'upscaler_comparison_{galaxy_id}.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_filename}")
    
    if save_fits:
        print("Saving FITS files...")
        for name, img in upscalers.items():
            if name not in ['Low Resolution Image', 'High Resolution Image']:
                img_array = img[0] if img.ndim == 3 else img
                hdu = fits.PrimaryHDU(img_array)
                fits_filename = f'upscaled_{name.replace(" ", "_").lower()}_{galaxy_id}.fits'
                fits.HDUList([hdu]).writeto(fits_filename, overwrite=True)
                print(f"  Saved {fits_filename}")
                pass
    
    plt.show()
    return upscalers, metrics

def batch_compare(galaxy_ids, model_path, device='cpu', save_individual=True):
    all_results = {}
    
    for galaxy_id in galaxy_ids:
        print("\n" + "="*70)
        print(f"Processing Galaxy: {galaxy_id}")
        print("="*70)
        
        try:
            upscalers, metrics = compare_upscalers(
                galaxy_id, 
                model_path=model_path,
                save_fits=False,
                device=device,
                include_cnn=True,
                scale_factor=SCALE_FACTOR,
                save_individual=save_individual
            )
            
            if metrics is not None:
                all_results[galaxy_id] = metrics
                
        except Exception as e:
            print(f"Error processing {galaxy_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("SUMMARY - Average Metrics Across All Galaxies")
    print("="*70)
    
    if all_results:
        method_names = list(next(iter(all_results.values())).keys())
        
        for method in method_names:
            psnrs = [res[method]['PSNR'] for res in all_results.values() if method in res]
            mses = [res[method]['MSE'] for res in all_results.values() if method in res]
            
            avg_psnr = np.mean(psnrs)
            avg_mse = np.mean(mses)
            std_psnr = np.std(psnrs)
            
            print(f"{method}:")
            print(f"  PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
            print(f"  MSE:  {avg_mse:.6f}")
    
    return all_results

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Scale factor: {SCALE_FACTOR}x")
    print(f"Artificial Downsampling: {USE_ARTIFICIAL_DOWNSAMPLING}")
    print(f"Noise: Corruption={NOISE_CORRUPTION_RATE*100:.0f}%, Amount=±{NOISE_AMOUNT}")
    print(f"Individual images will be saved to: {OUTPUT_DIR}/")
    print(f"Random selection: {USE_RANDOM_GALAXY}")
    
    galaxy_id = select_galaxy()
    
    upscalers, metrics = compare_upscalers(
        galaxy_id, 
        model_path=CHECKPOINT_PATH,
        save_fits=False,
        device=device,
        include_cnn=True,
        scale_factor=SCALE_FACTOR,
        save_individual=False
    )
    
    # Example: Batch comparison on multiple galaxies
    # galaxy_ids = ["587722982299402394", "587722982292717776", "587722984429060253"]
    # results = batch_compare(galaxy_ids, CHECKPOINT_PATH, device, save_individual=True)
    
    # Example: Random batch comparison
    # available = get_available_galaxies()
    # random_galaxies = random.sample(available, min(5, len(available)))
    # results = batch_compare(random_galaxies, CHECKPOINT_PATH, device, save_individual=True)