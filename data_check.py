
import matplotlib.pyplot as plt
import random

from upscaler import get_fits, get_galaxy_ids, normalize_image

CUTOUT_DIR_SMALL = "fitssmall"
CUTOUT_DIR_BIG = "fits2"


def save_data_check(num_pairs=5):
    try:
        ids = get_galaxy_ids(require_pairs=True)
        
        if len(ids) < num_pairs:
            print(f"Warning: Only found {len(ids)} paired galaxies, showing all available.")
            num_pairs = len(ids)
        
        random.shuffle(ids)
        selected_ids = ids[:num_pairs]
        
        fig, axes = plt.subplots(num_pairs, 2, figsize=(8, 4 * num_pairs))
        
        for i, galaxy_id in enumerate(selected_ids):
            low_res = get_fits(CUTOUT_DIR_SMALL, galaxy_id)
            high_res = get_fits(CUTOUT_DIR_BIG, galaxy_id)
            
            if low_res is None or high_res is None:
                print(f"Skipping {galaxy_id} - missing data")
                continue
            
            low_res = normalize_image(low_res[0])
            high_res = normalize_image(high_res[0])
            
            axes[i, 0].imshow(low_res, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(f'Low Res (DSS1) - ID: {galaxy_id}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(high_res, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(f'High Res (SDSS) - ID: {galaxy_id}')
            axes[i, 1].axis('off')
            
            print(f"Pair {i+1}: {galaxy_id}")
            print(f"  Low res shape: {low_res.shape}, range: [{low_res.min():.3f}, {low_res.max():.3f}]")
            print(f"  High res shape: {high_res.shape}, range: [{high_res.min():.3f}, {high_res.max():.3f}]")
        
        plt.tight_layout()
        plt.savefig('data_check.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved data_check.png showing {num_pairs} galaxy pairs")
        print("Check if the galaxy orientations match between low-res and high-res images.")
        
    except Exception as e:
        print(f"Could not save data check image: {e}")

if __name__ == "__main__":
    save_data_check(5)