import os
import pandas
import numpy
import torch
import cv2 as cv
import sklearn.model_selection as skmod
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import astropy.io.fits as fits

from upscaler import CnnUpscaler, normalize_image, downsample_fits


HIGH_RES_DIR = "fits2"
LOW_RES_DIR = "fitssmall"
CSV_PATH = "datasets/galaxies_categorical.csv"
FAILED_LOG = "failed_classifier.txt"

UPSCALER_CHECKPOINT = "checkpoints/downsample_noiseAdded_50.pth"
SCALE_FACTOR = 4

ARTIFICIAL_DOWNSAMPLE = False 
ADD_NOISE = False
NOISE_CORRUPTION_RATE = 0.5
NOISE_AMOUNT = 0.4

SIZE_SMALL = 64
SIZE_BIG = 256

BALANCE_CLASSES = True
RANDOM_SEED = 42
FEATURE_LABEL = "edgeon"  # "morphology", "edgeon"
DATA_MODE = "cnn"  # Options: "small", "big", "cnn", "nearest", "bilinear", "bicubic", "lanczos"
REQUIRE_PAIRED_IMAGES = False 


def get_galaxy_ids(use_high_res_dir=True):
    target_dir = HIGH_RES_DIR if use_high_res_dir else LOW_RES_DIR
    
    if not os.path.isdir(target_dir):
        raise Exception(f"Directory: {target_dir} does not exist")
    
    if REQUIRE_PAIRED_IMAGES:
        if not os.path.isdir(HIGH_RES_DIR):
            raise Exception(f"Directory: {HIGH_RES_DIR} does not exist")
        if not os.path.isdir(LOW_RES_DIR):
            raise Exception(f"Directory: {LOW_RES_DIR} does not exist")
        
        high_res_ids = set(os.listdir(HIGH_RES_DIR))
        low_res_ids = set(os.listdir(LOW_RES_DIR))
        
        paired_ids = list(high_res_ids.intersection(low_res_ids))
        print(f"Found {len(paired_ids)} paired galaxies (exist in both directories)")
        print(f"  High-res only: {len(high_res_ids - low_res_ids)}")
        print(f"  Low-res only: {len(low_res_ids - high_res_ids)}")
        
        return paired_ids
    else:
        all_ids = os.listdir(target_dir)
        print(f"Found {len(all_ids)} galaxies in {target_dir}")
        return all_ids

def get_fits(base_dir, galaxy_id):
    galaxy_dir = os.path.join(base_dir, f"{galaxy_id}")
    
    candidates = [
        f"SDSSr_{galaxy_id}.fits",
        f"DSS1 Red_{galaxy_id}.fits",
        f"{galaxy_id}.fits"
    ]
    
    path = None
    for fname in candidates:
        p = os.path.join(galaxy_dir, fname)
        if os.path.exists(p):
            path = p
            break
            
    if path is None:
        return None

    try:
        with fits.open(path) as hdul:
            data = hdul[0].data
            data = data.astype(numpy.float32)

            if len(data.shape) == 2:
                data = data[numpy.newaxis, :, :]
            elif len(data.shape) == 3 and data.shape[0] > 4:
                data = numpy.moveaxis(data, -1, 0)
            return data
    except:
        return None

def resize_to_fixed(img, target_size):
    c, h, w = img.shape
    if h == target_size and w == target_size:
        return img
    
    img_cv = numpy.moveaxis(img, 0, -1)
    
    interp = cv.INTER_AREA if h > target_size else cv.INTER_CUBIC
    img_resized = cv.resize(img_cv, (target_size, target_size), interpolation=interp)
    
    if len(img_resized.shape) == 2:
        img_resized = img_resized[:, :, numpy.newaxis]
        
    return numpy.moveaxis(img_resized, -1, 0)

def upscale_with_interpolation(data, scale_factor, method='bicubic'):
    img = data[0] if data.ndim == 3 else data
    h, w = img.shape
    
    interp_methods = {
        'nearest': cv.INTER_NEAREST,
        'bilinear': cv.INTER_LINEAR,
        'bicubic': cv.INTER_CUBIC,
        'lanczos': cv.INTER_LANCZOS4
    }
    
    if method not in interp_methods:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    interp_flag = interp_methods[method]
    upscaled = cv.resize(img, (w * scale_factor, h * scale_factor), interpolation=interp_flag)
    
    return upscaled[numpy.newaxis, :, :]

def upscale_with_cnn(data, model, device='cpu'):
    data_tensor = torch.from_numpy(data).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        upscaled = model(data_tensor)
    
    upscaled_np = upscaled.squeeze(0).cpu().numpy()
    
    return upscaled_np

def get_labelled_data(ids, balance_classes=True):

    df = pandas.read_csv(CSV_PATH)
    
    df = df[df["objectID"].astype(str).isin(ids)]
    

    if FEATURE_LABEL == "morphology" and "morphology" in df.columns:
        df = df[df["morphology"] != "artifact"]

    if balance_classes:
        class_counts = df[FEATURE_LABEL].value_counts()
        print(f"\nOriginal class distribution for '{FEATURE_LABEL}':")
        for class_val, count in class_counts.items():
            print(f"  {class_val}: {count}")
        
        min_samples = class_counts.min()
        print(f"\nBalancing classes to {min_samples} samples each.")
        
        balanced_dfs = []
        for class_val in class_counts.index:
            class_df = df[df[FEATURE_LABEL] == class_val]
            balanced_class = class_df.sample(n=min_samples, random_state=RANDOM_SEED)
            balanced_dfs.append(balanced_class)
        
        df = pandas.concat(balanced_dfs, ignore_index=True)
        df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        print(f"\nBalanced class distribution:")
        for class_val, count in df[FEATURE_LABEL].value_counts().items():
            print(f"  {class_val}: {count}")
        
    return df

def build_dataset(labelDf, mode="small", model=None, device='cpu'):
    ids_to_load = labelDf["objectID"].tolist()
    
    loaded_data = []
    fails = 0
    
    upscale_modes = ['cnn', 'nearest', 'bilinear', 'bicubic', 'lanczos']
    is_upscale_mode = mode in upscale_modes
    
    print(f"\nBuilding dataset | Mode: {mode.upper()} | Artificial Downsample: {ARTIFICIAL_DOWNSAMPLE}")
    if is_upscale_mode:
        print(f"Upscaling method: {mode}")
    print(f"Noise settings: Corruption={NOISE_CORRUPTION_RATE}, Amount={NOISE_AMOUNT}")
    
    with open(FAILED_LOG, "w") as log:
        for idx, galaxy_id in enumerate(ids_to_load):
            if fails > 10000:
                print("Too many load failures, stopping.")
                break
            
            try:
                img = None
                
                need_high_res_source = (mode == "big") or ARTIFICIAL_DOWNSAMPLE or is_upscale_mode
                
                if need_high_res_source:
                    img = get_fits(HIGH_RES_DIR, galaxy_id)
                    
                    if img is not None:
                        img = normalize_image(img)
                        
                        if mode == "small" or is_upscale_mode:
                            img = downsample_fits(img, SCALE_FACTOR, ADD_NOISE, 
                                                NOISE_CORRUPTION_RATE, NOISE_AMOUNT)
                else:
                    img = get_fits(LOW_RES_DIR, galaxy_id)
                    if img is not None:
                        img = normalize_image(img)

                if img is None:
                    raise FileNotFoundError
                
                if is_upscale_mode:
                    if mode == "cnn":
                        if model is None:
                            raise ValueError("CNN model required for 'cnn' mode")
                        img = upscale_with_cnn(img, model, device)
                    else:
                        img = upscale_with_interpolation(img, SCALE_FACTOR, method=mode)

                target_dim = SIZE_SMALL if mode == "small" else SIZE_BIG
                img = resize_to_fixed(img, target_dim)

                flat_data = img.flatten()
                loaded_data.append({'objectID': galaxy_id, 'data': flat_data})
            
            except Exception as e:
                fails += 1
                log.write(f"{galaxy_id}: {e}\n")
                continue
            
            if (idx + 1) % 100 == 0:
                print(f"Processed: {idx + 1}/{len(ids_to_load)}")

    print("\nMerging data with labels...")
    data_df = pandas.DataFrame(loaded_data)
    
    labelDf['objectID'] = labelDf['objectID'].astype(str)
    data_df['objectID'] = data_df['objectID'].astype(str)
    
    merged_df = pandas.merge(labelDf, data_df, on='objectID', how='inner')
    
    print(f"Successfully loaded and merged {len(merged_df)} samples.")
    print(f"Failed to load: {fails} samples")
    
    X = numpy.stack(merged_df['data'].values)
    y = merged_df[FEATURE_LABEL].values

    print(f"\nData range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Data mean: {X.mean():.4f}, std: {X.std():.4f}")

    return X, y

def train_and_evaluate(mode="small"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    upscaler_model = None
    if mode == "cnn":
        print(f"Loading CNN upscaler from {UPSCALER_CHECKPOINT}...")
        upscaler_model = CnnUpscaler(scale_factor=SCALE_FACTOR).to(device)
        
        chk = torch.load(UPSCALER_CHECKPOINT, map_location=device, weights_only=True)
        if isinstance(chk, dict) and 'model_state_dict' in chk:
            upscaler_model.load_state_dict(chk['model_state_dict'])
        else:
            upscaler_model.load_state_dict(chk)
        upscaler_model.eval()
    
    upscale_modes = ['cnn', 'nearest', 'bilinear', 'bicubic', 'lanczos']
    use_high_dir_for_ids = (mode == "big") or ARTIFICIAL_DOWNSAMPLE or (mode in upscale_modes)
    ids = get_galaxy_ids(use_high_res_dir=use_high_dir_for_ids)
    
    labelled_df = get_labelled_data(ids, balance_classes=BALANCE_CLASSES)
    
    data, labels = build_dataset(labelled_df, mode=mode, model=upscaler_model, device=device)
    
    print(f"\nDataset ready: {data.shape[0]} samples, {data.shape[1]} features")
    
    print("\nTraining Random Forest...")
    classifier = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        n_jobs=-1, 
        random_state=RANDOM_SEED
    )
    
    trainx, testx, trainy, testy = skmod.train_test_split(data, labels, test_size=0.2, random_state=RANDOM_SEED)
    
    classifier.fit(trainx, trainy)
    predictions = classifier.predict(testx)
    
    print("\n" + "="*40)
    print(f"RESULTS - Mode: {mode.upper()}")
    print(f"Feature: {FEATURE_LABEL}")
    print(f"Noise: {NOISE_CORRUPTION_RATE*100:.0f}% corruption, ±{NOISE_AMOUNT} magnitude")
    print("="*40)
    print(confusion_matrix(testy, predictions))
    print(classification_report(testy, predictions))
    
    bal_acc = balanced_accuracy_score(testy, predictions)
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    
    return bal_acc

if __name__ == "__main__":
    train_and_evaluate(mode=DATA_MODE)