import os
import pandas
import numpy
import torch
import sklearn.tree as tree
import sklearn.model_selection as skmod
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import astropy.io.fits as fits
from upscaler import CnnUpscaler, normalize_image


CUTOUT_DIR_SMALL = "fitssmall"
CUTOUT_DIR_BIG = "fits"
FAILED_LOG = "failed_classifier.txt"

USE_UPSCALER = "none"
UPSCALER_MODEL_PATH = "upscale_model.pth"

CUTOUT_LABEL_SMALL = "DSS1 "
CUTOUT_LABEL_BIG = "SDSS"
CUTOUT_BANDS_SMALL = ["Red"]
CUTOUT_BANDS_BIG = ["r"]


def get_galaxy_ids(use_big=False):
    dir_path = CUTOUT_DIR_BIG if use_big else CUTOUT_DIR_SMALL
    if not os.path.isdir(dir_path):
        raise Exception(f"Directory: {dir_path} does not exist")
    return os.listdir(dir_path)


def get_fits(galaxy_id, use_big=False):
    dir_path = CUTOUT_DIR_BIG if use_big else CUTOUT_DIR_SMALL
    label = CUTOUT_LABEL_BIG if use_big else CUTOUT_LABEL_SMALL
    bands = CUTOUT_BANDS_BIG if use_big else CUTOUT_BANDS_SMALL
    
    galaxy_dir = os.path.join(dir_path, f"{galaxy_id}")
    band_data = []
    
    for band in bands:
        path = os.path.join(galaxy_dir, f"{label}{band}_{galaxy_id}.fits")
        
        if not os.path.exists(path) or os.path.getsize(path) < 6000:
            return None
        
        try:
            band_data.append(fits.open(path)[0].data)
        except:
            return None
    
    try:
        stacked = numpy.stack(band_data)
        stacked = numpy.moveaxis(stacked, 0, -1)
        return stacked
    except:
        return None


def upscale_with_cnn(data, model, device='cpu'):
    stats = numpy.load('normalization_stats.npy', allow_pickle=True).item()
    
    if data.ndim == 2:
        data = data[numpy.newaxis, :, :]
    elif data.ndim == 3 and data.shape[2] == 1:
        data = numpy.moveaxis(data, -1, 0)
    
    normalized = normalize_image(data, stats['low_min'], stats['low_max'])
    
    data_tensor = torch.from_numpy(normalized).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        upscaled = model(data_tensor)
    
    upscaled_np = upscaled.squeeze(0).cpu().numpy()
    
    if upscaled_np.shape[0] == 1:
        upscaled_np = numpy.moveaxis(upscaled_np, 0, -1)
    
    return upscaled_np


def get_labelled_data(ids):
    df = pandas.read_csv("datasets/galaxies_categorical.csv")
    labels = df[["objectID", "morphology", "edgeon", "spiral"]]
    return labels[labels["objectID"].astype(str).isin(ids)]


def build_dataset(labelDf, upscale_mode="none", model=None, device='cpu'):
    ids = labelDf["objectID"].tolist()
    fits_list = []
    valid_ids = []
    fails = 0

    use_big = upscale_mode == "big"
    
    print(f"Building dataset with upscale mode: {upscale_mode}")
    
    with open(FAILED_LOG, "w") as log:
        for idx, galaxy_id in enumerate(ids):
            if fails > 10000:
                print("Breaking due to excessive fails")
                break
            
            data = get_fits(galaxy_id, use_big=use_big)
            
            if data is None:
                fails += 1
                log.write(f"{galaxy_id}\n")
                continue
            
            if upscale_mode == "cnn" and model is not None:
                try:
                    data = upscale_with_cnn(data, model, device)
                except Exception as e:
                    print(f"Failed to upscale {galaxy_id}: {e}")
                    fails += 1
                    log.write(f"{galaxy_id}\n")
                    continue
            
            fits_list.append(data)
            valid_ids.append(galaxy_id)
            
            if (idx + 1) % 100 == 0:
                print(f"Loaded: {idx + 1} galaxies")

    try:
        data_array = numpy.stack(fits_list)
    except:
        array_shapes = [arr.shape for arr in fits_list]
        unique_shapes, counts = numpy.unique(array_shapes, axis=0, return_counts=True)
        print("Array shapes and their counts in fits_list:")
        for shape, count in zip(unique_shapes, counts):
            print(f"Shape: {shape}, Count: {count}")
        raise SystemExit()
    
    data_flat = data_array.reshape(data_array.shape[0], -1)
    labels = labelDf[labelDf["objectID"].isin(valid_ids)][["morphology"]].values

    return data_flat, labels


def train_and_evaluate(upscale_mode="none", model_path=UPSCALER_MODEL_PATH):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    upscaler_model = None
    if upscale_mode == "cnn":
        print(f"Loading CNN upscaler from {model_path}...")
        upscaler_model = CnnUpscaler(scale_factor=4).to(device)
        upscaler_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        upscaler_model.eval()
    
    use_big = upscale_mode == "big"
    ids = get_galaxy_ids(use_big=use_big)
    labelled = get_labelled_data(ids)
    
    data, labels = build_dataset(labelled, upscale_mode=upscale_mode, model=upscaler_model, device=device)
    
    print(f"Dataset created: {data.shape[0]} samples, {data.shape[1]} features")
    
    classifier = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        n_jobs=-1, 
        random_state=42, 
        class_weight="balanced"
    )
    
    trainx, testx, trainy, testy = skmod.train_test_split(data, labels.ravel(), random_state=42)
    
    print("Training classifier...")
    classifier.fit(trainx, trainy)
    
    print("Evaluating...")
    predictions = classifier.predict(testx)
    
    print("\n" + "="*60)
    print(f"RESULTS - Upscale Mode: {upscale_mode.upper()}")
    print("="*60)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(testy, predictions))
    
    print("\nClassification Report:")
    print(classification_report(testy, predictions))
    
    bal_acc = balanced_accuracy_score(testy, predictions)
    print(f"\nBalanced Accuracy: {bal_acc:.3f}")
    
    return bal_acc, classifier


def compare_all_modes():
    modes = ["small", "big", "cnn"]
    results = {}
    
    print("="*60)
    print("COMPARING ALL UPSCALING MODES")
    print("="*60)
    
    for mode in modes:
        print(f"\n\n{'='*60}")
        print(f"Testing mode: {mode.upper()}")
        print(f"{'='*60}\n")
        
        try:
            bal_acc, classifier = train_and_evaluate(upscale_mode=mode)
            results[mode] = bal_acc
        except Exception as e:
            print(f"Failed to evaluate mode {mode}: {e}")
            results[mode] = None
    
    print("\n\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    for mode, acc in results.items():
        if acc is not None:
            print(f"{mode.upper()}: {acc:.3f}")
        else:
            print(f"{mode.upper()}: FAILED")
    
    return results


if __name__ == "__main__":
    upscale_mode = "cnn"
    
    bal_acc, classifier = train_and_evaluate(upscale_mode=upscale_mode)
    
    # results = compare_all_modes()