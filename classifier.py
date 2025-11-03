import os
import pandas
import numpy
import sklearn.tree as tree
import sklearn.model_selection as skmod
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import astropy.io.fits as fits

CUTOUT_DIR = "fitssmall"
FAILED_LOG = "failed.txt"

# CUTOUT_LABEL = "SDSS"
# CUTOUT_BANDS = ["r"]

CUTOUT_LABEL = "DSS1 "
CUTOUT_BANDS = ["Red"]

def get_galaxy_ids():
    if not os.path.isdir(CUTOUT_DIR):
        raise Exception(f"Directory: {CUTOUT_DIR} does not exist")
    return os.listdir(CUTOUT_DIR)

def get_fits(galaxy_id):
    galaxy_dir = os.path.join(CUTOUT_DIR, f"{galaxy_id}")
    
    bands = []
    
    for band in CUTOUT_BANDS:
        path = os.path.join(galaxy_dir, f"{CUTOUT_LABEL}{band}_{galaxy_id}.fits")
        
        if not os.path.exists(path) or os.path.getsize(path) < 1028:
            return None

        bands.append(fits.open(path)[0].data)
        
    

    try:
        stacked = numpy.stack(bands)
        stacked = numpy.moveaxis(stacked, 0, -1)
    except:
        return None

    return stacked

def get_labelled_data(ids):
    df = pandas.read_csv("datasets\\galaxies_categorical.csv")
    labels = df[["objectID", "morphology", "edgeon", "spiral"]]
    
    return labels[labels["objectID"].astype(str).isin(ids)]

def build_dataset(labelDf):
    ids = labelDf["objectID"].tolist()
    fits_list = []
    valid_ids = []

    index = 0
    fails = 0

    with open(FAILED_LOG, "w") as log:
        for galaxy_id in ids:
            
            if fails > 10000:
                print("breaking due to fails")
                break
            
            data = get_fits(galaxy_id)
            if data is None:
                fails += 1
                log.write(f"{galaxy_id}\n")
                continue

            fits_list.append(data)
            valid_ids.append(galaxy_id)
            index += 1
            
            if index % 100 == 0:
                print(f"Done with: {index}")

    try:
        data = numpy.stack(fits_list)
    except:
        array_shapes = [arr.shape for arr in fits_list]
        unique_shapes, counts = numpy.unique(array_shapes, axis=0, return_counts=True)
        print("Array shapes and their counts in fits_list:")
        for shape, count in zip(unique_shapes, counts):
            print(f"Shape: {shape}, Count: {count}")
        SystemExit()
    
    data = data.reshape(data.shape[0], -1) # flatten data
    labels = labels = labelDf[labelDf["objectID"].isin(valid_ids)][["edgeon"]].values

    return data, labels

data, labels = build_dataset(get_labelled_data(get_galaxy_ids()))

print("dataset made")
model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42, class_weight="balanced")

trainx, testx, trainy, testy = skmod.train_test_split(data, labels.ravel())

print("dataset split")
model = model.fit(trainx, trainy)
print("model trained")

predictions = model.predict(testx)

print("Confusion Matrix:")
print(confusion_matrix(testy, predictions))

print("\nClassification Report:")
print(classification_report(testy, predictions))

print(f"\nBalanced Accuracy: {balanced_accuracy_score(testy, predictions):.3f}")