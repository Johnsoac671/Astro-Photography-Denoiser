import os
import pandas
import numpy
import sklearn.tree as tree
import sklearn.model_selection as skmod
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import astropy.io.fits as fits

CUTOUT_DIR = "fits"
FAILED_LOG = "failed.txt"

def get_galaxy_ids():
    if not os.path.isdir(CUTOUT_DIR):
        raise Exception(f"Directory: {CUTOUT_DIR} does not exist")
    return os.listdir(CUTOUT_DIR)

def get_fits(galaxy_id):
    galaxy_dir = os.path.join(CUTOUT_DIR, f"{galaxy_id}")

    g_path = os.path.join(galaxy_dir, f"SDSSg_{galaxy_id}.fits")
    r_path = os.path.join(galaxy_dir, f"SDSSr_{galaxy_id}.fits")
    i_path = os.path.join(galaxy_dir, f"SDSSi_{galaxy_id}.fits")

    if not (os.path.exists(g_path) and os.path.exists(r_path) and os.path.exists(i_path)):
        return None

    if os.path.getsize(g_path) < 6000 or os.path.getsize(r_path) < 6000 or os.path.getsize(i_path) < 6000:
        return None

    g_fit = fits.open(g_path)[0].data
    r_fit = fits.open(r_path)[0].data
    i_fit = fits.open(i_path)[0].data

    try:
        stacked = numpy.stack([g_fit, r_fit, i_fit])
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
            
            if fails > 1000:
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

    data = numpy.stack(fits_list)
    data = data.reshape(data.shape[0], -1) # flatten data
    labels = labels = labelDf[labelDf["objectID"].isin(valid_ids)][["morphology"]].values

    return data, labels

data, labels = build_dataset(get_labelled_data(get_galaxy_ids()))

print("dataset made")
model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)

trainx, testx, trainy, testy = skmod.train_test_split(data, labels)

print("dataset split")
model = model.fit(trainx, trainy)
print("model trained")

predictions = model.predict(testx)

print("Confusion Matrix:")
print(confusion_matrix(testy, predictions))

print("\nClassification Report:")
print(classification_report(testy, predictions))

print(f"\nBalanced Accuracy: {balanced_accuracy_score(testy, predictions):.3f}")