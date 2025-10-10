import pandas as pd
import os

df = pd.read_csv("gz2_hart16.csv")
df = df.rename(columns={
    "dr7objid": "objectID",
    "t01_smooth_or_features_a01_smooth_debiased": "p_smooth",
    "t01_smooth_or_features_a02_features_or_disk_debiased": "p_features",
    "t01_smooth_or_features_a03_star_or_artifact_debiased": "p_artifact",
    "t02_edgeon_a04_yes_debiased": "p_edgeon_yes",
    "t02_edgeon_a05_no_debiased": "p_edgeon_no",
    "t04_spiral_a08_spiral_debiased": "p_spiral",
    "t04_spiral_a09_no_spiral_debiased": "p_no_spiral"
})

df = df[["objectID", "ra", "dec", "total_votes", "p_smooth", "p_features", "p_artifact",
         "p_edgeon_yes", "p_edgeon_no", "p_spiral", "p_no_spiral"]]

p_columns = ["p_smooth", "p_features", "p_artifact", "p_edgeon_yes", "p_edgeon_no", "p_spiral", "p_no_spiral"]
df[p_columns] = df[p_columns].astype(float)
df["total_votes"] = df["total_votes"].astype(int)

imageIds = pd.read_csv("gz2_filename_mapping.csv")[["objectID", "imgID"]]
merged = df.merge(imageIds, on="objectID", how="left")

image_dir = "images"
valid_ids = set(
    os.path.splitext(f)[0]
    for f in os.listdir(image_dir)
    if os.path.isfile(os.path.join(image_dir, f))
)

merged["imgID"] = merged["imgID"].astype(str)
merged["imgID"] = merged["imgID"].apply(lambda x: x if x in valid_ids else "-1")

group1 = ["p_smooth", "p_features", "p_artifact"]
group2 = ["p_edgeon_yes", "p_edgeon_no"]
group3 = ["p_spiral", "p_no_spiral"]

TOLERANCE = 0.8

filtered = merged[
    (merged[group1] > TOLERANCE).any(axis=1) &
    (merged[group2] > TOLERANCE).any(axis=1) &
    (merged[group3] > TOLERANCE).any(axis=1)
]

filtered["morphology"] = filtered[group1].idxmax(axis=1).str.replace("p_", "")
filtered["edgeon"] = filtered[group2].idxmax(axis=1).str.replace("p_edgeon_", "")
filtered["spiral"] = filtered[group3].idxmax(axis=1).str.replace("p_", "").str.replace("no_spiral", "no")
filtered_combined = filtered.drop(columns=p_columns)
filtered_combined.to_csv("datasets/galaxies_combined.csv", index=False)

filtered_noImg = filtered[filtered["imgID"] == "-1"]
filtered_noImg = filtered_noImg.drop(columns=p_columns)
filtered_noImg.to_csv("datasets/galaxies_noImg.csv", index=False)

filtered_with_img = filtered[filtered["imgID"] != "-1"].sort_values(by="imgID")
filtered_with_img.to_csv("datasets/galaxies_probabilities.csv", index=False)


categorical = filtered_with_img.drop(columns=p_columns)
categorical.to_csv("datasets/galaxies.csv", index=False)


