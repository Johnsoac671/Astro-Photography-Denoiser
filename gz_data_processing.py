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


group1 = ["p_smooth", "p_features", "p_artifact"]
group2 = ["p_edgeon_yes", "p_edgeon_no"]
group3 = ["p_spiral", "p_no_spiral"]

TOLERANCE = 0.8

filtered = df[
    (df[group1] > TOLERANCE).any(axis=1) &
    (df[group2] > TOLERANCE).any(axis=1) &
    (df[group3] > TOLERANCE).any(axis=1)
]
categorical = filtered.copy()
categorical["morphology"] = filtered[group1].idxmax(axis=1).str.replace("p_", "")
categorical["edgeon"] = filtered[group2].idxmax(axis=1).str.replace("p_edgeon_", "")
categorical["spiral"] = filtered[group3].idxmax(axis=1).str.replace("p_", "").str.replace("no_spiral", "no")

categorical = categorical.drop(columns=p_columns)
categorical.to_csv("datasets/galaxies_categorical.csv", index=False)

filtered.to_csv("datasets/galaxies_probabilities.csv", index=False)



