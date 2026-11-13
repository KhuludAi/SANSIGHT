
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent

df = pd.read_csv(ROOT / "oct_train_clean.csv")

def patient_id(path):
    name = Path(path).stem
    match = re.match(
        r"^(CNV|DME|DRUSEN|NORMAL)-(.+)-\d+$",
        name
    )
    if not match:
        return None
    return match.group(2)

df["patient"] = df["path"].apply(patient_id)

if df["patient"].isna().any():
    print("Filename pattern needs checking!")
    print(df[df["patient"].isna()]["path"].head(10))
    raise SystemExit

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(df["path"], df["label"], df["patient"])
)

df["split"] = "train"
df.loc[val_idx, "split"] = "validation"

df.to_csv(
    ROOT / "oct_train_val_split.csv",
    index=False
)

print("\nSplit results:")
print(pd.crosstab(df["split"], df["label"]))

print("\nUnique patients:")
print(df.groupby("split")["patient"].nunique())

train_patients = set(df[df.split == "train"].patient)
val_patients = set(df[df.split == "validation"].patient)

print("\nPatient overlap:", len(train_patients & val_patients))
print("Total images:", len(df))
print("Saved: oct_train_val_split.csv")
