import pandas as pd

from vitals.processing.spo2 import classify_spo2

df = pd.read_csv("data/raw/Oxygen_Saturation_in_Body.csv", header=None, names=["spo2", "label"])
df["rule"] = df["spo2"].apply(lambda v: classify_spo2(v)["level"])

match = (df["rule"] == df["label"]).mean() * 100
print(f"Rule matches dataset labels: {match:.2f}%")
print(classify_spo2(97), classify_spo2(88))