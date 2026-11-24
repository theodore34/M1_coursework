import numpy as np
import pandas as pd


SEED = 0


data = np.load("data/learning_set.npz")
series = data["series"]  # (objects, 3, days): a [km], e, drag term [m^2/kg]
label = data["label"]  # index into data["class_names"]; -1 where withheld
split = data["split"]  # index into data["split_names"]: train/validation/test
cat = pd.read_csv("data/catalogue.csv")  # one row per element set
for object_id, rows in cat.groupby("object_id"):
    rows = rows.sort_values("epoch_day")
t, a = rows["epoch_day"].to_numpy(), rows["a_km"].to_numpy()
print(t, a)
