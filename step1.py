# Importing Librairies

import numpy as np
import pandas as pd

# Hyperameters
SEED = 0

# Functions definition

def load_learning_set(path="data/learning_set.npz"):
    """
    Load the learning set and return a pandas dataframe
    """
    data = np.load(path)
    df = pd.DataFrame({
        "object_id": np.arange(len(data["label"])),
        "label": data["label"],    
        "split": data["split"],
    })
    return df.set_index("object_id"), data["class_names"], data["split_names"]

df, class_names, split_names = load_learning_set()

split_counts = df["split"].value_counts().sort_index()
split_counts.index = split_names[split_counts.index]

print(split_counts)
