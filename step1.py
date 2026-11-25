# Runtime: about 1 s on Macbook

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

SEED = 0
results = {}

# Functions definition


def print_results():
    """
    Print every results cleanly
    """
    for key, value in results.items():
        if isinstance(value, (float, np.floating)):
            value = f"{value:.3g}"
        print(f"row {key}: {value}")


def load_learning_set(path="data/learning_set.npz"):
    """
    Load the learning set and return a pandas dataframe
    """
    data = np.load(path)
    df = pd.DataFrame(
        {
            "object_id": np.arange(len(data["label"])),
            "label": data["label"],
            "split": data["split"],
        }
    )
    return df.set_index("object_id"), data["class_names"], data["split_names"]


def question_1_1(df, split_names):
    """
    Answer the question 1.1: counting data + dummy classifier
    """

    # Counting the train/test/valid
    split_counts = df["split"].value_counts().sort_index()
    split_counts.index = split_names[split_counts.index]

    # Majority class baseline
    train_df = df[df["split"] == 0]
    valid_df = df[df["split"] == 1]
    class_counts = train_df["label"].value_counts(normalize=True)

    maj_class = class_counts.idxmax()
    y_pred = np.full(len(valid_df), maj_class)
    acc = accuracy_score(valid_df["label"], y_pred)
    bal_acc = balanced_accuracy_score(valid_df["label"], y_pred)

    # Adding the results to the dictionnary for the printing
    results["1.1a"] = split_counts["train"]
    results["1.1b"] = split_counts["validation"]
    results["1.1c"] = split_counts["test"]
    results["1.1d"] = acc
    results["1.1e"] = bal_acc

    return bal_acc

def question_1_2(df):
    """
    Answer to question 1.2
    """
    print("Welcome to question 2")

def main():
    df, class_names, split_names = load_learning_set()
    #bal_acc = question_1_1(df, split_names)
    question_1_2(df)

    #print_results()


if __name__ == "__main__":
    main()
