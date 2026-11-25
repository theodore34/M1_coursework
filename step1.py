# Runtime: about 1 s on Macbook

import utils
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt

SEED = 0
rng = np.random.default_rng(SEED)
plt.rcParams.update(utils.MPL_STYLE)

results = {}

# Functions definition



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
    return df.set_index("object_id"), data["class_names"], data["split_names"], data['series']


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

def question_1_2(df, series, class_names):
    """
    Answer to question 1.2
    """

    # Selection of 3 points per class within the training set
    train_df = df[df["split"] == 0]
    points = {}
    for i in range(4):
        ids = train_df[train_df["label"] == i].index
        points[i] = rng.choice(ids, 3, replace=False) 

    # Getting the trajectories
    traj = {i: series[points[i]] for i in points}

    # Plotting


    ylabels = [r"$a$ [km]", r"$e$", r"drag term [m$^2$ kg$^{-1}$]"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, layout="constrained")

    for i, curves in traj.items():
        for k, c in enumerate(curves):
            for q in range(3):
                axes[q].plot(c[q], color=utils.colors[i], lw=1, alpha=0.85,
                             label=class_names[i] if (k == 0 and q == 0) else None)

    for q in range(3):
        axes[q].set_ylabel(ylabels[q])
    axes[-1].set_xlabel("day")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", frameon=False)
    plt.show()
    fig.savefig("figures/1.2.png", dpi=200)
    plt.close(fig)
    
    
    

def main():
    df, class_names, split_names, series = load_learning_set()
    #bal_acc = question_1_1(df, split_names)
    question_1_2(df, series, class_names)
    print(class_names)
    #print_results()


if __name__ == "__main__":
    main()
