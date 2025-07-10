from pathlib import Path

import numpy as np
import pandas as pd


class Dockstring:
    """
    DOCKSTRING dataset loader.

    Args:
        - target: DOCKSTRING target
        - n_train: Number of train molecules in train / test split
        - seed: Random seed for sampling training set

    Note that the test set is constant as determined by cluster_split.tsv
    """

    def __init__(self, target: str = "PARP1", n_train: int = 10000, seed: int = 42):
        self.target = target
        self.n_train = n_train
        self.seed = seed

    def load(self):
        current_file_dir = Path(__file__).parent

        # Build relative paths
        dataset_path = current_file_dir / "data" / "dockstring" / "dockstring-dataset.tsv"
        dataset_split_path = current_file_dir / "data" / "dockstring" / "cluster_split.tsv"

        assert dataset_path.exists()
        assert dataset_split_path.exists()

        df = pd.read_csv(dataset_path, sep="\t")

        splits = pd.read_csv(dataset_split_path, sep="\t").loc[df.index]

        # Create train and test datasets
        df_train = df[splits["split"] == "train"]
        df_test = df[splits["split"] == "test"]

        if self.n_train < len(df_train):
            df_train = df_train.sample(n=self.n_train, random_state=self.seed)

        smiles_train = df_train["smiles"].values
        smiles_test = df_test["smiles"].values

        y_train = np.minimum(df_train[self.target].values, 5.0)
        y_test = np.minimum(df_test[self.target].values, 5.0)

        smiles_train = smiles_train[~np.isnan(y_train)]
        y_train_nonan = y_train[~np.isnan(y_train)]

        smiles_test = smiles_test[~np.isnan(y_test)]
        y_test_nonan = y_test[~np.isnan(y_test)]

        return smiles_train, smiles_test, y_train_nonan, y_test_nonan


class MoleculeNet:
    pass
