import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


def load_dataset(csv_path):
    """
    Load interactions.csv dan convert user_id & item_id ke index numerik.
    Digunakan untuk implicit feedback (NCF).
    """
    df = pd.read_csv(csv_path)

    # Encode ID ke index 0..N (WAJIB untuk embedding)
    df["user_id"] = df["user_id"].astype("category").cat.codes
    df["item_id"] = df["item_id"].astype("category").cat.codes

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()

    # Simpan semua item yang pernah diinteraksi oleh setiap user
    user_items = df.groupby("user_id")["item_id"].apply(set).to_dict()

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df, n_users, n_items, user_items


class InteractionDataset(Dataset):
    """
    Dataset untuk implicit feedback.
    Semua interaksi diberi label = 1
    Negative sampling dilakukan di train.py
    """
    def __init__(self, df):
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_id"].values, dtype=torch.long)

        # Label implicit: semua interaksi positif = 1
        self.labels = torch.ones(len(df), dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]
