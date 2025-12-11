import numpy as np
import torch

def evaluate_model(model, test_df, user_items, n_items, top_k=10):
    model.eval()

    hits = 0
    ndcg = 0
    total = 0

    for _, row in test_df.iterrows():
        user = int(row["user_id"])
        true_item = int(row["item_id"])

        # Pilih 99 negative samples (items yang belum pernah di-interact user)
        candidates = list(set(range(n_items)) - user_items[user])
        candidates = np.random.choice(candidates, 99, replace=False).tolist()

        # Gabungkan dengan 1 positive item
        items = [true_item] + candidates

        # Input ke model
        users = torch.tensor([user] * len(items), dtype=torch.long)
        items_tensor = torch.tensor(items, dtype=torch.long)

        scores = model(users, items_tensor)
        ranked = torch.argsort(scores, descending=True)

        # Posisi item sebenarnya dalam ranking
        rank = ranked.tolist().index(0)

        # Hit@10
        if rank < top_k:
            hits += 1
            ndcg += 1 / np.log2(rank + 2)

        total += 1

    return hits / total, ndcg / total
