from dataset import load_dataset, InteractionDataset
from model import NeuMF
from train import train_ncf
from evaluate import evaluate_model
from torch.utils.data import DataLoader

CSV_PATH = "interactions.csv"

print("Loading dataset...")
train_df, test_df, n_users, n_items, user_items = load_dataset(CSV_PATH)

train_loader = DataLoader(
    InteractionDataset(train_df),
    batch_size=256,
    shuffle=True
)

print("Building model...")
model = NeuMF(n_users, n_items)

print("Training...")
model = train_ncf(model, train_loader, epochs=3, lr=0.001)

print("Evaluating...")
HR, NDCG = evaluate_model(model, test_df, user_items, n_items)

print("\n=== Final Results ===")
print(f"Hit Ratio@10: {HR:.4f}")
print(f"NDCG@10     : {NDCG:.4f}")
