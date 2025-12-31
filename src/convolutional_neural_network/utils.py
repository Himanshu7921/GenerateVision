import os
import pickle
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "cifar-10-batches-py")

def load_batch(file):
    with open(file, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    X = torch.tensor(batch[b"data"], dtype=torch.float32)
    y = torch.tensor(batch[b"labels"], dtype=torch.long)

    X = (X.view(-1, 3, 32, 32) / 127.5) - 1.0
    return X, y


def get_data():
    X_train, y_train = [], []

    for i in range(1, 6):
        path = os.path.join(DATA_DIR, f"data_batch_{i}")
        X, y = load_batch(path)
        X_train.append(X)
        y_train.append(y)

    X_train = torch.cat(X_train)
    y_train = torch.cat(y_train)

    X_test, y_test = load_batch(os.path.join(DATA_DIR, "test_batch"))

    return X_train, y_train, X_test, y_test

    
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()
    print(f"X_train.shape = {tuple(X_train.shape)} | (N, C, H, W)")
    print(f"X_test.shape = {tuple(X_test.shape)} | (N, C, H, W)")
    print(f"x_train min value = {X_train.min().item():.2f} | x_train max value = {X_train.max().item():.2f}")
    print(f"x_train mean = {X_train.mean().item():.4f} | x_train std = {X_train.std().item():.4f}")