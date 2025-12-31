from model import CNNModel
from utils import load_batch, get_data
import torch


def train_model(model: CNNModel, epochs: int, loss_fn: torch.nn.CrossEntropyLoss, optimizer: torch.optim, batch_size: int, device: torch.device):
    X_train, y_train, X_test, y_test = get_data()
    X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)
    N = X_train.shape[0]

    for epoch in range(epochs):
        # Shuffeling the batch
        perm = torch.randperm(N)
        X_train = X_train[perm]
        y_train = y_train[perm]

        for b in range(0, N, batch_size):
            X_train_batch, y_train_batch = X_train[b: b + batch_size, :, :, :], y_train[b: b + batch_size]

            logits = model(X_train_batch)
            loss = loss_fn(logits, y_train_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 1 == 0:
            print(f"Epoch = {epoch}/{epochs} | train loss = {loss.item():.3f}")
            model.eval()
            with torch.no_grad():
                test_logits = model(X_test)
                test_loss = loss_fn(test_logits, y_test)
                print()
            model.train()
            print(f"Epoch = {epoch}/{epochs} | test loss = {test_loss.item():.3f}")
        
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(n_classes = 10)
    print(f"device = {device}")
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters())
    batch_size = 32
    train_epochs = 20
    trained_model = train_model(model = model, epochs = train_epochs, loss_fn = loss_fn, optimizer = optimizer, batch_size = batch_size, device = device)

    print("Training Completed.....")

    print("Time to check accuracy..")
    _, _, X_test, y_test = get_data()
    X_test, y_test = X_test.to(device), y_test.to(device)
    X_test, y_test = X_test[:30], y_test[:30]
    logits = trained_model(X_test)
    probs = torch.softmax(logits, dim = -1)
    predictions = torch.argmax(probs, dim = -1)
    print("Model's Prediction: ", predictions)
    print("Actual Label: ", y_test)
    acc = (predictions == y_test).float().mean() * 100
    print(f"Model's Accuracy: {acc:.2f}%")


