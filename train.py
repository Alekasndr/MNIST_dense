from torch import nn
from torch.utils.data import DataLoader

import graphics
from model import NeuralNetwork
from data_loader import *
from utils import *
from graphics import *

batch_size = 128

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


def loss_and_accuracy_check(dataloader, model, loss_fn, loss, accuracy):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct = correct / size * 100
    print(f"Accuracy: {(correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    loss.append(correct)
    accuracy.append(test_loss)


def train(dataloader, model, loss_fn, optimizer, loss_values, accuracy_value):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(X)

        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    indices = torch.randperm(len(training_data))[:10000]
    subset = torch.utils.data.Subset(training_data, indices)
    print(f"Train:")
    loss_and_accuracy_check(DataLoader(subset, batch_size=batch_size), model, loss_fn, loss_values, accuracy_value)


def test(dataloader, model, loss_fn, val_loss_values, val_accuracy_value):
    print(f"Test:")
    loss_and_accuracy_check(dataloader, model, loss_fn, val_loss_values, val_accuracy_value)


if __name__ == '__main__':
    epochs = 100
    loss_values = []
    val_loss_values = []
    accuracy_value = []
    val_accuracy_value = []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, loss_values, accuracy_value)
        test(test_dataloader, model, loss_fn, val_loss_values, val_accuracy_value)
    print("Done!")

    graphics.loss_graphic_creation(epochs, loss_values, val_loss_values)
    graphics.accuracy_graphic_creation(epochs, accuracy_value, val_accuracy_value)

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
