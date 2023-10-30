from torch import nn
from torch.utils.data import DataLoader

import graphics
from model import NeuralNetwork
from data_loader import *
from utils import *
from graphics import *

batch_size = 64

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
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer, loss_values, accuracy_value):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    correct = 0
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
        train_loss += loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct = correct / size * 100
    accuracy_value.append(correct)
    loss_values.append(train_loss / len(dataloader))


def test(dataloader, model, loss_fn, val_loss_values, val_accuracy_value):
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
    print(f"Test Error: \n Accuracy: {(correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    val_accuracy_value.append(correct)
    val_loss_values.append(test_loss)


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
