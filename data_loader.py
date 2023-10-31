import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

random_indices = torch.randperm(len(training_data))
train_indices = random_indices[:50000]
valid_indices = random_indices[50000:]

train_subset = torch.utils.data.Subset(training_data, train_indices)
valid_subset = torch.utils.data.Subset(training_data, valid_indices)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
