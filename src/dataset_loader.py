import torch 
from torchvision import datasets
from torch.utils.data import DataLoader

from preprocessing import train_transform, test_transform

train_dir = "../data/train"
val_dir = "../data/val"
test_dir = "../data/test"


train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=test_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

#data loaders
batch_size = 30

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)