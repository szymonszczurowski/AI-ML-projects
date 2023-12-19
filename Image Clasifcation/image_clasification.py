from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import torch
from torch import optim

mnist = fetch_openml("mnist_784", parser='auto')

data = mnist.data.to_numpy()
targets = np.vectorize(lambda x: int(x))(mnist.target.to_numpy()) # changing a string to an int

print(data.shape)
print(targets.shape)
print(targets)

# Data split
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, train_size=0.8, stratify=targets)

# Data Set
class DigitsDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx] / 255
        y = self.targets[idx]

        return x, y


train_dataset = DigitsDataset(train_data, train_targets)
test_dataset = DigitsDataset(test_data, test_targets)

# print(train_dataset[0][0])
# print(test_dataset[0][1])

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Neural Network

class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        if not self.training:
            out = F.softmax(out, dim=1)
        return out