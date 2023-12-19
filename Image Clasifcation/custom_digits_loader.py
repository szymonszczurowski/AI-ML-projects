from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import torch
from torch import optim

# Data downloading
data = load_digits().data
print(f'data shape {data.shape}')

targets = load_digits().target
print(f'targets shape {targets.shape}')

# Data split
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, train_size=0.8, stratify=targets)
print(f'train_data shape {train_data.shape}')
print(f'test_data shape {test_data.shape}')

# Dataset class
class DigitsDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx] / 16
        y = self.targets[idx]

        return x, y


# Dataset initialzation
train_dataset = DigitsDataset(train_data, train_targets)
test_dataset = DigitsDataset(test_data, test_targets)

print(f'test_dataset[0] {test_dataset[0]}')

# Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

single_batch = next(iter(test_dataloader))
print(single_batch[0].shape)


# layers
fc = nn.Linear(10, 20) # in_features - input size, out_features - output size
print(f'layers{fc}')

# wage
print(fc.weight) # random weight
print(f'fc.weight.shape {fc.weight.shape}')

# activation functions
print(F.softmax(Tensor([1, 2, 3]), dim=0))


# SIMPLE NEURAL NETWORK IN PyTorch
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        out = self.fc1(x) # x -> first layer
        out = F.relu(out) # activation function for first laye
        out = self.fc2(out) # outout first layer -> second layer

        if not self.training:
            out = F.softmax(out, dim=1) # if the network is not in training mode the output of the second layer goes through softmax activation functions
        return out



model = Model(64, 10)

# loss function
ce_loss = nn.CrossEntropyLoss() # clasification
mse_loss = nn.MSELoss() # reggresion

# error functions return scalars
print(ce_loss(Tensor([[0.1,0.1,0.8]]), Tensor([[0,0,1]])))
print(mse_loss(Tensor([[0.1,0.1,0.8]]), Tensor([[0,0,1]])))

# gradinet counting with loss function ( loss.backward() )
x = torch.zeros((1,3), dtype=torch.float32, requires_grad=True)
y = torch.ones((1,3), dtype= torch.float32, requires_grad=True)

loss = ce_loss(x, y)
print(f'loss{loss}')

loss.backward()
print(x.grad)

# Optimisers
print(model.parameters())
adam = optim.Adam(model.parameters()) # aktualization parameters neural network

print(f'weights layer first on start {model.fc1.weight}')

y = model(torch.rand(100, 64))
loss = ce_loss(y, torch.ones(100, 10)) # counting loss function
loss.backward() # back propagation
print(f'weights layer first on start {model.fc1.weight.grad}') # weights gradient

adam.step()
print(f'weights layer first after first step {model.fc1.weight}') # weights gradient

# Gradients cleaning

print(f'gradient value {model.fc1.weight.grad}')
adam.zero_grad() # clening gradients
print(f'gradient value after cleaning{model.fc1.weight.grad}')