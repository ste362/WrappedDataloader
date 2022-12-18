import random

import torch
import torchvision
import torchvision.transforms as transforms
from time import perf_counter
import torch.nn as nn
import torch.nn.functional as F
from torch import dtype

from optimize import WrappedDataLoader

torch.cuda.get_device_name(0)

cuda = torch.cuda.is_available()
torch.cuda.init()
transform = transforms.ToTensor()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

train_dataset = torchvision.datasets.FashionMNIST(root='.', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='.', train=False, transform=transform, download=True)


BATCH_SIZE=32
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,pin_memory=True)

###############################################################################

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(device), y.to(device)

train_dataloader = WrappedDataLoader(train_dataloader, preprocess, shuffle=True)
test_dataloader = WrappedDataLoader(test_dataloader, preprocess, shuffle=True)

#################################################################################


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input Image -28 * 28 * 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        # Size= (no. of out_channel-kernel_size+ 2*Padding/Stride) +1
        # Size= (32-5+ 2*0/1 ) +1 = 24 * 24 * 32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Size= 12 * 12 * 32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(in_features=1 * 1 * 64, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Size =4 * 4 * 32
        x = self.pool(F.relu(self.conv3(x)))
        # Size =1 * 1 * 64
        x = x.view(-1, 64 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
if cuda:
    net = net.cuda()

print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


NUM_EPOCH=10
NUM_BATCH=len(train_dataset)//(BATCH_SIZE)

t1_start = perf_counter()
for epoch in range(NUM_EPOCH):
    running_loss = 0.0
    epoch_start=perf_counter()
    for i, data in enumerate(train_dataloader, 0):
        images, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        outputs = net(images)
        # calculate the loss between the target and the actuals
        loss = criterion(outputs, labels)
        # Gradient calculation using backward pass
        loss.backward()
        # update the weights
        optimizer.step()
        # calculate loss
        running_loss += loss.item()
        if i % NUM_BATCH == NUM_BATCH-1:
            epoch_time=perf_counter()-epoch_start
            print('Epoch: %2d/%d - Batch count: %5d/%d - Loss: %.3f - Time: %.2fs' % (epoch + 1, NUM_EPOCH, i + 1, NUM_BATCH, running_loss/NUM_BATCH, epoch_time))
            running_loss = 0.0
t1_end = perf_counter()
print("Time for training using PyTorch %.2f seconds" % (t1_end - t1_start))

torch.save(net.state_dict(), 'mnist_pyt.pt')

# evaluate on test dataset
correct = 0
total = 0
t1_start = perf_counter()
with torch.no_grad():
    '''Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward(). 
    It will reduce memory consumption for computations that would otherwise have requires_grad=True.'''
    for data in test_dataloader:
        images,labels = data
        outputs = net(images)
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    t1_end = perf_counter()
    print("Eval accuracy using PyTorch is %.2f and execution time %.2f seconds" % ((100 * (correct / total)), (t1_end - t1_start)))
