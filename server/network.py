import torch
import torch.nn as nn
import torch.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # Net takes 1 chanel images (black and white), produces 10 output convolutions, with 5x5
        self.conv1 = nn.Conv2D(1, 10, 5)
        self.conv2 = nn.Conv2D(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def initializeNetwork():
    learning_rate = 0.01
    epochs = 3
    log_interval = 10


    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate)

