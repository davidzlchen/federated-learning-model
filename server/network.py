import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import itertools
import pickle
from PIL import Image



BATCH_SIZE_TRAIN = 10
LEARNING_RATE = 0.01
EPOCHS = 5
LOG_INTERVAL = 10

person_dataset = []
no_person_dataset = []
people_dataset = []


person_train_loader = []
no_person_train_loader = []
people_train_loader = []

class CocoDataset(Dataset):
    """ Our custom Coco Dataset """
    
    def __init__(self, matrix, person, transform=None):
        self.matrix = matrix
        self.transform = transform
        self.person = person
        
    def __len__(self):
        return len(self.matrix)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.matrix[idx]
        person = self.person[idx]

        if self.transform:
            sample = (self.transform(Image.fromarray(sample[0]).convert('LA'))[0].unsqueeze(0), person)
        else:
            sample = (Image.fromarray(sample[0]).convert('LA')[0].unsqueeze(0), person)

        return sample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
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



network = Net().to("cpu")



def loadData():
    no_person_pkl = open('../client/files/COCO/nopersonimages.pkl', 'rb')
    no_person_matrix = pickle.load(no_person_pkl)

    person_pkl = open('../client/files/COCO/personimages.pkl', 'rb')
    person_matrix = pickle.load(person_pkl)


    global person_dataset
    global no_person_dataset
    global people_dataset
    
    global person_train_loader
    global no_person_train_loader
    global people_train_loader

    person_dataset = CocoDataset(person_matrix,
                             transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((28, 28)),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307, ), (0.3081,))
                             ]),
                             person=np.ones(120))
                             
    no_person_dataset = CocoDataset(no_person_matrix,
                             transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((28, 28)),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307, ), (0.3081,))
                             ]),
                             person=np.zeros(120))



    people_dataset = CocoDataset(person_matrix + no_person_matrix,
                             transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((28, 28)),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307, ), (0.3081,))
                             ]),
                             person=np.concatenate((np.ones(120, dtype=np.int_), np.zeros(120, dtype=np.int_)))
                            )

    person_train_loader = torch.utils.data.DataLoader(
        person_dataset,
        batch_size=BATCH_SIZE_TRAIN, 
        shuffle=True
    )

    no_person_train_loader = torch.utils.data.DataLoader(
        no_person_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True
    )

    batch_size_train = 60
    people_train_loader = torch.utils.data.DataLoader(
        people_dataset,
        batch_size=batch_size_train,
        shuffle=True
    )

def train(device, epoch):
    global network

    optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE)

    network.train() #set network to training mode

        
    batch_idx = -1
    for (data, target) in people_train_loader:
        batch_idx += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(people_train_loader.dataset),
            100. * batch_idx / len(people_train_loader), loss.item()
        ))

def run():
    loadData()
    for epoch in range(EPOCHS):
        train("cpu", epoch)


    torch.save(network.state_dict(), "./network.pth")
    



