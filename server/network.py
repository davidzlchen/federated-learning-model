import numpy as np
import itertools
import pickle
import time
import copy

from PIL import Image

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate



BATCH_SIZE_TRAIN = 10
LEARNING_RATE = 0.01
EPOCHS = 5
LOG_INTERVAL = 10

person_dataset = []
no_person_dataset = []
people_dataset = []


person_train_loader = []
no_person_train_loader = []
people_data_loader = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = models.resnet50(pretrained=True)

num_samples = 0


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

        try:
            if self.transform:
                sample = (self.transform(Image.fromarray(sample)), person)
            else:
                sample = (Image.fromarray(sample), person)
        except:
            sample = None

        return sample


def collate_fn(batch):
    # print(batch)
    batch = list(filter(lambda x : x is not None, batch))
    return default_collate(batch)


def loadData(raw_data):
    
    # no_person_pkl = open('../client/files/COCO/nopersonimages.pkl', 'rb')
    # no_person_matrix = pickle.load(no_person_pkl)

    # person_pkl = open('../client/files/COCO/personimages.pkl', 'rb')
    # person_matrix = pickle.load(person_pkl)

    global num_samples
    global people_dataset
    global people_data_loader


    images = []
    labels = []

    for client in raw_data:
        images.extend(raw_data[client]["imageData"])
        labels.extend(raw_data[client]["labels"])

    num_samples = len(images)
    

    people_dataset = CocoDataset(
        images,
        transform=torchvision.transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        person=labels
    )

    batch_size_train = 60
    people_data_loader = DataLoader(
        people_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

def train_model(model, criterion, optimizer, scheduler, dataloader, dataset_size, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def run(clientData):

    global network
    global people_data_loader


    loadData(clientData)

    for param in network.parameters():
        param.requires_grad = False
    network.fc = nn.Linear(2048, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    network = train_model(network, criterion, optimizer_ft, exp_lr_scheduler, people_data_loader, num_samples, num_epochs=2)

    # for epoch in range(EPOCHS):
    #     train("cpu", epoch)


    torch.save(network.state_dict(), "./network.pth")
    



