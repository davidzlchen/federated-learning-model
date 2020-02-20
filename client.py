import time
import paho.mqtt.client as mqtt
import pickle
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import itertools
import base64
import random
import string
import math
import copy
import os
import torchvision.models as models
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


network_str = ""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_sizes = {}


#########################################
# model stuff
#########################################

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
                sample = (self.transform(Image.fromarray(sample[0])), person)
            else:
                sample = (Image.fromarray(sample[0]), person)
        except BaseException:
            sample = None

        return sample


# def load_model(network, filepath=""):
#     checkpoint = torch.load(('../server/network.pth'))
#     network.load_state_dict(checkpoint)

def reconstruct_model(network):
    # global network_str
    # network_bytes = network_str.encode()
    # network_decoded = base64.decodebytes(network_bytes)
    # checkpoint = torch.load(BytesIO(network_decoded))
    checkpoint = torch.load(('../server/network.pth'))
    network.fc.load_state_dict(checkpoint)
    network_str = ""


def create_test_loader():
    global dataset_sizes
    person_test_pkl = open('./files/COCO/personimagesTest.pkl', 'rb')
    person_test_matrices = pickle.load(person_test_pkl)
    no_person_test_pkl = open('./files/COCO/nopersonimagesTest.pkl', 'rb')
    no_person_test_matrices = pickle.load(no_person_test_pkl)

    num_test_samples = len(person_test_matrices)
    dataset_sizes = {'val': num_test_samples * 2}
    test_dataset = CocoDataset(
        person_test_matrices + no_person_test_matrices,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        person=np.concatenate((np.ones(num_test_samples, dtype=np.int_), np.zeros(num_test_samples, dtype=np.int_)))
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )
    return test_loader


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def test_model(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs=25):
    global device
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ['val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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


########################################
# sending image stuff
########################################


def convertImageToBase64(to_encode):
    encoded = base64.b64encode(to_encode)
    return encoded


def randomword(length):
    return ''.join(random.choice(string.ascii_lowercase)
                   for i in range(length))


def publishEncodedImage(image, label):
    packet_size = 3000
    encoded = base64.b64encode(image)

    end = packet_size
    start = 0
    length = len(encoded)
    no_of_packets = math.ceil(length / packet_size)

    client.publish("client/pi01",
                   json.dumps({"message": "sending_data",
                               "dimensions": image.shape,
                               "label": label}))

    while start <= len(encoded):
        data = {"message": "chunk", "data": encoded[start:end].decode('utf-8')}
        data_packet = json.dumps(data)

        client.publish("client/pi01", data_packet)

        end += packet_size
        start += packet_size

    client.publish("client/pi01", json.dumps({"message": "done"}))


def send_images():
    # person_pkl = open('./files/COCO/personimages.pkl', 'rb')
    persons_data = pickle.load(open('./files/COCO/personimages.pkl', 'rb'))
    no_persons_data = pickle.load(
        open('./files/COCO/nopersonimages.pkl', 'rb'))
    person_images = []
    no_person_images = []

    for image in persons_data:
        person_images.append(image[0])

    for image in no_persons_data:
        no_person_images.append(image[0])

    Image.fromarray(no_person_images[0])
    end = int(120 / 15)
    for j in range(0, end):
        time.sleep(5)
        for i in range(15 * j, 15 * j + 15):
            ''' must send in batches bc broker can't handle 240 images sent at once'''
            publishEncodedImage(person_images[i], 1)
            publishEncodedImage(no_person_images[i], 0)

#########################################
# mqtt stuff
#########################################


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("server/network")
    send_images()
    print("publishing images done")

    # person_pkl = open('./files/COCO/personimages.pkl', 'rb')
    # person_matrix = pickle.load(person_pkl)
    # person_matrix = json.dumps(person_matrix[:1])
    # client.publish("data", "hello world")
    # client.publish("data", person_matrix)
    # client.publish("data", "hello world")

# The callback for when a PUBLISH message is received from the server.


def on_message(client, userdata, msg):
    print("on message")
    print(msg.topic)
    global network_str
    global dataset_sizes
    #print(msg.topic+" "+str(msg.payload))
    payload = json.loads(msg.payload.decode())
    if(payload["message"] == "sending_data"):
        pass
    elif(payload["message"] == "network_chunk"):
        network_str += payload["data"]
    elif(payload["message"] == "end_transmission"):
        test_loader = create_test_loader()
        dataloaders = {'val': test_loader}
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Linear(2048, 2)
        # resnet.load_state_dict()

        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=7, gamma=0.1)
        reconstruct_model(resnet)
        test_model(
            resnet,
            criterion,
            optimizer_ft,
            exp_lr_scheduler,
            dataloaders,
            dataset_sizes,
            num_epochs=1)


def on_publish(client, userdata, result):
    print("data published")


client = mqtt.Client()
#client.on_publish = on_publish
client.on_connect = on_connect
client.on_message = on_message

client.connect("localhost", 1883, 65534)

# # Blocking call that processes network traffic, dispatches callbacks and
# # handles reconnecting.
# # Other loop*() functions are available that give a threaded interface and a
# # manual interface.

# test_loader = create_test_loader()
# dataloaders = {'val': test_loader}
# resnet = models.resnet50(pretrained=True)
# for param in resnet.parameters():
#     param.requires_grad = False
# resnet.fc = nn.Linear(2048, 2)
# # resnet.load_state_dict()
#
# criterion = nn.CrossEntropyLoss()
# optimizer_ft = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# reconstruct_model(resnet)
# test_model(resnet, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=1)
client.loop_forever()
