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
import random, string
import math


#########################################
#### model stuff
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


def reconstruct_model(network, filepath=""):
    checkpoint = torch.load(('../server/network.pth'))
    network.load_state_dict(checkpoint)


def create_people_dataset():
    person_pkl = open('./files/COCO/personimages.pkl', 'rb')
    person_matrix = pickle.load(person_pkl)
    no_person_pkl = open('./files/COCO/nopersonimages.pkl', 'rb')
    no_person_matrix = pickle.load(no_person_pkl)
    people_dataset = CocoDataset(person_matrix + no_person_matrix,
                                 transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize((28, 28)),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307, ), (0.3081,))
                                 ]),
                                 person=np.concatenate((np.ones(120, dtype=np.int_), np.zeros(120, dtype=np.int_)))
                                )
    return people_dataset
def create_train_DataLoader(batch_size_train, people_dataset):
    people_train_loader = torch.utils.data.DataLoader(
        people_dataset,
        batch_size=batch_size_train,
        shuffle=True
    )
    return people_train_loader

def create_test_DataLoader(batch_size_test, people_dataset):
    people_test_loader = torch.utils.data.DataLoader(
        people_dataset,
        batch_size=batch_size_test,
        shuffle=True
    )
    return people_test_loader


def train(epoch, people_train_loader, network, optimizer, device):
    network.train()  # set network to training mode

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


def test(test_losses, people_test_loader, network, device):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in people_test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        num_samples = len(people_test_loader.dataset)
        test_loss /= num_samples
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, num_samples,
                100. * correct / num_samples))


def get_train_test_times(test_losses, people_train_loader, people_test_loader, network, optimizer):
    start_time = time.time()

    test(test_losses,network,"cpu")
    for epoch in range(epochs):
        train(epoch, people_train_loader,network,optimizer,"cpu")
        test(test_losses,people_test_loader,network,"cpu")

    print("CPU took %s seconds" % (time.time() - start_time))

########################################
###### sending image stuff
########################################


def convertImageToBase64(to_encode):
     encoded = base64.b64encode(to_encode)
     return encoded


def randomword(length):
     return ''.join(random.choice(string.ascii_lowercase) for i in range(length))


def publishEncodedImage(client, topic, encoded):
     packet_size = 3000
     end = packet_size
     start = 0
     length = len(encoded)
     picId = randomword(8)
     pos = 0
     no_of_packets = math.ceil(length/packet_size)

     while start <= len(encoded):
         data = {"message": "chunk", "data": encoded[start:end].decode('utf-8'), "pic_id":picId, "pos": pos, "size": no_of_packets}
         thing = json.dumps(data)
         client.publish(topic, thing)
         end += packet_size
         start += packet_size
         pos = pos +1
     client.publish(topic, json.dumps({"message": "done"}))


#########################################
#### mqtt stuff
#########################################


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("client-to-server")

    # person_pkl = open('./files/COCO/personimages.pkl', 'rb')
    # person_matrix = pickle.load(person_pkl)
    # person_matrix = json.dumps(person_matrix[:1])
    # client.publish("data", "hello world")
    # client.publish("data", person_matrix)
    # client.publish("data", "hello world")
    
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))
    if(ms.payload=="sending_model"):
        pass
    else:
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        batch_size_train = 60
        batch_size_test = 120
        test_losses = []
        learning_rate = 0.01
        epochs = 100
        log_interval = 10

        network = Net().to("cpu")
        optimizer = optim.SGD(network.parameters(), lr=learning_rate)

        people_dataset = create_people_dataset()
        #people_train_loader = create_train_DataLoader(batch_size_train, people_dataset)
        people_test_loader = create_test_DataLoader(batch_size_test, people_dataset)
        # train(epoch, people_train_loader,network,optimizer,"cpu")
        # test(test_losses,people_test_loader, network,"cpu")
        #get_train_test_times(test_losses, people_train_loader, people_test_loader, network, optimizer)
        reconstruct_model(network)
        test(test_losses, people_test_loader, network, "cpu")

def on_publish(client, userdata, result):
    print("data published")


client = mqtt.Client()
client.on_publish = on_publish
client.on_connect = on_connect
client.on_message = on_message

client.connect("localhost", 1883, 65534)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
with open('./files/COCO/personimages.pkl', 'rb') as f:
    image_list = pickle.load(f)
for (image, label) in image_list:
    #print(image)
    topic = "client/pi01"
    encoded = convertImageToBase64(image)
    client.publish(topic, json.dumps({"message": "sending_data", "dimensions": image.shape}))
    publishEncodedImage(client,topic,encoded)
client.loop_forever()






