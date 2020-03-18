import pickle
import numpy as np
from torchvision import transforms

from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader

from common.models import PersonBinaryClassifier, ModelRunner
from common.datasets import CocoDataset, collate_fn
from common.datablock import Datablock

BATCH_SIZE_TRAIN = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
STEP_SIZE = 7
GAMMA = 0.1
EPOCHS = 5
LOG_INTERVAL = 10


def get_test_data():
    person_test_samples = pickle.load(
        open('./data/personimagesTest.pkl', 'rb'))
    no_person_test_samples = pickle.load(
        open('./data/nopersonimagesTest.pkl', 'rb'))

    person_test_images = [sample[0] for sample in person_test_samples[:25]]
    no_person_test_images = [sample[0]
                             for sample in no_person_test_samples[:25]]

    num_test_samples = len(person_test_images)
    images = np.concatenate((person_test_images, no_person_test_images))
    labels = np.concatenate((
        np.ones(num_test_samples, dtype=np.int_),
        np.zeros(num_test_samples, dtype=np.int_)
    ))

    datablocks = {'1': Datablock(images=images, labels=labels)}
    return datablocks


def load_data(datablocks):
    images = []
    labels = []

    for client in datablocks:
        images.extend(datablocks[client].image_data)
        labels.extend(datablocks[client].labels)

    people_dataset = CocoDataset(
        images,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        labels=labels
    )

    people_data_loader = DataLoader(
        people_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        collate_fn=collate_fn
    )

    return people_data_loader


def initialize_runner(train_dataloader, val_dataloader, epochs):
    model = PersonBinaryClassifier()
    criterion = CrossEntropyLoss()
    optimizer_ft = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    runner = ModelRunner(
        model=model,
        criterion=criterion,
        dataloaders=dataloaders,
        epochs=epochs,
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler
    )
    return runner


def train(client_data):
    runner = get_model_runner(client_data, 2)
    return runner.train_model()


def test(client_data):
    runner = get_model_runner(client_data)
    return runner.test_model()


def get_model_runner(client_data=get_test_data(), num_epochs=EPOCHS):
    people_data_loader = load_data(client_data)
    test_data_loader = load_data(get_test_data())
    runner = initialize_runner(
        people_data_loader,
        test_data_loader,
        num_epochs)
    return runner
