import torchvision
import torchvision.transforms as transforms

from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader

from models import PersonBinaryClassifier, ModelRunner
from datasets import CocoDataset, collate_fn

BATCH_SIZE_TRAIN = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
STEP_SIZE = 7
GAMMA = 0.1
EPOCHS = 2
LOG_INTERVAL = 10

def load_data(datablocks):
    images = []
    labels = []
    
    for client in datablocks:
        images.extend(datablocks[client].image_data)
        labels.extend(datablocks[client].labels)

    people_dataset = CocoDataset(
        images,
        transform=torchvision.transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        labels=labels
    )

    print(len(people_dataset))

    people_data_loader = DataLoader(
        people_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        collate_fn=collate_fn
    )

    return people_data_loader

def initialize_runner(dataloader, epochs):
    model = PersonBinaryClassifier()
    criterion = CrossEntropyLoss()
    optimizer_ft = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA)
    dataloaders = {'val': dataloader, 'train': dataloader}

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

def get_model_runner(client_data, num_epochs=1):
    people_data_loader = load_data(client_data)
    runner = initialize_runner(people_data_loader, num_epochs)
    return runner
