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

def load_data(raw_data):
    images = []
    labels = []

    for client in raw_data:
        images.extend(raw_data[client].image_data)
        labels.extend(raw_data[client].labels)

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

    people_data_loader = DataLoader(
        people_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        collate_fn=collate_fn
    )

    return people_data_loader

def train(client_data):
    people_data_loader = load_data(client_data)
    model = PersonBinaryClassifier()
    criterion = CrossEntropyLoss()
    optimizer_ft = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA)
    num_epochs = EPOCHS

    runner = ModelRunner(
        model=model,
        criterion=criterion,
        dataloaders=people_data_loader,
        epochs=num_epochs,
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler
    )
    return runner.train_model()
    #model.save('./network.pth')
