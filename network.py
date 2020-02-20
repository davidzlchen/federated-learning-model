from PIL import Image

from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from models import PersonBinaryClassifier, ModelRunner

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

        image = self.matrix[idx]
        person = self.person[idx]

        try:
            if self.transform:
                sample = (self.transform(Image.fromarray(image)), person)
            else:
                sample = (Image.fromarray(image), person)
        except BaseException:
            sample = None

        return sample


def collate_fn(batch):
    # print(batch)
    batch = list(filter(lambda x: x is not None, batch))
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


def run(clientData):
    global network
    global people_data_loader

    loadData(clientData)

    model = PersonBinaryClassifier()
    criterion = CrossEntropyLoss()
    optimizer_ft = SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)
    num_epochs = 2

    runner = ModelRunner(
        model=model,
        criterion=criterion,
        dataloaders=people_data_loader,
        epochs=num_epochs,
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler
    )
    runner.train_model()

    model.save('./network.pth')
