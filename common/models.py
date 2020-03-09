import torch
import torch.nn as nn
import torchvision.models as models

from copy import deepcopy


class PersonBinaryClassifier(nn.Module):
    def __init__(self):
        super(PersonBinaryClassifier, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Linear(2048, 2)
        self.model = resnet

    def forward(self, x):
        return self.model.forward(x)

    def save(self, path):
        torch.save(self.model.fc.state_dict(), path)
        print("Successfully saved model to: {}".format(path))

    def get_state_dictionary(self):
        return self.model.fc.state_dict()

    def load_last_layer_state_dictionary(self, state_dict):
        self.model.fc.load_state_dict(state_dict)
        print('Successfully loaded last layer state dictionary.')


class ModelRunner(object):
    def __init__(
            self,
            model,
            criterion,
            dataloaders,
            epochs,
            optimizer,
            scheduler,
    ):
        self.model = model
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_model(self):
        best_model_wts = deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch, self.epochs - 1))
            print("-" * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                dataloader = self.dataloaders[phase]
                dataset_size = len(dataloader.dataset)
                for inputs, labels in dataloader:
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model.forward(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects.double() / dataset_size

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = deepcopy(self.model.state_dict())

        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model

    def test_model(self):
        self.model.eval()

        print("Epoch 1/1")
        print("-" * 10)

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        dataloader = self.dataloaders['val']
        dataset_size = len(dataloader.dataset)
        for inputs, labels in dataloader:
            # forward
            # track history if only in train
            outputs = self.model.forward(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print('Test Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))
