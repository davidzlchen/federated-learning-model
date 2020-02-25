import torch

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from utils.image_helper import transform_matrix_to_image


class CocoDataset(Dataset):
    """ Our custom Coco Dataset """

    def __init__(self, image_matrices, labels, transform=None):
        self.image_matrices = image_matrices
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_matrices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_matrix = self.image_matrices[idx]
        image = transform_matrix_to_image(image_matrix)
        label = self.labels[idx]

        try:
            if self.transform:
                image = self.transform(image)
            sample = (image, label)
        except BaseException:
            sample = None

        return sample


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)
