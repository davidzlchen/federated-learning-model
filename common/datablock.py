from utils.image_helper import transform_json_data_to_image_matrix
from utils.image_helper import get_images_for_cluster
import random


class Datablock(object):
    def __init__(
        self,
        images=[],
        labels=[]
    ):
        self.num_images = len(list(images))
        self.current_image = self.num_images - 1
        self.image_data = list(images)
        self.dimensions = []
        self.labels = list(labels)

    def __getitem__(self, key):
        if isinstance(key, int):
            temp_datablock = Datablock(images=[], labels=[])
            temp_datablock.init_new_image(
                self.dimensions[key], self.labels[key])
            temp_datablock.image_data[-1] = self.image_data[key]
            return temp_datablock

        elif isinstance(key, slice):
            temp_datablock = Datablock(images=[], labels=[])
            for i in range(key.start, key.stop):
                temp_datablock.init_new_image(
                    self.dimensions[i], self.labels[i])
                temp_datablock.image_data[-1] = self.image_data[i]
            return temp_datablock
        else:
            raise TypeError(
                'Index must be int, not {}'.format(
                    type(key).__name__))

    def init_new_image(self, dimensions, label):
        self.num_images += 1
        self.current_image += 1
        self.image_data.append("")
        self.dimensions.append(dimensions)
        self.labels.append(label)

    def add_image_chunk(self, image_chunk):
        i_idx = self.current_image
        self.image_data[i_idx] += image_chunk

    def convert_current_image_to_matrix(self):
        i_idx = self.current_image
        image_dimensions = self.dimensions[i_idx]
        image_ascii_rep = self.image_data[i_idx]
        image_matrix_rep = transform_json_data_to_image_matrix(
            image_ascii_rep, image_dimensions)
        self.image_data[i_idx] = image_matrix_rep

    def shuffle_data(self):
        zipped = list(zip(self.image_data, self.dimensions, self.labels))
        random.shuffle(zipped)

        self.image_data, self.dimensions, self.labels = zip(*zipped)

    def reset(self):
        self.num_images = 0
        self.current_image = -1
        self.image_data = []
        self.dimensions = []
        self.labels = []

    # Used to add images for a cluster or cluster partition from preprocessed .pkl data
    def add_images_for_cluster(self, images, cluster_topic, partition=False, partition_index=0, num_partitions=1):
        images_in_cluster = get_images_for_cluster(images, cluster_topic)
        if(partition):
            images_per_partition = len(images_in_cluster) / num_partitions
            start = partition_index * images_per_partition
            end = partition_index * images_per_partition + images_per_partition
            images_in_cluster = images_in_cluster[start:end]
            print("# of Images in Cluster Partition: ", len(images_in_cluster))
        else:
            print("# of Images in Cluster: ", len(images_in_cluster))

        for image, attributes in images_in_cluster:
            label = 0
            if "person" in attributes:
                label = 1
            elif "no-person" in attributes:
                label = 0
            else:
                print("SOMETHING IS BROKEN WITH DATA LABELING")
            self.init_new_image(image.shape, label)
            self.image_data[-1] = image

        self.shuffle_data()
