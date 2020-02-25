from utils.image_helper import transform_json_data_to_image_matrix


class Datablock(object):
    def __init__(
        self,
        images=[],
        labels=[]
    ):
        self.num_images = len(images)
        self.current_image = self.num_images - 1
        self.image_data = images
        self.dimensions = []
        self.labels = labels

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
