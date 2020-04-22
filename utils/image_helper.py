import base64
import numpy as np
from PIL import Image

DEFAULT_PATH = './pictures'


def transform_matrix_to_image(matrix):
    return Image.fromarray(matrix)


def save_images(matrices):
    for idx, matrix in enumerate(matrices):
        img = transform_matrix_to_image(matrix)
        image_path = '{}/{}.jpg'.format(DEFAULT_PATH, idx)
        img.save(image_path)


def transform_json_data_to_image_matrix(ascii_base64_img_rep, dimensions):
    base64_img_rep = ascii_base64_img_rep.encode()
    buffer = base64.decodebytes(base64_img_rep)
    np_array = np.frombuffer(buffer, dtype=np.uint8)
    image_matrix = np.reshape(np_array, dimensions)
    return image_matrix


def get_images_for_cluster(images, cluster_topic):
    images_in_cluster = []
    cluster_name = cluster_topic.split("/")[1]
    for image, attributes in images:
        if cluster_name in attributes:
            images_in_cluster.append((image, attributes))

    return images_in_cluster
