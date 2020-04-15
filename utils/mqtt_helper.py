import base64
import json
import pickle

from common.configuration import ConfigurationEncoder
from utils import constants
from enum import Enum
from common.ResultData import ResultDataEncoder

DEFAULT_PACKET_SIZE = 3000


class MessageType(Enum):
    NETWORK_CHUNK = 1
    IMAGE_CHUNK = 2
    SIMPLE = 3  # use when short enough to not need chunks
    CONFIGURATION = 4
    RESULT_DATA = 5


def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except BaseException:
        return False
    return True


def divide_chunks(array, size=DEFAULT_PACKET_SIZE):
    len_array = len(array)
    for idx in range(0, len_array, size):
        yield array[idx:idx + size]


def send_message(client, topic, message, encoder=None):
    if not is_json(message):
        if encoder is None:
            message = json.dumps(message)
        else:
            message = json.dumps(message, cls=encoder)
    client.publish(topic, message)


def send_network_chunk_message(client, topic, network_data):
    network_message_chunk = constants.DEFAULT_NETWORK_MESSAGE_CHUNK

    client.publish(topic, json.dumps(constants.DEFAULT_NETWORK_INIT_MESSAGE))
    for chunk in divide_chunks(network_data):
        network_message_chunk['data'] = chunk
        client.publish(topic, json.dumps(network_message_chunk))
    client.publish(topic, json.dumps(constants.DEFAULT_NETWORK_END_MESSAGE))


def send_image_chunk_message(client, topic, sample):
    image, label = sample

    image_init_message = constants.DEFAULT_IMAGE_INIT_MESSAGE
    image_init_message['label'] = label
    image_init_message['dimensions'] = image.shape
    client.publish(topic, json.dumps(image_init_message))

    image_message_chunk = constants.DEFAULT_IMAGE_MESSAGE_CHUNK
    image_encoded = base64.b64encode(image).decode('utf-8')
    for chunk in divide_chunks(image_encoded):
        image_message_chunk['data'] = chunk
        client.publish(topic, json.dumps(image_message_chunk))
    client.publish(topic, json.dumps(constants.DEFAULT_IMAGE_END_MESSAGE))

def send_result_data_message(client, topic, result_data_instance):
    message = constants.RESULT_DATA_MESSAGE
    message['data'] = result_data_instance
    send_message(client, topic, message, encoder=ResultDataEncoder)

def send_configuration_message(client, topic, configuration_instance):
    #serialized_configuration = pickle.dumps(configuration_instance)
    message = constants.CONFIGURATION_MESSAGE
    message['data'] = configuration_instance
    send_message(client, topic, message, encoder=ConfigurationEncoder)


def send_typed_message(client, topic, message, message_type):
    if message_type is MessageType.NETWORK_CHUNK:
        send_network_chunk_message(client, topic, message)
    elif message_type is MessageType.IMAGE_CHUNK:
        send_image_chunk_message(client, topic, message)
    elif message_type is MessageType.CONFIGURATION:
        send_configuration_message(client, topic, message)
    elif message_type is MessageType.SIMPLE:
        send_message(client, topic, message)
    elif message_type is MessageType.RESULT_DATA:
        send_result_data_message(client, topic, message)
    else:
        print('{} not handled'.format(message_type))
