import base64
import json

from utils import constants 
from enum import Enum

DEFAULT_PACKET_SIZE = 3000

class MessageType(Enum):
    NETWORK_CHUNK = 1
    IMAGE_CHUNK = 2
    SIMPLE = 3 #use when short enough to not need chunks

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True

def divide_chunks(array, size=DEFAULT_PACKET_SIZE):
    len_array = len(array)
    for idx in range(0, len_array, size):
        yield array[idx:idx+size]

def send_message(client, topic, message):
    if not is_json(message):
        message = json.dumps(message)
    client.publish(topic, message)

def send_network_chunk_message(client, topic, network_data):
    network_message_chunk = constants.DEFAULT_NETWORK_MESSAGE_CHUNK
    network_data_encoded = base64.b64encode(network_data)
    network_data_encoded = network_data_encoded.decode('utf-8')

    client.publish(topic, json.dumps(constants.DEFAULT_NETWORK_INIT_MESSAGE))
    for chunk in divide_chunks(network_data_encoded):
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

def send_typed_message(client, topic, message, message_type):
    if message_type is MessageType.NETWORK_CHUNK:
        send_network_chunk_message(client, topic, message)
    elif message_type is MessageType.IMAGE_CHUNK:
        send_image_chunk_message(client, topic, message)
    elif message_type is MessageType.SIMPLE:
        send_message(client,topic,message)
    else:
        print('{} not handled'.format(message_type))