import base64
import json

from enum import Enum

DEFAULT_NETWORK_INIT_MESSAGE = {
    'message': 'sending_data'
}
DEFAULT_NETWORK_MESSAGE_CHUNK = {
    'message': 'network_chunk'
}
DEFAULT_NETWORK_END_MESSAGE = {
    'message': 'end_transmission'
}
DEFAULT_PACKET_SIZE = 3000

class MessageType(Enum):
    NETWORK_CHUNK = 1

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True

def divide_chunks(array, size):
    len_array = len(array)
    for idx in range(0, len_array, size):
        yield array[idx:idx+size]

def send_message(client, topic, message):
    if not is_json(message):
        message = json.dumps(message)
    client.publish(topic, message)

def send_network_chunk_message(client, topic, network_data):
    network_message_chunk = DEFAULT_NETWORK_MESSAGE_CHUNK
    network_data_encoded = base64.b64encode(network_data).encode('utf-8')

    client.publish(topic, json.dumps(DEFAULT_NETWORK_INIT_MESSAGE))
    for chunk in divide_chunks(network_data_encoded, DEFAULT_PACKET_SIZE):
        network_message_chunk['data'] = chunk
        client.publish(topic, json.dumps(network_message_chunk))
    client.publish(topic, json.dumps(DEFAULT_NETWORK_END_MESSAGE))

def send_typed_message(client, topic, message, message_type):
    if message_type is MessageType.NETWORK_CHUNK:
        send_network_chunk_message(client, topic, message)
    else:
        print('{} not handled'.format(message_type))