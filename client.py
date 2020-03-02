import pickle
import time
import json
import numpy as np
import uuid
import torch

import paho.mqtt.client as mqtt
from common import person_classifier
from common.datablock import Datablock
import utils.constants as constants
from utils.mqtt_helper import send_typed_message, MessageType, divide_chunks
from utils.model_helper import get_state_dictionary

NETWORK_STRING = ''
DEFAULT_BATCH_SIZE = 15
PI_ID = 'pi{}'.format(uuid.uuid4())
DEVICE_TOPIC = 'client/{}'.format(PI_ID)
SEND_MODEL = False

########################################
# model stuff
########################################


def reconstruct_model():
    global NETWORK_STRING

    state_dict = get_state_dictionary(NETWORK_STRING)
    return state_dict


def test():
    person_test_samples = pickle.load(
        open('./data/personimagesTest.pkl', 'rb'))
    person_test_images = [sample[0] for sample in person_test_samples]
    no_person_test_samples = pickle.load(
        open('./data/nopersonimagesTest.pkl', 'rb'))
    no_person_test_images = [sample[0] for sample in no_person_test_samples]

    images = np.concatenate((person_test_images, no_person_test_images))

    num_test_samples = len(person_test_images)
    labels = np.concatenate((
        np.ones(num_test_samples, dtype=np.int_),
        np.zeros(num_test_samples, dtype=np.int_)
    ))

    datablocks = {'1': Datablock(images=images, labels=labels)}
    runner = person_classifier.get_model_runner(datablocks)

    state_dictionary = reconstruct_model()
    runner.model.load_last_layer_state_dictionary(state_dictionary)
    runner.test_model()

########################################
# sending stuff
########################################


def publish_encoded_image(image, label):
    sample = (image, label)
    send_typed_message(client, DEVICE_TOPIC, sample, MessageType.IMAGE_CHUNK)


def publish_encoded_model(payload):
    send_typed_message(
        client,
        DEVICE_TOPIC,
        payload,
        MessageType.NETWORK_CHUNK)


def send_images():
    persons_data = pickle.load(open('./data/personimages.pkl', 'rb'))
    no_persons_data = pickle.load(open('./data/nopersonimages.pkl', 'rb'))

    batch_size = DEFAULT_BATCH_SIZE
    for label, images in enumerate([no_persons_data, persons_data]):
        for chunk in divide_chunks(images, batch_size):
            for sample in chunk:
                image, _ = sample
                publish_encoded_image(image, label)
            print('Sleep after sending batch of {}'.format(batch_size))
            time.sleep(1)
    print('sent all images!')


def send_model():
    persons_data = pickle.load(open('./data/personimages.pkl', 'rb'))
    no_persons_data = pickle.load(open('./data/nopersonimages.pkl', 'rb'))

    datablock = Datablock()

    for label, images in enumerate([no_persons_data, persons_data]):
        for image, _ in images:
            datablock.init_new_image(image.shape, label)
            datablock.image_data[-1] = image

    datablock_dict = {"pi01": datablock}

    model = person_classifier.train(datablock_dict)

    model.save('./network.pth')

    state_dict = open('./network.pth', 'rb').read()

    publish_encoded_model(state_dict)

    print('model_sent!')

#########################################
# mqtt stuff
#########################################


def send_client_id():
    global DEVICE_TOPIC
    message = {
        "message": PI_ID
    }
    client.subscribe(DEVICE_TOPIC)
    send_typed_message(
        client,
        constants.NEW_CLIENT_INITIALIZATION_TOPIC,
        message,
        MessageType.SIMPLE)


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    send_client_id()
    client.subscribe("server/network")
    if SEND_MODEL:
        send_model()
    else:
        send_images()

    print("publishing images done")

# The callback for when a PUBLISH message is received from the server.


def on_message(client, userdata, msg):
    global NETWORK_STRING

    payload = json.loads(msg.payload.decode())
    message_type = payload["message"]
    if message_type == constants.DEFAULT_NETWORK_INIT:
        print("transmitting network data")
        print("-" * 10)
    elif message_type == constants.DEFAULT_NETWORK_CHUNK:
        NETWORK_STRING += payload["data"]
    elif message_type == constants.DEFAULT_NETWORK_END:
        print("done, running evaluation on transmitted model")
        test()
    else:
        print('Could not handle message')


def on_publish(client, userdata, result):
    print("data published")


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883, 65534)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.

client.loop_forever()
