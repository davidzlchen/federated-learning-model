import pickle
import time
import json
import numpy as np

import paho.mqtt.client as mqtt
import person_classifier
import utils.constants as constants
from datablock import Datablock
from utils.mqtt_helper import send_typed_message, MessageType, divide_chunks
from utils.model_helper import get_state_dictionary

NETWORK_STRING = ''
DEFAULT_BATCH_SIZE = 15
DEFAULT_TOPIC = 'client/pi01'

########################################
# model stuff
########################################

def reconstruct_model():
    global NETWORK_STRING

    state_dict = get_state_dictionary(NETWORK_STRING)
    return state_dict

def test():
    person_test_samples = pickle.load(open('./data/personimagesTest.pkl', 'rb'))
    person_test_images = [sample[0] for sample in person_test_samples]
    no_person_test_samples = pickle.load(open('./data/nopersonimagesTest.pkl', 'rb'))
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
# sending image stuff
########################################

def publish_encoded_image(image, label):
    sample = (image, label)
    send_typed_message(client, DEFAULT_TOPIC, sample, MessageType.IMAGE_CHUNK)

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

#########################################
# mqtt stuff
#########################################

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("server/network")
    send_images()
    print("publishing images done")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global NETWORK_STRING

    payload = json.loads(msg.payload.decode())
    message_type = payload["message"]
    if (message_type == constants.DEFAULT_NETWORK_INIT):
        print("transmitting network data")
        print("-"*10)
    elif (message_type == constants.DEFAULT_NETWORK_CHUNK):
        NETWORK_STRING += payload["data"]
    elif (message_type == constants.DEFAULT_NETWORK_END):
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
