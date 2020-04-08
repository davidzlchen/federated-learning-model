import pickle
import time
import json
import numpy as np
import uuid

import paho.mqtt.client as mqtt
from common import person_classifier
from common.datablock import Datablock
import utils.constants as constants
from utils.mqtt_helper import send_typed_message, MessageType, divide_chunks
from utils.model_helper import decode_state_dictionary, encode_state_dictionary
from common.configuration import *
import traceback

NETWORK_STRING = ''
DEFAULT_BATCH_SIZE = 15

DATABLOCK = Datablock()
DATA_INDEX = 0
SEND_MODEL = False
MODEL_TRAIN_SIZE = 24
RUNNER = None
CONFIGURATION = Configuration()

PI_ID = 'pi{}'.format(uuid.uuid4())
DEVICE_TOPIC = 'client/{}'.format(PI_ID)


########################################
# model stuff
########################################


def reconstruct_model():
    global NETWORK_STRING

    state_dict = decode_state_dictionary(NETWORK_STRING)
    return state_dict


def test(reconstruct=False):
    if not SEND_MODEL:
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
        runner.model.load_state_dictionary(state_dictionary)
        runner.test_model()
    else:
        global RUNNER

        if not RUNNER:
            RUNNER = person_classifier.get_model_runner(None)
        if reconstruct:
            state_dictionary = reconstruct_model()
            RUNNER.model.load_state_dictionary(state_dictionary)

        RUNNER.test_model()

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

    end_msg = {
        'message': 'all_images_sent'
    }
    send_typed_message(client, DEVICE_TOPIC, json.dumps(end_msg), MessageType.SIMPLE)


def setup_data():
    global DATABLOCK, DATA_INDEX
    persons_data = pickle.load(open('./data/personimages.pkl', 'rb'))
    no_persons_data = pickle.load(open('./data/nopersonimages.pkl', 'rb'))

    for label, images in enumerate([no_persons_data, persons_data]):
        for image, _ in images:
            DATABLOCK.init_new_image(image.shape, label)
            DATABLOCK.image_data[-1] = image

    DATABLOCK.shuffle_data()

def send_model(statedict):
    global DATABLOCK, DATA_INDEX, MODEL_TRAIN_SIZE, RUNNER
    print("State dict before training: ")
    print(statedict)
    datablock_dict = {
        'pi01': DATABLOCK[DATA_INDEX:DATA_INDEX + MODEL_TRAIN_SIZE]}

    RUNNER = person_classifier.get_model_runner(datablock_dict)

    if DATA_INDEX != 0:
        RUNNER.model.load_state_dictionary(statedict)

    print(
        "Training on images {} to {}".format(
            DATA_INDEX,
            DATA_INDEX +
            MODEL_TRAIN_SIZE -
            1))
    DATA_INDEX += MODEL_TRAIN_SIZE

    RUNNER.train_model()
    print("Successfully trained model.")
    test()
    print("Finished testing model.")

    state_dict = RUNNER.model.get_state_dictionary()
    binary_state_dict = encode_state_dictionary(state_dict)
    publish_encoded_model(binary_state_dict)

    print('State dictionary sent to central server!')


#########################################
# mqtt stuff
#########################################

def send_client_id():
    global DEVICE_TOPIC
    message = {
        "message": PI_ID
    }
    send_typed_message(
        client,
        constants.NEW_CLIENT_INITIALIZATION_TOPIC,
        message,
        MessageType.SIMPLE)


def on_connect(client, userdata, flags, rc):
    send_client_id()
    client.subscribe("server/network")
    print("Connected with result code " + str(rc))

# The callback for when a PUBLISH message is received from the server.


def on_log(client, userdata, level, buf):
    if level != mqtt.MQTT_LOG_DEBUG:
        print(traceback.format_exc())
        print("log: ",buf)
        print("level", level)
        exit()


def on_message(client, userdata, msg):
    global NETWORK_STRING
    global CONFIGURATION

    client.on_log = on_log

    payload = json.loads(msg.payload.decode())
    message_type = payload["message"]
    if message_type == constants.DEFAULT_NETWORK_INIT:
        print("-" * 10)
        print("Receiving network data...")
        NETWORK_STRING = ''
    elif message_type == constants.DEFAULT_NETWORK_CHUNK:
        NETWORK_STRING += payload["data"]
    elif message_type == constants.DEFAULT_NETWORK_END:
        try:
            print("Finished receiving network data, loading state dictionary")
            state_dict = decode_state_dictionary(NETWORK_STRING)
            if SEND_MODEL:
                send_model(state_dict)
            else:
                test()
        except Exception as e:
            print(traceback.format_exc())
    elif message_type == constants.SEND_CLIENT_DATA:
        if CONFIGURATION.learning_type == LearningType.FEDERATED:
            print("federated")
            setup_data()
            send_model(None)
        elif CONFIGURATION.learning_type == LearningType.CENTRALIZED:
            print("centralized")
            send_images()
        print("end")
    elif message_type == constants.CONFIGURATION_MESSAGE_SIGNAL:
        configuration_object = as_configuration(payload['data'])
        CONFIGURATION = configuration_object
    else:
        print('Could not handle message: ', message_type)


def on_publish(client, userdata, result):
    print("data published")


client = mqtt.Client(client_id=PI_ID)
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883, 65534)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.

client.loop_forever()
