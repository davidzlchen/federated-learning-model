import pickle
import time
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
MODEL_TRAIN_SIZE = 24
RUNNER = None
CONFIGURATION = Configuration()

PI_ID = 'pi{}'.format(uuid.uuid4())
DEVICE_TOPIC = 'client/{}'.format(PI_ID)

CLUSTER_TOPIC = None


########################################
# model stuff
########################################


def reconstruct_model():
    global NETWORK_STRING

    state_dict = decode_state_dictionary(NETWORK_STRING)
    return state_dict


def test(reconstruct=False):
    if CONFIGURATION.learning_type == LearningType.CENTRALIZED:
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
    counter = 0
    for label, images in enumerate([no_persons_data, persons_data]):

        # for chunk in divide_chunks(images, batch_size):
        #     for sample in chunk:
        #         image, _ = sample
        #         publish_encoded_image(image, label)
        #     print('Sleep after sending batch of {}'.format(batch_size))
        #     time.sleep(1)

        for image, attributes in images:
            publish_encoded_image(image, label)
            counter += 1
            if counter % 15 == 0:
                time.sleep(1)
                print('{} images sent, sleeping for 1 second.'.format(batch_size))
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

    print(state_dict)

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

    client.subscribe("server/general")
    # client.subscribe("server/network")
    print("Connected with result code " + str(rc))

# The callback for when a PUBLISH message is received from the server.


def on_log(client, userdata, level, buf):
    if level != mqtt.MQTT_LOG_DEBUG:
        print(traceback.format_exc())
        print("log: ",buf)
        print("level", level)
        exit()


def on_message(client, userdata, msg):
    global CLUSTER_TOPIC

    payload = json.loads(msg.payload.decode())
    message_type = payload["message"]

    if msg.topic == CLUSTER_TOPIC:
        process_network_data(message_type, payload)

    elif message_type == constants.SEND_CLIENT_DATA:
        if CONFIGURATION.learning_type == LearningType.FEDERATED:
            setup_data()
            send_model(None)
        elif CONFIGURATION.learning_type == LearningType.CENTRALIZED:
            send_images()

    elif message_type == constants.SUBSCRIBE_TO_CLUSTER:
        # remove current cluster topic and subscribe to new cluster topic

        if payload['client_id'] != PI_ID:
            return

        if payload['learning_type'] == 'federated':
            CONFIGURATION.learning_type = LearningType.FEDERATED
        else:
            CONFIGURATION.learning_type = LearningType.CENTRALIZED

        if CLUSTER_TOPIC is not None:
            client.unsubscribe(CLUSTER_TOPIC)
        CLUSTER_TOPIC = payload[constants.CLUSTER_TOPIC_NAME]

        print("New cluster topic: {}".format(CLUSTER_TOPIC))
        client.subscribe(CLUSTER_TOPIC)

    elif message_type == constants.RESET_CLIENT:
        reset_client()

    else:
        print(message_type)
        print('Could not handle message: {} -- topic: {}'.format(message_type, msg.topic))

def process_network_data(message_type, payload):
    global NETWORK_STRING

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
            if CONFIGURATION.learning_type == LearningType.FEDERATED:
                send_model(state_dict)
            else:
                test()
        except Exception as e:
            print(traceback.format_exc())

def reset_client():
    global CONFIGURATION, CLUSTER_TOPIC, NETWORK_STRING, DATABLOCK, DATA_INDEX, RUNNER

    CONFIGURATION.learning_type = LearningType.NONE
    CLUSTER_TOPIC = None
    NETWORK_STRING = ''

    DATABLOCK = Datablock()
    DATA_INDEX = 0

    RUNNER = None


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
