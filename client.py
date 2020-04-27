import pickle
import time
import json
import numpy as np
import uuid
import sys
import platform

import paho.mqtt.client as mqtt
from common import person_classifier
from common.datablock import Datablock
from common.result_data import *
import utils.constants as constants
from utils.mqtt_helper import send_typed_message, MessageType, divide_chunks
from utils.model_helper import decode_state_dictionary, encode_state_dictionary
from common.configuration import *
import traceback

NETWORK_STRING = ''
DATA_SIZE = 0

DATABLOCK = Datablock()
TEST_DATABLOCK = Datablock()
DATA_INDEX = 0
MODEL_TRAIN_SIZE = 25
RUNNER = None
CONFIGURATION = Configuration()
TOTAL_DATA_COUNT = 0
DATA_PARTITION_INDEX = 0
NUM_DATA_PARTITIONS = 0

PI_ID = 'pi{}'.format(uuid.uuid4())
print(PI_ID)
DEVICE_TOPIC = 'client/{}'.format(PI_ID)

CLUSTER_TOPIC = None


########################################
# personalized model
########################################

def personalized():
    global MODEL_TRAIN_SIZE
    setup_data()
    MODEL_TRAIN_SIZE = 240
    train(None)
    test()
    print("Finished testing model.")

########################################
# model stuff
########################################


def train(statedict):
    global DATABLOCK, TEST_DATABLOCK, DATA_INDEX, MODEL_TRAIN_SIZE, RUNNER
    # print("State dict before training: ")
    # print(statedict)
    datablock_dict = {
        'pi01': DATABLOCK[DATA_INDEX:DATA_INDEX + MODEL_TRAIN_SIZE]}

    test_datablock_dict = {
        'pi01': TEST_DATABLOCK}

    RUNNER = person_classifier.get_model_runner(client_data=datablock_dict, test_data=test_datablock_dict)

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


def reconstruct_model():
    global NETWORK_STRING

    state_dict = decode_state_dictionary(NETWORK_STRING)
    return state_dict


def test(reconstruct=False):
    global RUNNER, TEST_DATABLOCK, DATA_SIZE, DATA_INDEX, MODEL_TRAIN_SIZE

    test_datablock_dict = {
        'pi01': TEST_DATABLOCK}

    if not RUNNER:
        RUNNER = person_classifier.get_model_runner(test_data=test_datablock_dict)
    if reconstruct:
        state_dictionary = reconstruct_model()
        RUNNER.model.load_state_dictionary(state_dictionary)

    ResultData = RUNNER.test_model()
    ResultData.size = DATA_SIZE
    ResultData.specs = platform.uname()
    ResultData.iteration = DATA_INDEX/MODEL_TRAIN_SIZE
    ResultData.epochs = RUNNER.epochs
    send_typed_message(
        client,
        DEVICE_TOPIC,
        ResultData,
        MessageType.RESULT_DATA)  # send results to server

########################################
# sending stuff
########################################


def publish_encoded_image(image, label):
    sample = (image, label)
    return send_typed_message(client, DEVICE_TOPIC, sample, MessageType.IMAGE_CHUNK)


def publish_encoded_model(payload):
    send_typed_message(
        client,
        DEVICE_TOPIC,
        payload,
        MessageType.NETWORK_CHUNK)


def send_images():
    global DATABLOCK, DATA_INDEX, DATA_SIZE

    for i in range(DATA_INDEX, DATA_INDEX + MODEL_TRAIN_SIZE):
        image = DATABLOCK.image_data[i]
        label = DATABLOCK.labels[i]
        DATA_SIZE += publish_encoded_image(image, label)
    print(
        "images {} to {} sent".format(
            DATA_INDEX,
            DATA_INDEX +
            MODEL_TRAIN_SIZE -
            1))
    DATA_INDEX += MODEL_TRAIN_SIZE


    end_msg = {
        'message': 'all_images_sent'
    }
    send_typed_message(
        client,
        DEVICE_TOPIC,
        json.dumps(end_msg),
        MessageType.SIMPLE)


def setup_data():
    global DATABLOCK, DATA_INDEX, TOTAL_DATA_COUNT

    data = pickle.load(open('./data/federated-learning-data.pkl', 'rb'))
    num_images = len(data)
    split_index = int(num_images * 4 / 5)   # 20% for testing
    train_data = data[0:split_index]
    test_data = data[split_index:]

    DATABLOCK.add_images_for_cluster(train_data, CLUSTER_TOPIC, partition=True, partition_index=DATA_PARTITION_INDEX, num_partitions=NUM_DATA_PARTITIONS)
    TEST_DATABLOCK.add_images_for_cluster(test_data, CLUSTER_TOPIC)

    TOTAL_DATA_COUNT = DATABLOCK.num_images


def send_model(statedict):
    global DATA_SIZE
    train(statedict)
    state_dict = RUNNER.model.get_state_dictionary()
    binary_state_dict = encode_state_dictionary(state_dict)
    DATA_SIZE = int(sys.getsizeof(binary_state_dict))
    print(DATA_SIZE)
    test()

    print("Finished testing model.")

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
        print("log: ", buf)
        print("level", level)
        exit()


def on_message(client, userdata, msg):
    global CLUSTER_TOPIC, NUM_DATA_PARTITIONS, DATA_PARTITION_INDEX

    payload = json.loads(msg.payload.decode())
    message_type = payload["message"]

    if msg.topic == CLUSTER_TOPIC:
        process_network_data(client, message_type, payload)

    elif message_type == constants.START_LEARNING:
        if CONFIGURATION.learning_type == LearningType.FEDERATED:
            setup_data()
            send_model(None)
        elif CONFIGURATION.learning_type == LearningType.CENTRALIZED:
            setup_data()
            send_images()
        elif CONFIGURATION.learning_type == LearningType.PERSONALIZED:
            personalized()

    elif message_type == constants.SUBSCRIBE_TO_CLUSTER:
        # remove current cluster topic and subscribe to new cluster topic

        if payload['client_id'] != PI_ID:
            return

        if payload['learning_type'] == 'federated':
            CONFIGURATION.learning_type = LearningType.FEDERATED
        elif payload['learning_type'] == 'centralized':
            CONFIGURATION.learning_type = LearningType.CENTRALIZED
        else:
            CONFIGURATION.learning_type = LearningType.PERSONALIZED

        if CLUSTER_TOPIC is not None:
            client.unsubscribe(CLUSTER_TOPIC)
        CLUSTER_TOPIC = payload[constants.CLUSTER_TOPIC_NAME]

        NUM_DATA_PARTITIONS = payload['num_clients_in_cluster']
        DATA_PARTITION_INDEX = payload['client_index_in_cluster']

        print("New cluster topic: {}".format(CLUSTER_TOPIC))
        client.subscribe(CLUSTER_TOPIC)

    elif message_type == constants.RESET_CLIENT:
        reset_client()

    else:
        print(message_type)
        print('Could not handle message: {} -- topic: {}'.format(message_type, msg.topic))


def process_network_data(client, message_type, payload):
    global NETWORK_STRING

    if message_type == constants.DEFAULT_NETWORK_INIT:
        print("-" * 10)
        print("Receiving network data...")
        NETWORK_STRING = ''
    elif message_type == constants.DEFAULT_NETWORK_CHUNK:
        NETWORK_STRING += payload["data"]
    elif message_type == constants.DEFAULT_NETWORK_END:
        print("Finished receiving network data, loading state dictionary")
        state_dict = decode_state_dictionary(NETWORK_STRING)
        if CONFIGURATION.learning_type == LearningType.FEDERATED:
            if DATA_INDEX + MODEL_TRAIN_SIZE > TOTAL_DATA_COUNT:
                send_typed_message(client, DEVICE_TOPIC, json.dumps(constants.DEFAULT_ITERATION_END_MESSAGE), MessageType.SIMPLE)
                print("client is finished")
            else:
                send_model(state_dict)
        elif CONFIGURATION.learning_type == LearningType.CENTRALIZED:
            if DATA_INDEX + MODEL_TRAIN_SIZE > TOTAL_DATA_COUNT:
                test(True)
                send_typed_message(client, DEVICE_TOPIC, json.dumps(constants.DEFAULT_ITERATION_END_MESSAGE), MessageType.SIMPLE)
                print("client is finished")
            else:
                test(True)
                send_images()
        else:
            test(True)


def reset_client():
    global CONFIGURATION, CLUSTER_TOPIC, NETWORK_STRING, DATABLOCK, DATA_INDEX, RUNNER, DATA_SIZE

    CONFIGURATION.learning_type = LearningType.NONE
    CLUSTER_TOPIC = None
    NETWORK_STRING = ''

    DATABLOCK = Datablock()
    DATA_INDEX = 0

    DATA_SIZE = 0
    RUNNER = None


def on_publish(client, userdata, result):
    print("data published")


client = mqtt.Client(client_id=PI_ID)
client.on_connect = on_connect
client.on_message = on_message
client.on_log = on_log
client.connect("broker.hivemq.com", 1883, 65534)
#client.connect("localhost", 1883, 65534)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.

client.loop_forever()
