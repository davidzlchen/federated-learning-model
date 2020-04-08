import torch
from common.configuration import *
from utils.mqtt_helper import *
from common.models import PersonBinaryClassifier
from common.networkblock import Networkblock, NetworkStatus
from utils.mqtt_helper import MessageType, send_typed_message
from common.datablock import Datablock
from common import person_classifier
from flask_mqtt import Mqtt
from flask import Flask
from utils.model_helper import decode_state_dictionary, encode_state_dictionary
from common.aggregation_scheme import get_aggregation_scheme
from enum import Enum
import logging
from utils import constants
import json
import base64
import sys
sys.path.append('.')

app = Flask(__name__)
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_REFRESH_TIME'] = 1.0  # refresh time in seconds
mqtt = Mqtt(app, mqtt_logging=True)

# global variables
PACKET_SIZE = 3000
CLIENT_IDS = set()
CLIENT_DATABLOCKS = {}
CLIENT_NETWORKS = {}

CONFIGURATION = Configuration(LearningType.FEDERATED)

pinged_once = False


@app.route('/')
def index():
    global CLIENT_DATABLOCKS
    global pinged_once

    if not pinged_once:

        send_typed_message(
            mqtt,
            "server/network",
            {'message': constants.SEND_CLIENT_DATA},
            MessageType.SIMPLE)
        pinged_once = True
        return "Sent command to receive models.\n"
    else:
        if CONFIGURATION == LearningType.CENTRALIZED:
            pbc = person_classifier.train(CLIENT_DATABLOCKS)
            encoded = encode_state_dictionary(pbc.model.state_dict())
            send_network_model(encoded)
            return 'Sent model to clients'


@app.route('/gui')
def start_with_configuration():
    global CLIENT_DATABLOCKS

    # Set global Configuration with input from GUI
    send_configuration_message(
        mqtt,
        "server/network",
        CONFIGURATION)

    send_typed_message(
        mqtt,
        "server/network",
        {'message': constants.SEND_CLIENT_DATA},
        MessageType.SIMPLE)

    return "Sent configuration command and then command to accept data"

@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print('connected')
    mqtt.subscribe(constants.NEW_CLIENT_INITIALIZATION_TOPIC)


@mqtt.on_message()
def handle_mqtt_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        dimensions = payload.get("dimensions", None)
        label = payload.get("label", None)
        data = payload.get("data", None)
        message = payload.get("message", None)

        # Add a new client and subscribe to appropriate topic
        if msg.topic == constants.NEW_CLIENT_INITIALIZATION_TOPIC:
            print('nice')
            initialize_new_clients(message)
            return

        client_name = msg.topic.split("/")[1]
        if client_name in CLIENT_IDS:
            if CONFIGURATION.learning_type == LearningType.FEDERATED:
                collect_federated_data(data, message, client_name)
            elif CONFIGURATION.learning_type == LearningType.CENTRALIZED:
                collect_centralized_data(
                    data, message, client_name, dimensions, label)

    except Exception as e:
        print(e)
        exit(1)


def collect_federated_data(data, message, client_id):
    global NETWORK, CLIENT_NETWORKS, CLIENT_IDS

    # get model
    if message == constants.DEFAULT_NETWORK_INIT:
        CLIENT_NETWORKS[client_id] = Networkblock()
        CLIENT_NETWORKS[client_id].reset_network_data()

    elif message == constants.DEFAULT_NETWORK_CHUNK:
        CLIENT_NETWORKS[client_id].add_network_chunk(data)

    elif message == constants.DEFAULT_NETWORK_END:
        print("All chunks received")
        state_dict = CLIENT_NETWORKS[client_id].reconstruct_state_dict()
        person_binary_classifier = PersonBinaryClassifier()
        person_binary_classifier.load_state_dictionary(state_dict)

        # check if all new models have been added
        for client in CLIENT_IDS:
            if CLIENT_NETWORKS[client].network_status == NetworkStatus.STALE:
                print("{} is stale, won't average.".format(client_id))
                return

        #average models
        averaged_state_dict = get_aggregation_scheme(
            CLIENT_IDS, CLIENT_NETWORKS)

        NETWORK = PersonBinaryClassifier()
        NETWORK.load_state_dictionary(averaged_state_dict)

        print("Averaging Finished")

        runner = person_classifier.get_model_runner()
        runner.model.load_state_dictionary(
            NETWORK.get_state_dictionary())
        runner.test_model()

        # reset models to stale and delete old data
        for client in CLIENT_IDS:
            print("Resetting network data for client {}..".format(client))
            CLIENT_NETWORKS[client].reset_network_data()
        publish_new_model()


def publish_new_model():
    global NETWORK
    print('Publishing new model to clients..')
    state_dict = encode_state_dictionary(NETWORK.get_state_dictionary())
    send_network_model(state_dict)
    print('Successfully published new models to clients.')


def collect_centralized_data(data, message, client_name, dimensions, label):
    if message == constants.DEFAULT_IMAGE_INIT:
        initialize_new_image(client_name, dimensions, label)
    elif message == constants.DEFAULT_IMAGE_CHUNK:
        add_data_chunk(client_name, data)
    elif message == constants.DEFAULT_IMAGE_END:
        convert_data(client_name)
    elif message == 'all_images_sent':
        print("you can train now")

def initialize_new_clients(client_id):
    print("New client connected: {}".format(client_id))
    CLIENT_IDS.add(client_id)
    initialize_datablocks(client_id)
    mqtt.subscribe('client/' + client_id)


def initialize_new_image(client_name, dimensions, label):
    global CLIENT_DATABLOCKS

    datablock = CLIENT_DATABLOCKS[client_name]
    datablock = datablock.init_new_image(dimensions, label)


def add_data_chunk(client_name, chunk):
    global CLIENT_DATABLOCKS

    datablock = CLIENT_DATABLOCKS[client_name]
    datablock.add_image_chunk(chunk)


def convert_data(client_name):
    global CLIENT_DATABLOCKS

    datablock = CLIENT_DATABLOCKS[client_name]
    datablock.convert_current_image_to_matrix()


def send_network_model(payload):
    send_typed_message(
        mqtt,
        "server/network",
        payload,
        MessageType.NETWORK_CHUNK)


def initialize_datablocks(client):
    global CLIENT_DATABLOCKS
    CLIENT_DATABLOCKS[client] = Datablock()


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
