import json
import sys

from common import person_classifier
from common.aggregation_scheme import get_aggregation_scheme
from common.datablock import Datablock
from common.models import PersonBinaryClassifier
from common.networkblock import Networkblock, NetworkStatus

from utils.enums import LearningType, ClientState

from flask_mqtt import Mqtt
from flask import Flask

from utils import constants
from utils.model_helper import decode_state_dictionary, encode_state_dictionary
from utils.mqtt_helper import MessageType, send_typed_message

import traceback

sys.path.append('.')

app = Flask(__name__)
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_REFRESH_TIME'] = 1.0  # refresh time in seconds
mqtt = Mqtt(app, mqtt_logging=True)


# global variables
PACKET_SIZE = 3000
CLIENTS = dict()
CLIENT_DATABLOCKS = {}
CLIENT_NETWORKS = {}
NETWORK = None

CONFIGURATION = LearningType.CENTRALIZED

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

@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print('connected')
    mqtt.subscribe(constants.NEW_CLIENT_INITIALIZATION_TOPIC)




@mqtt.on_message()
def handle_mqtt_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    dimensions = payload.get("dimensions", None)
    label = payload.get("label", None)
    data = payload.get("data", None)
    message = payload.get("message", None)

    # Add a new client and subscribe to appropriate topic
    if msg.topic == constants.NEW_CLIENT_INITIALIZATION_TOPIC:
        initialize_new_clients(message)
        return



    client_name = msg.topic.split("/")[1]
    if client_name in CLIENTS:
        if CLIENTS[client_name]["learning_type"] == LearningType.FEDERATED:
            collect_federated_data(data, message, client_name)
        elif CLIENTS[client_name]["learning_type"] == LearningType.CENTRALIZED:
            collect_centralized_data(
                data, message, client_name, dimensions, label)


    # check if all clients finished sending data

    if all_clients_finished():
        perform_aggregation_and_send_model()


def perform_aggregation_and_send_model():
    global NETWORK, CLIENTS, CLIENT_NETWORKS

    # Current method averages and then trains
    try:
        # Average models
        averaged_state_dict = get_aggregation_scheme(
            CLIENTS, CLIENT_NETWORKS)

        if averaged_state_dict is not None:
            NETWORK = PersonBinaryClassifier()
            NETWORK.load_state_dictionary(averaged_state_dict)

            print("Averaging Finished")

            # runner = person_classifier.get_model_runner()
            # runner.model.load_state_dictionary(
            #     NETWORK.get_state_dictionary())
            # runner.test_model()

            # reset models to stale and delete old data
            for client in CLIENTS:
                print("Resetting network data for client {}..".format(client))
                CLIENT_NETWORKS[client].reset_network_data()

        if len(CLIENT_DATABLOCKS) != 0:
            runner = person_classifier.get_model_runner(client_data=CLIENT_DATABLOCKS, num_epochs=1)
            if averaged_state_dict is not None:
                runner.model.load_state_dictionary(NETWORK.get_state_dictionary())
            runner.train_model()
            encoded = encode_state_dictionary(runner.model.get_state_dictionary())
        else:
            encoded = encode_state_dictionary(NETWORK.get_state_dictionary())

        send_network_model(encoded)
        for client in CLIENTS:
            CLIENTS[client]["state"] = ClientState.STALE
    except Exception as e:
        print(traceback.format_exc())


def all_clients_finished():
    for client_id in CLIENTS:
        if CLIENTS[client_id]["state"] != ClientState.FINISHED:
            # print("client {} is not finished".format(client_id))
            return False

    print("All clients finished")

    return True


def collect_federated_data(data, message, client_id):
    global CLIENT_NETWORKS, CLIENTS

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

        CLIENTS[client_id]["state"] = ClientState.FINISHED


def publish_new_model(network):
    print('Publishing new model to clients..')
    state_dict = encode_state_dictionary(network.get_state_dictionary())
    send_network_model(state_dict)
    print('Successfully published new models to clients.')


def collect_centralized_data(data, message, client_name, dimensions, label):
    global CLIENTS
    if message == constants.DEFAULT_IMAGE_INIT:
        initialize_new_image(client_name, dimensions, label)
    elif message == constants.DEFAULT_IMAGE_CHUNK:
        add_data_chunk(client_name, data)
    elif message == constants.DEFAULT_IMAGE_END:
        convert_data(client_name)
    elif message == 'all_images_sent':
        CLIENTS[client_name]["state"] = ClientState.FINISHED
        print("you can train now")

def initialize_new_clients(client_id):
    print("New client connected: {}".format(client_id))
    CLIENTS[client_id] = {
        "learning_type": LearningType.FEDERATED,
        "state": ClientState.STALE}

    if CLIENTS[client_id]["learning_type"] == LearningType.CENTRALIZED:
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
