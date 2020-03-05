import base64
import json
import utils.constants as constants
from enum import Enum

from common.aggregation_scheme import get_aggregation_scheme
from utils.model_helper import get_state_dictionary

from flask import Flask
from flask_mqtt import Mqtt
from common import person_classifier
from common.datablock import Datablock
from utils.mqtt_helper import MessageType, send_typed_message
from common.networkblock import Networkblock, NetworkStatus
from common.models import get_default_model

app = Flask(__name__)
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_REFRESH_TIME'] = 1.0  # refresh time in seconds
mqtt = Mqtt(app)

class LearningType(Enum):
    CENTRALIZED=1
    FEDERATED=2

# global variables
PACKET_SIZE = 3000
CLIENT_IDS = set()
CLIENT_DATABLOCKS = {}
CLIENT_NETWORKS = {}

CONFIGURATION = LearningType.FEDERATED
NETWORK = get_default_model()


@app.route('/')
def index():
    global CLIENT_DATABLOCKS


    send_typed_message(
        mqtt,
        "server/network",
        {'message': constants.SEND_CLIENT_DATA},
        MessageType.SIMPLE)

    # model = person_classifier.train(CLIENT_DATABLOCKS)
    # model.save('./network.pth')
    # state_dict = open('./network.pth', 'rb').read()
    # send_network_model(state_dict)

    return "Sent command to accept data."


@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print("connected")
    mqtt.subscribe(constants.NEW_CLIENT_INITIALIZATION_TOPIC)

@mqtt.on_log()
def handle_logging(client, userdata, level, buf):
    # print(level, buf)
    pass


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
    if client_name in CLIENT_IDS:
        if CONFIGURATION == LearningType.FEDERATED:
            collect_federated_data(data, message, client_name)
        elif CONFIGURATION == LearningType.CENTRALIZED:
            collect_centralized_data(
                data, message, client_name, dimensions, label)


def collect_federated_data(data, message, client_id):
    global NETWORK

    # get model
    if message == constants.DEFAULT_NETWORK_INIT:
        CLIENT_NETWORKS[client_id] = Networkblock()
        CLIENT_NETWORKS[client_id].destroy_network_data()

    elif message == constants.DEFAULT_NETWORK_CHUNK:
        CLIENT_NETWORKS[client_id].add_network_chunk(data)

    elif message == constants.DEFAULT_NETWORK_END:
        state_dict = CLIENT_NETWORKS[client_id].reconstruct_state_dict()
        person_binary_classifier = get_default_model()
        person_binary_classifier.load_last_layer_state_dictionary(state_dict)


        # check if all new models have been added
        for client_id in CLIENT_IDS:
            if CLIENT_NETWORKS[client_id].network_status == NetworkStatus.STALE:
                print("returning because {} is stale.".format(client_id))
                return


        weights, bias = get_aggregation_scheme(
            NETWORK, CLIENT_IDS, CLIENT_NETWORKS)


        NETWORK.model.fc.state_dict()['weight'].copy_(weights)
        NETWORK.model.fc.state_dict()['bias'].copy_(bias)


        for client_id in CLIENT_IDS:
            print("client: {}".format(client_id))
            print(CLIENT_NETWORKS[client_id].state_dict)
            print("")


        print("averaged: ")
        print(NETWORK.model.fc.state_dict())

        # reset models to stale and delete old data
        for client_id in CLIENT_IDS:
            print("reseting")
            CLIENT_NETWORKS[client_id].reset_network_data()

        publish_new_model()


def publish_new_model():
    global NETWORK
    print("NEW")
    print(NETWORK.model.fc.state_dict())
    NETWORK.save('./new_network.pth')
    print("-3")
    state_dict = open('./new_network.pth', 'rb').read()

    print("-4")


    new_state_dict = torch.load('./new_network.pth')

    print("<-1>")


    new_classifier = get_default_model()

    print("-.5")

    new_classifier.load_last_layer_state_dictionary(new_state_dict)
    print("<0>")


    print(NETWORK.model.fc.state_dict())


    print("<1>")
    print(new_classifier.model.fc.state_dict())

    print("<2>")
    
    send_network_model(state_dict)


    state_dict.close()
   


def collect_centralized_data(data, message, client_name, dimensions, label):
    if message == constants.DEFAULT_IMAGE_INIT:
        initialize_new_image(client_name, dimensions, label)
    elif message == constants.DEFAULT_IMAGE_CHUNK:
        add_data_chunk(client_name, data)
    elif message == constants.DEFAULT_IMAGE_END:
        convert_data(client_name)


def initialize_new_clients(client_id):
    print("connect: {}".format(client_id))
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

    print('done')


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
