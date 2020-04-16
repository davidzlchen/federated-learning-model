import json
import sys
import time
import traceback

from flask import Flask, request
from flask_socketio import SocketIO
from flask_mqtt import Mqtt
from flask_cors import CORS
from flask_socketio import SocketIO, send, emit

from common.aggregation_scheme import get_aggregation_scheme
from common.configuration import *
from utils.mqtt_helper import *
from common.models import PersonBinaryClassifier
from common.networkblock import Networkblock, NetworkStatus
from common.clientblock import ClientBlock
from common.clusterblock import ClusterBlock

from utils.enums import LearningType, ClientState

from utils.mqtt_helper import MessageType, send_typed_message
from common.datablock import Datablock
from common import person_classifier
from flask_mqtt import Mqtt
from flask import Flask
from utils.model_helper import encode_state_dictionary
from common.aggregation_scheme import get_aggregation_scheme
from utils import constants
from utils.model_helper import decode_state_dictionary, encode_state_dictionary
from utils.mqtt_helper import MessageType, send_typed_message

import traceback

import json
import sys
import traceback
sys.path.append('.')

app = Flask(__name__)
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_REFRESH_TIME'] = 1.0  # refresh time in seconds
app.config['MQTT_KEEPALIVE'] = 1000
app.config['SECRET_KEY'] = 'secret!'
cors = CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
mqtt = Mqtt(app, mqtt_logging=True)

# global variables
PACKET_SIZE = 3000
CLIENTS = {}
CLIENT_DATABLOCKS = {}
CLIENT_NETWORKS = {}
NETWORK = None

CONFIGURATION = Configuration(LearningType.FEDERATED)

pinged_once = False
NUM_CLIENTS = 2
CLUSTERS = {}

# TODO: set global CONFIGURATION with GUI data, not programmer setting
@app.route('/')
def index():
    clusters = {
        "indoor": LearningType.CENTRALIZED,
        "outdoor": LearningType.FEDERATED
    }

    num_clients = 4

    initialize_server(clusters, num_clients)


    send_typed_message(
            mqtt,
            'server/general',
            constants.START_LEARNING_MESSAGE,
            MessageType.SIMPLE)


    return "server initialized and message sent"

    # global CLIENT_DATABLOCKS
    # global pinged_once
    #
    # if not pinged_once:
    #
    #     send_typed_message(
    #         mqtt,
    #         "server/network",
    #         {'message': constants.SEND_CLIENT_DATA},
    #         MessageType.SIMPLE)
    #     pinged_once = True
    #     return "Sent command to receive models.\n"
    # else:
    #     if CONFIGURATION == LearningType.CENTRALIZED:
    #         pbc = person_classifier.train(CLIENT_DATABLOCKS)
    #         encoded = encode_state_dictionary(pbc.model.state_dict())
    #         send_network_model(encoded)
    #         return 'Sent model to clients'




@socketio.on('connect')
def connection():
    print('websocket connect')
    socketio.emit("FromAPI", "test")


@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print('connected')
    mqtt.subscribe(constants.NEW_CLIENT_INITIALIZATION_TOPIC)


@mqtt.on_message()
def handle_mqtt_message(client, userdata, msg):
    global CLUSTERS, faults

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
        if CLIENTS[client_name].get_learning_type() == LearningType.FEDERATED:
            collect_federated_data(data, message, client_name)
        elif CLIENTS[client_name].get_learning_type() == LearningType.CENTRALIZED:
            collect_centralized_data(
                data, message, client_name, dimensions, label)
    else:
            print("Client not initialized correctly (client not in CLIENT_IDS)")
    # check if all clients finished sending data

    finished_clusters = get_completed_clusters()

    for cluster in finished_clusters:
        learning_type = CLUSTERS[cluster].get_learning_type()
        clients = CLUSTERS[cluster].get_clients()

        if learning_type == LearningType.CENTRALIZED:
            perform_centralized_learning(clients, cluster)
        elif learning_type == LearningType.FEDERATED:
            perform_federated_learning(clients, cluster)
        elif learning_type == LearningType.HYBRID:
            # this needs to be fixed, not sure if we're including this in final demo
            perform_hybrid_learning()


def initialize_server(required_clusters, num_clients):
    global CLUSTERS, CLIENTS

    reset()

    if num_clients % len(required_clusters) != 0:
        raise ValueError("Number of clients not evenly divisible by number of required cluster.")

    clients_per_cluster = num_clients / len(required_clusters)
    print("clients per cluster: {}".format(clients_per_cluster))

    for cluster_name in required_clusters:
        free_clients = get_free_clients(clients_per_cluster)
        CLUSTERS[cluster_name] = ClusterBlock(free_clients, 'cluster/'+cluster_name, required_clusters[cluster_name])

        if required_clusters[cluster_name] == LearningType.CENTRALIZED:
            learning_type = 'centralized'
        else:
            learning_type = 'federated'

        for client_id in free_clients:
            CLIENTS[client_id].set_learning_type(required_clusters[cluster_name])
            if required_clusters[cluster_name] == LearningType.CENTRALIZED:
                initialize_datablocks(client_id)


        # send msg to those clients saying this your cluster (for subscription)
        for client_id in free_clients:

            message = {
                'message': constants.SUBSCRIBE_TO_CLUSTER,
                constants.CLUSTER_TOPIC_NAME: CLUSTERS[cluster_name].get_mqtt_topic_name(),
                'learning_type': learning_type,
                'client_id': client_id
            }

            send_typed_message(
                mqtt,
                'server/general',
                message,
                MessageType.SIMPLE)


# grabs [num_required] free clients and sets status of clients to STALE
def get_free_clients(num_required):
    global CLIENTS

    free_client_ids = []

    for client in CLIENTS:
        if CLIENTS[client].get_state() == ClientState.FREE:
            free_client_ids.append(client)
            CLIENTS[client].set_state(ClientState.STALE)

        if len(free_client_ids) == num_required:
            return free_client_ids

    raise RuntimeError('Not enough available clients')


def reset():
    global CLIENT_NETWORKS, CLIENT_DATABLOCKS, CLUSTERS, CLIENTS

    CLIENT_NETWORKS.clear()
    CLIENT_DATABLOCKS.clear()
    CLUSTERS.clear()

    for client in CLIENTS:
        CLIENTS[client].set_state(ClientState.FREE)

    send_typed_message(
        mqtt,
        'server/general',
        constants.RESET_CLIENT_MESSAGE,
        MessageType.SIMPLE)


# takes the clients that need to be aggregated as input and sends the model
def perform_federated_learning(clients, cluster):
    global CLUSTERS, CLIENTS, CLIENT_NETWORKS

    print("averaging for cluster: {}".format(cluster))

    averaged_state_dict = get_aggregation_scheme(clients, CLIENT_NETWORKS)
    CLUSTERS[cluster].set_state_dict(averaged_state_dict)

    for client in clients:
        CLIENT_NETWORKS[client].reset_network_data()
        CLIENTS[client].set_state(ClientState.STALE)

    send_network_model(encode_state_dictionary(averaged_state_dict), CLUSTERS[cluster].get_mqtt_topic_name())


def perform_centralized_learning(clients, cluster):
    global CLIENTS, CLIENT_DATABLOCKS, CLUSTERS

    applicable_client_datablocks = {k: v for (k, v) in CLIENT_DATABLOCKS.items() if k in clients}

    runner = person_classifier.get_model_runner(client_data=applicable_client_datablocks, num_epochs=1)

    if CLUSTERS[cluster].get_state_dict() is not None:
        runner.model.load_state_dictionary(CLUSTERS[cluster].get_state_dict())

    runner.train_model()

    CLUSTERS[cluster].set_state_dict(runner.model.get_state_dictionary())

    for client_id in clients:
        CLIENTS[client_id].set_state(ClientState.STALE)

    send_network_model(encode_state_dictionary(runner.model.get_state_dictionary()), CLUSTERS[cluster].get_mqtt_topic_name())


def perform_hybrid_learning():
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

        send_network_model(encoded) # ======== BROKEN =========
        for client in CLIENTS:
            CLIENTS[client].set_state(ClientState.STALE)

    except Exception as e:
        print(traceback.format_exc())


def get_completed_clusters():
    finished_clusters = []

    for cluster in CLUSTERS:
        complete = True
        for client_id in CLUSTERS[cluster].get_clients():
            if CLIENTS[client_id].get_state() != ClientState.FINISHED:
                complete = False
                break
        if complete:
            finished_clusters.append(cluster)

    return finished_clusters


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

        CLIENTS[client_id].set_state(ClientState.FINISHED)


def collect_centralized_data(data, message, client_name, dimensions, label):
    global CLIENTS
    if message == constants.DEFAULT_IMAGE_INIT:
        initialize_new_image(client_name, dimensions, label)
    elif message == constants.DEFAULT_IMAGE_CHUNK:
        add_data_chunk(client_name, data)
    elif message == constants.DEFAULT_IMAGE_END:
        convert_data(client_name)
    elif message == 'all_images_sent':
        CLIENTS[client_name].set_state(ClientState.FINISHED)
        print("All images received from client: {}".format(client_name))


def initialize_new_clients(client_id):
    print("New client connected: {}".format(client_id))

    CLIENTS[client_id] = ClientBlock(ClientState.FREE)

    # CLIENTS[client_id] = ClientBlock(LearningType.FEDERATED, ClientState.Stale)
    #
    # add_client_to_cluster(client_id)
    #
    # if CLIENTS[client_id].get_learning_type() == LearningType.CENTRALIZED:
    #     initialize_datablocks(client_id)

    mqtt.subscribe('client/' + client_id)


def add_client_to_cluster(client_id):
    global CLUSTERS

    for cluster in CLUSTERS.keys():
        if len(CLUSTERS[cluster].get_clients()) < (NUM_CLIENTS / len(CLUSTERS.keys())):
            CLUSTERS[cluster].get_clients().append(client_id)
            return

    print("ERROR: Could not find an empty cluster.")



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


def send_network_model(payload, topic):
    send_typed_message(
        mqtt,
        topic,
        payload,
        MessageType.NETWORK_CHUNK)


def initialize_datablocks(client):
    global CLIENT_DATABLOCKS
    CLIENT_DATABLOCKS[client] = Datablock()


if __name__ == '__main__':
    socketio.run(app, port=5000, host='0.0.0.0')
