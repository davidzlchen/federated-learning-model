import base64
import json
import utils.constants as constants

from flask import Flask
from flask_mqtt import Mqtt
from common import person_classifier
from common.datablock import Datablock
from utils.mqtt_helper import MessageType, send_typed_message

app = Flask(__name__)
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_REFRESH_TIME'] = 1.0  # refresh time in seconds
mqtt = Mqtt(app)

# global variables
PACKET_SIZE = 3000
CLIENT_IDS = set()
CLIENT_DATABLOCKS = {}


@app.route('/')
def index():
    global CLIENT_DATABLOCKS

    model = person_classifier.train(CLIENT_DATABLOCKS)
    model.save('./network.pth')
    state_dict = open('./network.pth', 'rb').read()
    send_network_model(state_dict)
    return "Successfully trained model and sent to subscribed clients."


@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print("connected")
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
    if client_name in CLIENT_IDS:
        if message == constants.DEFAULT_IMAGE_INIT:
            initialize_new_image(client_name, dimensions, label)
        elif message == constants.DEFAULT_IMAGE_CHUNK:
            add_data_chunk(client_name, data)
        elif message == constants.DEFAULT_IMAGE_END:
            convert_data(client_name)


def initialize_new_clients(client_id):
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
