import base64
import json
import person_classifier

from datablock import Datablock
from flask import Flask
from flask_mqtt import Mqtt
from utils.mqtt_helper import MessageType, send_typed_message

app = Flask(__name__)
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_REFRESH_TIME'] = 1.0  # refresh time in seconds
mqtt = Mqtt(app)

# global variables
PACKET_SIZE = 3000
CLIENT_IDS = set(["pi01"])
CLIENT_DATABLOCKS = {}

@app.route('/')
def index():
    state_dict = person_classifier.train(CLIENT_DATABLOCKS)
    #model = open('./network.pth', 'rb').read()
    send_network_model(state_dict)
    return "Successfully trained model and sent to subscribed clients."

@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    for c_id in CLIENT_IDS:
        mqtt.subscribe('client/' + c_id)

@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    client_name = message.topic.split("/")[1]

    payload = json.loads(message.payload.decode())
    dimensions = payload["dimensions"]
    label = payload["label"]
    data = payload["data"]

    if client_name in CLIENT_IDS:
        if payload["message"] == "sending_data":
            initialize_new_image(client_name, dimensions, label)
        elif payload["message"] == "chunk":
            add_data_chunk(client_name, data)
        elif payload["message"] == "done":
            convert_data(client_name)

def initialize_new_image(client_name, dimensions, label):
    datablock = CLIENT_DATABLOCKS[client_name]
    datablock = datablock.init_new_image(dimensions, label)
    CLIENT_DATABLOCKS[client_name] = datablock

def add_data_chunk(client_name, chunk):
    datablock = CLIENT_DATABLOCKS[client_name]
    datablock.add_image_chunk(chunk)

def convert_data(client_name):
    datablock = CLIENT_DATABLOCKS[client_name]
    datablock.convert_current_image_to_matrix()

def send_network_model(payload):
    send_typed_message(mqtt, "server/network", payload, MessageType.NETWORK_CHUNK)

def initialize_datablocks():
    for client in CLIENT_IDS:
        CLIENT_DATABLOCKS[client] = Datablock()

if __name__ == '__main__':
    initialize_datablocks()
    app.run(host='localhost', port=5000)
