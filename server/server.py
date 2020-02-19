from flask import Flask
from flask_mqtt import Mqtt
import json
import numpy as np

from PIL import Image
from io import BytesIO
import base64

import pandas as pd
import pickle
import network

app = Flask(__name__)
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
#app.config['MQTT_USERNAME'] = 'user'
#app.config['MQTT_PASSWORD'] = 'secret'
app.config['MQTT_REFRESH_TIME'] = 1.0  # refresh time in seconds
mqtt = Mqtt(app)


PACKET_SIZE = 3000


pictures = {}
clientIds = set(["pi01"])
clientDataBlock = {}


def saveImages():
    counter = 0


    for client in clientDataBlock:
        images = clientDataBlock[client]["imageData"]

        for i in range(0, len(images)):
            img = Image.fromarray(images[i])
            path = "./pictures/img" + str(counter) + ".jpg"
            img.save(path)

            counter += 1

def send_network_model(payload):
    encoded = base64.b64encode(payload)

    end = PACKET_SIZE
    start = 0

    length = len(encoded)
    num_packets = np.ceil(length/PACKET_SIZE)

    mqtt.publish("server/network", json.dumps({"message": "sending_data"}))

    while start <= length:

        data = {"message" : "network_chunk", "data": encoded[start:end].decode('utf-8')}

        data_packet = json.dumps(data)

        mqtt.publish('server/network', data_packet)


        end += PACKET_SIZE
        start = PACKET_SIZE

    mqtt.publish("server/network", json.dumps({"message": "end_transmission"}))


@app.route('/')
def index():
    # network.run(clientDataBlock)
    # send_network_model(network)

    saveImages()

    return "training model"

@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):

    for client in clientIds:
        
        mqtt.subscribe('client/' + client)




def add_data_chunk(clientName, chunk):

    global clientDataBlock

    currentImage = clientDataBlock[clientName]["currentImage"]

    clientDataBlock[clientName]["imageData"][currentImage] = clientDataBlock[clientName]["imageData"][currentImage] + chunk


@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    global clientDataBlock


    payload = json.loads(message.payload.decode())

    clientName = message.topic.split("/")[1]

    if clientName in clientIds:
        if payload["message"] == "sending_data":
            clientDataBlock[clientName]["numImages"] += 1
            clientDataBlock[clientName]["currentImage"] += 1
            clientDataBlock[clientName]["imageData"].append("")
            clientDataBlock[clientName]["dimensions"].append(payload["dimensions"])
            clientDataBlock[clientName]["labels"].append(payload["label"])

        elif payload["message"] == "done":
            convert_data(clientName)
        elif payload["message"] == "chunk":
            add_data_chunk(clientName, payload["data"])


def convert_data(clientName):


    currentImage = clientDataBlock[clientName]["currentImage"]

    dims = clientDataBlock[clientName]["dimensions"][currentImage]

    image_base64 = clientDataBlock[clientName]["imageData"][currentImage].encode()

    img = base64.decodebytes(image_base64)

    buf = np.frombuffer(img, dtype=np.uint8)

    buf = np.reshape(buf, dims)

    clientDataBlock[clientName]["imageData"][currentImage] = buf


def initialize():
    global clientDataBlock

    for client in clientIds:
        clientDataBlock[client] = {
            "numImages" : 0,
            "currentImage": -1,
            "imageData" : [],
            "dimensions": [],
            "labels": [] 
        }



if __name__ == '__main__':
    initialize()
    app.run(host='localhost', port=5000)


