from flask import Flask
from flask_mqtt import Mqtt
import json
import numpy as np

from PIL import Image
import PIL
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

pictures = {}

photo = ""


@app.route('/')
def index():
    network.run()
    return "training model"

@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    mqtt.subscribe('client-to-server')




def reconstructBase64String(chunk):
    global pictures
    global photo
    pChunk = json.loads(chunk["payload"])
    #pChunk = JSON.parse(chunk["d"])

    photo += pChunk["data"]

    # #creates a new picture object if receiving a new picture, else adds incoming strings to an existing picture
    # if (pictures[pChunk["pic_id"]] is None):
    #
    #     pictures[pChunk["pic_id"]] = {"count":0, "total":pChunk["size"], "pieces": {}, "pic_id": pChunk["pic_id"]}
    #
    #     pictures[pChunk["pic_id"]]["pieces"][pChunk["pos"]] = pChunk["data"]
    #
    # else:
    #     pictures[pChunk["pic_id"]].pieces[pChunk["pos"]] = pChunk["data"]
    #     pictures[pChunk["pic_id"]].count += 1
    #     print("check3")
    #     if (pictures[pChunk["pic_id"]].count == pictures[pChunk["pic_id"]].total):
    #         print("Image reception compelete")
    #         str_image=""
    #
    #     i=0
    #     while(i <= pictures[pChunk["pic_id"]].total):
    #         str_image = str_image + pictures[pChunk["pic_id"]].pieces[i]
    #
    #         #displays image
    #         '''
    #         source = 'data:image/jpeg;base64,'+str_image
    #         myImageElement = document.getElementById("picture_to_show")
    #         myImageElement.href = source
    #         '''
    #         i+=1
    #     print(str_image)

@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    print("message: ", message.payload.decode())
    data = dict(
        topic=message.topic,
        payload=message.payload.decode()
    )
    print(data)

    if message.topic == "client-to-server":

        if data["payload"] == "done":
            display_image(photo)
            print("reconstruct done")
        else:
            reconstructBase64String(data)



def display_image(image_data):
    print("display_image reached")
    print(image_data, "image_data")
    image_bytes = image_data.encode()
    print(image_bytes, "image bytes")
    r = base64.b64decode(image_bytes)
    print(r, 'r')
    with open('./my_pickle.pkl', 'wb') as f:
        f.write(r)

    #new_img = Image.fromarray(np.reshape(q, (427, 640, 3)))
    #new_img.save("image.jpg")



if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)


