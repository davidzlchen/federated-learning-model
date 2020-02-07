from flask import Flask
from flask_mqtt import Mqtt

import pandas as pd
import pickle
import torch

app = Flask(__name__)
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
#app.config['MQTT_USERNAME'] = 'user'
#app.config['MQTT_PASSWORD'] = 'secret'
app.config['MQTT_REFRESH_TIME'] = 1.0  # refresh time in seconds
mqtt = Mqtt(app)




@app.route('/')
def index():
    return render_template('index.html')

@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    mqtt.subscribe('central/getdata')
    

@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    data = dict(
        topic=message.topic,
        payload=message.payload.decode()
    )


    if data.topic == 'central/getdata':
        #convert pickle
        new_data = pd.read_pickle(data.payload)

        #update model



        #pickle the model

        #send model back down



    print(data)


if __name__ == '__main__':
    app.run(host='localhost', port=5000)