import time
import paho.mqtt.client as mqtt
import pickle
import json

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("model")

    person_pkl = open('./files/COCO/personimages.pkl', 'rb')
    person_matrix = pickle.load(person_pkl)
    person_matrix = json.dumps(person_matrix[:1])
    client.publish("data", "hello world")
    client.publish("data", person_matrix)
    client.publish("data", "hello world")
    
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

def on_publish(client, userdata, result):
    print("data published")

client = mqtt.Client()
client.on_publish = on_publish
client.on_connect = on_connect
client.on_message = on_message

client.connect("localhost", 1883, 65534)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_forever()

