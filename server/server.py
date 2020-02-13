from flask import Flask
from flask_mqtt import Mqtt
import json

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
    mqtt.subscribe('data')

pictures = {}
def reconstructBase64String(chunk):
    global pictures
    pChunk = json.loads(chunk["payload"])
    #pChunk = JSON.parse(chunk["d"])
    print("check1")

    #creates a new picture object if receiving a new picture, else adds incoming strings to an existing picture
    if (pictures[pChunk["pic_id"]]==null):
        pictures[pChunk["pic_id"]] = {"count":0, "total":pChunk["size"], pieces: {}, "pic_id": pChunk["pic_id"]}

        pictures[pChunk["pic_id"]].pieces[pChunk["pos"]] = pChunk["data"]
        print("check2")
    else:
        pictures[pChunk["pic_id"]].pieces[pChunk["pos"]] = pChunk["data"]
        pictures[pChunk["pic_id"]].count += 1
        print("check3")
        if (pictures[pChunk["pic_id"]].count == pictures[pChunk["pic_id"]].total):
            print("Image reception compelete")
            str_image=""

        i=0
        while(i <= pictures[pChunk["pic_id"]].total):
            str_image = str_image + pictures[pChunk["pic_id"]].pieces[i]

            #displays image
            '''
            source = 'data:image/jpeg;base64,'+str_image
            myImageElement = document.getElementById("picture_to_show")
            myImageElement.href = source
            '''
            i+=1
            print(str_image)
def hello():
    print("hello")

@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    #print("handling message")
    data = dict(
        topic=message.topic,
        payload=message.payload.decode()
    )
    #print(data)
    reconstructBase64String(data)





