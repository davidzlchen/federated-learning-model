from utils.enums import LearningType

class ClusterBlock(object):
    def __init__(self, clients, mqtt_topic_name, learning_type: LearningType, state_dict=None):
        self.clients = list(clients)
        self.mqtt_topic_name = mqtt_topic_name
        self.state_dict = state_dict
        self.learning_type = learning_type

    def get_clients(self):
        return self.clients

    def get_state_dict(self):
        return self.state_dict

    def set_state_dict(self, state_dict):
        self.state_dict = state_dict

    def get_mqtt_topic_name(self):
        return self.mqtt_topic_name

    def get_learning_type(self):
        return self.learning_type
