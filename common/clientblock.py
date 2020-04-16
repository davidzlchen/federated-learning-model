from utils.enums import LearningType, ClientState


class ClientBlock(object):
    def __init__(self, state: ClientState, learning_type: LearningType = None):
        self.learning_type = learning_type
        self.state = state
        self.model_accuracy = None
        self.communication_cost = None

    def get_learning_type(self):
        return self.learning_type

    def set_learning_type(self, learning_type: LearningType):
        self.learning_type = learning_type

    def get_state(self):
        return self.state

    def set_state(self, state: ClientState):
        self.state = state
