from enum import Enum


class LearningType(Enum):
    CENTRALIZED = 1
    FEDERATED = 2


class Configuration(object):
    def __init__(self, learning_type=LearningType.CENTRALIZED):
        self.learning_type = learning_type
