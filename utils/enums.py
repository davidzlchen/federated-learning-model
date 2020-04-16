from enum import Enum


class LearningType(Enum):
    NONE = -1
    CENTRALIZED = 0
    FEDERATED = 1
    HYBRID = 2
    PERSONALIZED = 3


class ClientState(Enum):
    STALE = 1
    SENDING = 2
    FINISHED = 3
    FREE = 4


class AggregationScheme(Enum):
    AVERAGE = 1
    WEIGHTED_AVERAGE = 2
