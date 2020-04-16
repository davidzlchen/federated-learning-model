from enum import Enum


class LearningType(Enum):
    NONE = 0
    CENTRALIZED = 1
    FEDERATED = 2
    HYBRID = 3
    PERSONALIZED = 4


class ClientState(Enum):
    STALE = 1
    SENDING = 2
    FINISHED = 3
    FREE = 4


class AggregationScheme(Enum):
    AVERAGE = 1
    WEIGHTED_AVERAGE = 2
