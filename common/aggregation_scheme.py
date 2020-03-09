from utils import constants
import numpy as np
from enum import Enum


class AggregationScheme(Enum):
    AVERAGE = 1
    WEIGHTED_AVERAGE = 2


DEFAULT_AGGREGATION_SCHEME = AggregationScheme.AVERAGE


def get_aggregation_scheme(server_model, CLIENT_IDS, CLIENT_NETWORKS):
    if DEFAULT_AGGREGATION_SCHEME == AggregationScheme.AVERAGE:
        print("averaging")
        return get_average(server_model, CLIENT_IDS, CLIENT_NETWORKS)


def get_average(server_model, CLIENT_IDS, CLIENT_NETWORKS):
    fc_state_dict = server_model.model.fc.state_dict()

    weights = np.zeros(fc_state_dict['weight'].size())
    bias = np.zeros(fc_state_dict['bias'].size())

    for client_id in CLIENT_IDS:

        fc_state_dict = CLIENT_NETWORKS[client_id].state_dict
        weights = np.add(weights, fc_state_dict['weight'])
        bias = np.add(bias, fc_state_dict['bias'])

    num_clients = len(CLIENT_IDS)

    weights = np.divide(weights, num_clients)
    bias = np.divide(bias, num_clients)

    return (weights, bias)
