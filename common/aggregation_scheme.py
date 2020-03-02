from utils import constants
import numpy as np
from enum import Enum

class AggregationScheme(Enum):
    AVERAGE=1
    WEIGHTED_AVERAGE=2 

DEFAULT_AGGREGATION_SCHEME = AggregationScheme.AVERAGE


def get_aggregation_scheme(server_model, CLIENT_IDS, CLIENT_NETWORKS):
    if DEFAULT_AGGREGATION_SCHEME == AggregationScheme.AVERAGE:
        return AggregationAverage(server_model, CLIENT_IDS, CLIENT_NETWORKS)


class AggregationAverage():

    def __init__(self, server_model, CLIENT_IDS, CLIENT_NETWORKS):
        self.server_model = server_model
        self.client_ids = CLIENT_IDS
        self.client_networks = CLIENT_NETWORKS

    def get_average(self):
        fc_state_dict = self.server_model.model.fc.state_dict()

        weights = np.zeros(fc_state_dict['weight'].size())
        bias = np.zeros(fc_state_dict['bias'].size())

        for client_id in self.client_ids:

            fc_state_dict = self.client_networks[client_id].network.model.fc.state_dict(
            )
            weights = np.add(weights, fc_state_dict['weight'])
            bias = np.add(bias, fc_state_dict['bias'])

        num_clients = len(self.client_ids)

        weights = np.divide(weights, num_clients)
        bias = np.divide(bias, num_clients)

        return (weights, bias)
