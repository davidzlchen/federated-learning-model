from utils import constants
import torch
import numpy as np
from enum import Enum
from common.models import PersonBinaryClassifier
import copy
import traceback


class AggregationScheme(Enum):
    AVERAGE = 1
    WEIGHTED_AVERAGE = 2


DEFAULT_AGGREGATION_SCHEME = AggregationScheme.AVERAGE


def get_aggregation_scheme(CLIENT_IDS, CLIENT_NETWORKS):
    if DEFAULT_AGGREGATION_SCHEME == AggregationScheme.AVERAGE:
        print("averaging")
        return get_average(CLIENT_IDS, CLIENT_NETWORKS)


def get_average(CLIENT_IDS, CLIENT_NETWORKS):
    # beta = 0.5  # The interpolation parameter
    # params1 = model1.named_parameters()
    # params2 = model2.named_parameters()
    #
    # dict_params2 = dict(params2)
    #
    # for name1, param1 in params1:
    #     if name1 in dict_params2:
    #         dict_params2[name1].data.copy_(beta * param1.data + (1 - beta) * dict_params2[name1].data)
    #
    # model.load_state_dict(dict_params2)
    clients = iter(CLIENT_IDS)

    averaged_state_dict = copy.deepcopy(CLIENT_NETWORKS[next(clients)].state_dict)

    temp_model = PersonBinaryClassifier()
    temp_model.load_state_dictionary(averaged_state_dict)

    for name, param in temp_model.model.named_parameters():
        print('name: ', name)
        print(param)
        print('=====')
        break

    while True:
        try:
            client_state_dict = CLIENT_NETWORKS[next(clients)].state_dict
            client_model = PersonBinaryClassifier()
            client_model.load_state_dictionary(client_state_dict)

            model_params = client_model.model.named_parameters()
            for param_name, param_value in model_params:
                if param_name in averaged_state_dict:
                    averaged_state_dict[param_name].add_(client_state_dict[param_name])

        except StopIteration:
            break

    averaged_model = PersonBinaryClassifier()
    averaged_model.load_state_dictionary(averaged_state_dict)

    num_clients = len(CLIENT_IDS)
    for param_name, param_value in averaged_model.model.named_parameters():
        averaged_state_dict[param_name] = torch.div(averaged_state_dict[param_name], num_clients)

    return averaged_state_dict
