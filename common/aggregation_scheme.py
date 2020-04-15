import torch
from utils.enums import AggregationScheme, LearningType
from common.models import PersonBinaryClassifier
import copy

DEFAULT_AGGREGATION_SCHEME = AggregationScheme.AVERAGE


def get_aggregation_scheme(CLIENTS, CLIENT_NETWORKS):
    if DEFAULT_AGGREGATION_SCHEME == AggregationScheme.AVERAGE:
        print("averaging")
        return get_average(CLIENTS, CLIENT_NETWORKS)


def get_average(clients, client_networks):
    clients_iterator = iter(clients.keys())

    # get first network to average
    num_federated_clients = 0
    while True: 
        try:
            client_id = next(clients_iterator)
            if clients[client_id].get_learning_type() == LearningType.FEDERATED:
                averaged_state_dict = copy.deepcopy(client_networks[client_id].state_dict)
                num_federated_clients += 1
                break
        except StopIteration:
            print("No federated clients to average.")
            return None

    temp_model = PersonBinaryClassifier()
    temp_model.load_state_dictionary(averaged_state_dict)

    # average rest of the networks
    while True:
        try:
            client_id = next(clients_iterator)
            if clients[client_id].get_learning_type() == LearningType.FEDERATED:
                client_state_dict = client_networks[client_id].state_dict
                client_model = PersonBinaryClassifier()
                client_model.load_state_dictionary(client_state_dict)

                model_params = client_model.model.named_parameters()
                for param_name, param_value in model_params:
                    if param_name in averaged_state_dict:
                        averaged_state_dict[param_name].add_(client_state_dict[param_name])

                num_federated_clients += 1
        except StopIteration:
            break

    averaged_model = PersonBinaryClassifier()
    averaged_model.load_state_dictionary(averaged_state_dict)

    for param_name, param_value in averaged_model.model.named_parameters():
        averaged_state_dict[param_name] = torch.div(averaged_state_dict[param_name], num_federated_clients)

    return averaged_state_dict
