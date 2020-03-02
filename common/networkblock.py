from utils.model_helper import get_state_dictionary
import utils.constants as constants


class Networkblock(object):
    def __init__(self):
        self.network_string = ""
        self.network_status = constants.NETWORK_STALE
        self.state_dict = None

    def add_network_chunk(self, chunk):
        self.network_string += chunk

    def reset_network_data(self):
        self.network_string = ""
        self.network_status = constants.NETWORK_STALE

    def reconstruct_state_dict(self):
        self.network_status = constants.NETWORK_NEW
        self.state_dict = get_state_dictionary(network_string=self.network_string)
        return self.state_dict
