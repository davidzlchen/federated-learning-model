import base64
import pickle

def decode_state_dictionary(
        network_string=''
    ):
    checkpoint = pickle.loads(
        base64.decodebytes(
            network_string.encode()))

    return checkpoint

def encode_state_dictionary(
        state_dict
    ):
    return base64.encodebytes(pickle.dumps(state_dict)).decode('utf-8')
