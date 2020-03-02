import base64
from io import BytesIO
import torch
from common.models import get_default_model

USE_LOCAL_NETWORK_CHECKPOINT = False
LOCAL_NETWORK_CHECKPOINT_PATH = './network.pth'


def get_state_dictionary(
        network_string='',
        path=LOCAL_NETWORK_CHECKPOINT_PATH):
    if USE_LOCAL_NETWORK_CHECKPOINT:
        checkpoint = torch.load(path)
    else:
        network_decoded = torch.load(BytesIO(base64.decodebytes(network_string.encode())))
        model = get_default_model()
        model.load_last_layer_state_dictionary(network_decoded)

    return model
