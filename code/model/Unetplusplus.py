
from model_block import *

def Unet_nested(input, num_class, initial_channel=64, rate=0.5):
    """
    The Unet++ network architecture
    Args:
        input:the network input.
        initial_channel:the network channel benchmark of each layer.
        rate:the middle dropout layer keep probability.
    Return:
        out:the network output.
    """
    