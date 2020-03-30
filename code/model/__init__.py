
from .model_define import *

model_dict = \
{
    'Unet' : Unet.net,
    'Unet-SE' : SEUnet.net,
    'R2Unet':R2Unet.net,
    'Attention-Unet':AttentionUnet.net,
    'Unet++':Unetpp.net,
    'CEnet':CEnet.net,
    'Wnet':Wnet.net,
    'Wnet_raw':Wnet_raw.net,
    'Attention-Unet-SE':AttentionSEUnet.net,
    'Attention-Wnet':AttentionWnet.net,
    'SE-Wnet':SEWnet.net,
    'Attention-Unet-SE':AttentionSEUnet.net
}