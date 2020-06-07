# The Code written by Ali Babolhaveji @ 6/1/2020

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import ipyvolume as ipv
from boto3.session import Session
import os
import yaml


class ClogLossDataset_from_compressed_data(Dataset):
    def __init__(self, config, split='train', type='train'):
        self.cfg = config
        self.dataPath = config['dataset']['path']
        
        self.flowing_Tensors = os.path.join(config['dataset']['path'], 'flowing_Tensors')
        self.stall_Tensors = os.path.join(config['dataset']['path'], 'stall_Tensors')
        
        





        return tensor_img


# train_Dataset = ClogLossDataset(config)