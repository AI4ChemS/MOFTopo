from os import listdir
from os.path import join, isfile

import torch
from torch.utils.data import Dataset

from MOFTopo.common.data_utils import list_file_paths_in_dir


class ZeoliteDataset(Dataset):
    def __init__(self, processed_dir):
        """
        Assumes that crystals have already been parsed. See MOFTopo/common/data_utils.py for tools to parse data
        """
        self.processed_dir = processed_dir
        self.processed_files = list_file_paths_in_dir(self.processed_dir)
    
    def __getitem__(self, index,):
        return torch.load(self.processed_files[index])
    
    def __len__(self,):
        return len(self.processed_files)

