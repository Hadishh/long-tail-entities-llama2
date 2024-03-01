import torch
from torch.utils.data import Dataset
import os
import json

class SignalDataset(Dataset):
    def __init__(self, signal_dir):
        self.dir = signal_dir
        self.data_files = list()
        for subdir, dir, files in os.walk(self.dir):
            for file in files:
                filepath = os.path.join(subdir, file)
                self.data_files.append(filepath)
        
    def __getitem__(self, index):
        with open(self.data_files[index], "r") as f:
            instance = json.load(f)
        
        return instance

    def __len__(self):
        return len(self.data_files)

