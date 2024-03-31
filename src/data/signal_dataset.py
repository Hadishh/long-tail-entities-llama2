import torch
from torch.utils.data import Dataset
import os
import json

class SignalDataset(Dataset):
    def __init__(self, signal_dir, semantics_dir=None):
        self.dir = signal_dir
        self.data_files = list()
        self.id2idx = {}
        self.semantic_files = list()
        i = 0
        for subdir, dir, files in os.walk(self.dir):
            for file in files:
                self.id2idx[file] = i
                i += 1
                filepath = os.path.join(subdir, file)
                self.data_files.append(filepath)
                if semantics_dir:
                    self.semantic_files.append(os.path.join(semantics_dir, file))
        
    def __getitem__(self, index):
        with open(self.data_files[index], "r", encoding="utf-8") as f:
            instance = json.load(f)
        
        if self.semantic_files:
            with open(self.semantic_files[index], "r") as f:
                semantic_data = json.load(f)
            
            instance["semantic_relations"] = semantic_data 
        return instance

    def get_by_id(self, id):
        return self[self.id2idx[id]]
    
    def __len__(self):
        return len(self.data_files)

