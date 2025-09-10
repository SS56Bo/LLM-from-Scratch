import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataLoader:
    def __init__(self, input: list[int], max_length, stride):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(input)-max_length, stride):
            input_chunk = input[i:i+max_length]
            target_chunk = input[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    @staticmethod
    def create_dataloader(text: list[int], batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, workers=0):
        data = GPTDataLoader(text, max_length, stride)
        dataload = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=workers
        )

        return dataload