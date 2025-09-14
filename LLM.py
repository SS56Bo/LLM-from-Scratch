import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GPTDataLoader(Dataset):
    def __init__(self, input: list[int], max_length, stride):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(input) - max_length, stride):
            input_chunk = input[i:i+max_length]
            target_chunk = input[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

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
    

class EmbedText(nn.Module):
    def __init__(self, vocab_size, dimen, context_size=4, device=device):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.output_dimension = dimen
        
        self.embed_model = nn.Embedding(vocab_size, dimen).to(device)
        self.pos_embed_model = nn.Embedding(context_size, dimen).to(device)

    def convert_tokens_to_embeddings(self, input_text):
        # ensure batch is on correct device
        input_text = input_text.to(self.device)

        # token embeddings
        token_embeddings = self.embed_model(input_text)

        # positional embeddings
        positions = torch.arange(input_text.shape[-1], device=self.device)
        position_embeddings = self.pos_embed_model(positions)

        # broadcast + add
        embeddings = token_embeddings + position_embeddings
        return embeddings
