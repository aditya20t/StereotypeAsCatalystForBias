# Import libraries
import torch
from torch.utils.data import DataLoader

# Define the function to create DataLoaders
def get_dataloader(dataset, batch_size, collate_fn=None):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )