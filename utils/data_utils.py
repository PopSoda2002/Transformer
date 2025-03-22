"""
Utility functions for data loading and preprocessing.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json

class TransformerDataset(Dataset):
    """
    Dataset for Transformer model.
    """
    def __init__(self, source_data, target_data):
        """
        Args:
            source_data: Source sequences as a list of tensors
            target_data: Target sequences as a list of tensors
        """
        assert len(source_data) == len(target_data), "Source and target must have the same length"
        self.source_data = source_data
        self.target_data = target_data
        
    def __len__(self):
        return len(self.source_data)
    
    def __getitem__(self, idx):
        """
        Return a pair of source and target sequences.
        """
        return self.source_data[idx], self.target_data[idx]

def load_data(data_path, split='train'):
    """
    Load dataset from file.
    
    Args:
        data_path: Path to data directory
        split: Data split to load ('train', 'valid', or 'test')
        
    Returns:
        Dictionary containing source and target data
    """
    file_path = os.path.join(data_path, f'{split}.json')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def process_data(data, tokenizer, max_seq_length):
    """
    Process raw data into tensor format.
    
    Args:
        data: Dictionary containing source and target data
        tokenizer: Tokenizer object for encoding text
        max_seq_length: Maximum sequence length
        
    Returns:
        Processed source and target data as lists of tensors
    """
    source_data = []
    target_data = []
    
    for source, target in zip(data['source'], data['target']):
        # Tokenize and convert to tensor
        source_ids = tokenizer.encode(source, max_length=max_seq_length, pad_to_max_length=True)
        target_ids = tokenizer.encode(target, max_length=max_seq_length, pad_to_max_length=True)
        
        # Convert to PyTorch tensors
        source_tensor = torch.tensor(source_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        
        source_data.append(source_tensor)
        target_data.append(target_tensor)
    
    return source_data, target_data

def create_data_loader(source_data, target_data, batch_size, shuffle=True):
    """
    Create a DataLoader for model training/evaluation.
    
    Args:
        source_data: List of source tensors
        target_data: List of target tensors
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader object
    """
    dataset = TransformerDataset(source_data, target_data)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=False
    )

def create_dummy_data(batch_size, src_seq_len, tgt_seq_len, vocab_size):
    """
    Create dummy data for testing.
    
    Args:
        batch_size: Number of sequences in the batch
        src_seq_len: Source sequence length
        tgt_seq_len: Target sequence length
        vocab_size: Size of the vocabulary
        
    Returns:
        Dummy source and target tensors
    """
    # Create random source and target sequences
    src = torch.randint(1, vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_seq_len))
    
    # Add padding to some positions to test masking
    src[:, -2:] = 0  # Pad last two positions of source
    tgt[:, -2:] = 0  # Pad last two positions of target
    
    return src, tgt 