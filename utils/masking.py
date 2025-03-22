"""
Utility functions for creating attention masks.
"""
import torch

def create_padding_mask(seq):
    """
    Create padding mask for attention.
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len)
        
    Returns:
        Padding mask of shape (batch_size, 1, 1, seq_len)
    """
    # 1 for padding token positions, 0 for non-padding
    mask = (seq == 0).float()
    
    # Add extra dimensions for broadcasting to attention logits
    return mask.unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    """
    Create look-ahead mask for decoder self-attention.
    
    Args:
        size: Size of the square mask
        
    Returns:
        Look-ahead mask of shape (1, size, size)
    """
    # Create upper triangular matrix with 1s
    # This ensures that position i cannot attend to positions j > i
    mask = torch.triu(torch.ones(size, size), diagonal=1).float()
    
    # Convert to binary mask (1 for positions to mask, 0 for valid positions)
    return mask.unsqueeze(0)

def create_combined_mask(seq):
    """
    Create combined padding and look-ahead mask for decoder self-attention.
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len)
        
    Returns:
        Combined mask of shape (batch_size, 1, seq_len, seq_len)
    """
    seq_len = seq.size(1)
    padding_mask = create_padding_mask(seq)  # (batch_size, 1, 1, seq_len)
    look_ahead_mask = create_look_ahead_mask(seq_len)  # (1, seq_len, seq_len)
    
    # Expand padding mask for broadcasting
    padding_mask = padding_mask.expand(-1, -1, seq_len, -1)
    
    # Combine masks: use max operation as we represent masks with 1s for masked positions
    combined_mask = torch.max(padding_mask, look_ahead_mask)
    
    return combined_mask 