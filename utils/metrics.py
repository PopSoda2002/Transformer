"""
Utility functions for evaluating model performance.
"""
import torch
import numpy as np
from collections import Counter

def accuracy(predictions, targets, pad_idx=0):
    """
    Calculate accuracy ignoring padding tokens.
    
    Args:
        predictions: Model predictions of shape (batch_size, seq_len, vocab_size)
        targets: Target indices of shape (batch_size, seq_len)
        pad_idx: Padding token index
        
    Returns:
        Accuracy value
    """
    with torch.no_grad():
        # Get the predicted token indices
        pred_tokens = predictions.argmax(dim=-1)
        
        # Create mask to ignore padding tokens
        mask = (targets != pad_idx)
        
        # Compare predictions with targets and apply mask
        correct = (pred_tokens == targets) & mask
        
        # Calculate accuracy
        accuracy = correct.float().sum() / mask.float().sum()
        
        return accuracy.item()

def bleu_score(predictions, targets, max_n=4, pad_idx=0):
    """
    Calculate BLEU score for machine translation evaluation.
    This is a simplified BLEU implementation.
    
    Args:
        predictions: Predicted token indices of shape (batch_size, seq_len)
        targets: Target token indices of shape (batch_size, seq_len)
        max_n: Maximum n-gram size to consider
        pad_idx: Padding token index
        
    Returns:
        BLEU score value
    """
    batch_size = predictions.size(0)
    scores = []
    
    for i in range(batch_size):
        # Extract single sequences and remove padding
        pred_seq = [token.item() for token in predictions[i] if token.item() != pad_idx]
        target_seq = [token.item() for token in targets[i] if token.item() != pad_idx]
        
        # Skip empty sequences
        if not pred_seq or not target_seq:
            continue
        
        # Calculate n-gram precision for each n
        precisions = []
        for n in range(1, min(max_n + 1, len(target_seq) + 1)):
            pred_ngrams = _get_ngrams(pred_seq, n)
            target_ngrams = _get_ngrams(target_seq, n)
            
            matches = len(pred_ngrams & target_ngrams)
            total = len(pred_ngrams) or 1  # Avoid division by zero
            precisions.append(matches / total)
        
        # Calculate brevity penalty
        bp = min(1.0, np.exp(1 - len(target_seq) / (len(pred_seq) or 1)))
        
        # Compute final BLEU score
        if any(precisions):
            score = bp * np.exp(np.mean(np.log(np.array(precisions) + 1e-10)))
            scores.append(score)
    
    # Return average BLEU score across batch
    return np.mean(scores) if scores else 0.0

def _get_ngrams(sequence, n):
    """
    Helper function to get n-grams from a sequence.
    
    Args:
        sequence: List of tokens
        n: n-gram size
        
    Returns:
        Set of n-grams
    """
    ngrams = set()
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i+n])
        ngrams.add(ngram)
    return ngrams

def perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity value
    """
    return torch.exp(loss).item() 