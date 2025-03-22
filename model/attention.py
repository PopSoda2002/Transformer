"""
Attention mechanisms for the Transformer model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism as described in 'Attention Is All You Need'.
    """
    def __init__(self, dropout_rate=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, q, k, v, mask=None):
        """
        Calculate scaled dot-product attention.
        
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len_q, d_k)
            k: Key tensor of shape (batch_size, num_heads, seq_len_k, d_k)
            v: Value tensor of shape (batch_size, num_heads, seq_len_v, d_v)
            mask: Optional mask tensor of shape (batch_size, 1, 1, seq_len_k) or (batch_size, 1, seq_len_q, seq_len_k)
            
        Returns:
            output: Attention output of shape (batch_size, num_heads, seq_len_q, d_v)
            attention_weights: Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        d_k = q.size(-1)
        
        # Compute attention scores: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as described in 'Attention Is All You Need'.
    """
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # Output projection
        self.wo = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout_rate)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, d_k)
        
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            batch_size: Batch size
            
        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, q, k, v, mask=None):
        """
        Multi-head attention forward pass.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len_q, d_model)
            k: Key tensor of shape (batch_size, seq_len_k, d_model)
            v: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
            
        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_model)
            attention_weights: Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = q.size(0)
        residual = q
        
        # Linear projections and split heads
        q = self.split_heads(self.wq(q), batch_size)  # (batch_size, num_heads, seq_len_q, d_k)
        k = self.split_heads(self.wk(k), batch_size)  # (batch_size, num_heads, seq_len_k, d_k)
        v = self.split_heads(self.wv(v), batch_size)  # (batch_size, num_heads, seq_len_v, d_k)
        
        # Apply scaled dot-product attention
        attn_output, attention_weights = self.attention(q, k, v, mask)
        
        # Concatenate heads and apply final linear projection
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len_q, num_heads, d_k)
        attn_output = attn_output.view(batch_size, -1, self.d_model)  # (batch_size, seq_len_q, d_model)
        
        output = self.wo(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        
        return output, attention_weights 