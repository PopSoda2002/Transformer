"""
Embedding modules for the Transformer model.
"""
import math
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Token embedding layer with scaled weights.
    """
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Embedded tensor of shape (batch_size, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Positional encoding as described in 'Attention Is All You Need'.
    """
    def __init__(self, d_model, max_seq_length=5000, dropout_rate=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Create constant positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and prepare for addition to embeddings
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Embedded tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Embeddings(nn.Module):
    """
    Combined token embedding and positional encoding.
    """
    def __init__(self, vocab_size, d_model, max_seq_length, dropout_rate=0.1):
        super(Embeddings, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout_rate)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Embedded tensor with positional encoding of shape (batch_size, seq_len, d_model)
        """
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        return x 