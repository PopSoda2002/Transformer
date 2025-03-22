"""
Encoder components for the Transformer model.
"""
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .ffn import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
    Single encoder layer consisting of self-attention and feed-forward network.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional padding mask of shape (batch_size, 1, 1, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
            Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Multi-head self-attention
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        
        # Position-wise feed-forward network
        output = self.feed_forward(attn_output)
        
        return output, attn_weights

class Encoder(nn.Module):
    """
    Transformer encoder consisting of multiple encoder layers.
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length, dropout_rate=0.1):
        super(Encoder, self).__init__()
        from .embedding import Embeddings
        
        self.embeddings = Embeddings(vocab_size, d_model, max_seq_length, dropout_rate)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            mask: Optional padding mask of shape (batch_size, 1, 1, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
            Dictionary of attention weights from each layer
        """
        # Apply embeddings and positional encoding
        x = self.embeddings(x)
        
        # Track attention weights from each layer
        attention_weights = {}
        
        # Apply encoder layers
        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x, mask)
            attention_weights[f'encoder_layer{i+1}'] = attn_weights
        
        return x, attention_weights 