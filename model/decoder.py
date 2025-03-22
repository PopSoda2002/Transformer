"""
Decoder components for the Transformer model.
"""
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .ffn import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    """
    Single decoder layer consisting of masked self-attention, 
    encoder-decoder attention, and feed-forward network.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        
    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            encoder_output: Output from encoder of shape (batch_size, input_seq_len, d_model)
            look_ahead_mask: Mask for masked self-attention of shape (batch_size, 1, seq_len, seq_len)
            padding_mask: Mask for encoder-decoder attention of shape (batch_size, 1, 1, input_seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
            Dictionary of attention weights
        """
        attention_weights = {}
        
        # Masked multi-head self-attention
        attn1_output, attn1_weights = self.masked_self_attention(x, x, x, look_ahead_mask)
        attention_weights['self_attention'] = attn1_weights
        
        # Multi-head encoder-decoder attention
        attn2_output, attn2_weights = self.encoder_decoder_attention(
            attn1_output, encoder_output, encoder_output, padding_mask
        )
        attention_weights['encoder_decoder_attention'] = attn2_weights
        
        # Position-wise feed-forward network
        output = self.feed_forward(attn2_output)
        
        return output, attention_weights

class Decoder(nn.Module):
    """
    Transformer decoder consisting of multiple decoder layers.
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length, dropout_rate=0.1):
        super(Decoder, self).__init__()
        from .embedding import Embeddings
        
        self.embeddings = Embeddings(vocab_size, d_model, max_seq_length, dropout_rate)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            encoder_output: Output from encoder of shape (batch_size, input_seq_len, d_model)
            look_ahead_mask: Mask for masked self-attention
            padding_mask: Mask for encoder-decoder attention
            
        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_size)
            Dictionary of attention weights from each layer
        """
        # Apply embeddings and positional encoding
        x = self.embeddings(x)
        
        # Track attention weights from each layer
        attention_weights = {}
        
        # Apply decoder layers
        for i, layer in enumerate(self.layers):
            x, layer_attn_weights = layer(x, encoder_output, look_ahead_mask, padding_mask)
            for attn_name, attn_weights in layer_attn_weights.items():
                attention_weights[f'decoder_layer{i+1}_{attn_name}'] = attn_weights
        
        # Apply final linear layer
        output = self.final_layer(x)
        
        return output, attention_weights 