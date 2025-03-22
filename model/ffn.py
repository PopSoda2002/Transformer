"""
Position-wise Feed-Forward Network for the Transformer model.
"""
import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network as described in 'Attention Is All You Need'.
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    """
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        residual = x
        
        # First linear layer with ReLU activation
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # Second linear layer
        x = self.linear2(x)
        x = self.dropout(x)
        
        # Add residual connection and apply layer normalization
        x = self.layer_norm(x + residual)
        
        return x 