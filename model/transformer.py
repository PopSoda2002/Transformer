"""
Complete Transformer model implementation based on 'Attention Is All You Need'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    """
    def __init__(self, config):
        """
        Args:
            config: TransformerConfig object containing model parameters
        """
        super(Transformer, self).__init__()
        
        self.config = config
        self.encoder = Encoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_layers=config.num_encoder_layers,
            max_seq_length=config.max_seq_length,
            dropout_rate=config.dropout_rate
        )
        
        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_layers=config.num_decoder_layers,
            max_seq_length=config.max_seq_length,
            dropout_rate=config.dropout_rate
        )
        
    def create_padding_mask(self, seq):
        """
        Create padding mask for attention.
        
        Args:
            seq: Input sequence tensor of shape (batch_size, seq_len)
            
        Returns:
            Padding mask of shape (batch_size, 1, 1, seq_len)
        """
        # 1 for padding token positions, 0 for non-padding
        mask = (seq == 0).float()
        
        # Add extra dimensions for broadcasting
        return mask.unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size):
        """
        Create look-ahead mask for decoder self-attention.
        
        Args:
            size: Size of the square mask
            
        Returns:
            Look-ahead mask of shape (1, size, size)
        """
        # Create upper triangular matrix with 1s
        mask = torch.triu(torch.ones(size, size), diagonal=1).float()
        
        # Convert to binary mask (1 for positions to mask, 0 for valid positions)
        return mask.unsqueeze(0)
    
    def forward(self, src, tgt):
        """
        Forward pass through the Transformer model.
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_seq_len)
            tgt: Target sequence tensor of shape (batch_size, tgt_seq_len)
            
        Returns:
            Output probabilities of shape (batch_size, tgt_seq_len, vocab_size)
            Dictionary of attention weights
        """
        # Create padding mask for encoder
        enc_padding_mask = self.create_padding_mask(src)
        
        # Create padding mask for decoder's attention to encoder output
        dec_padding_mask = self.create_padding_mask(src)
        
        # Create look-ahead mask for decoder self-attention
        tgt_seq_len = tgt.size(1)
        look_ahead_mask = self.create_look_ahead_mask(tgt_seq_len)
        
        # Create combined mask for decoder self-attention (combines padding and look-ahead)
        dec_target_padding_mask = self.create_padding_mask(tgt)
        combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
        
        # Encode source sequence
        enc_output, enc_attention_weights = self.encoder(src, enc_padding_mask)
        
        # Decode target sequence
        dec_output, dec_attention_weights = self.decoder(
            tgt, enc_output, combined_mask, dec_padding_mask
        )
        
        # Combine attention weights from encoder and decoder
        attention_weights = {**enc_attention_weights, **dec_attention_weights}
        
        return dec_output, attention_weights
    
    def loss_fn(self, logits, targets, smoothing=0.0):
        """
        Compute loss with optional label smoothing.
        
        Args:
            logits: Model output logits of shape (batch_size, seq_len, vocab_size)
            targets: Target indices of shape (batch_size, seq_len)
            smoothing: Label smoothing factor
            
        Returns:
            Loss value
        """
        batch_size, seq_len, vocab_size = logits.size()
        
        if smoothing > 0.0:
            # Create smoothed target distribution
            targets_one_hot = torch.zeros_like(logits).scatter_(
                2, targets.unsqueeze(-1), 1.0
            )
            smoothed_targets = (
                targets_one_hot * (1.0 - smoothing) + 
                smoothing / vocab_size
            )
            
            # Compute log-likelihood
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(smoothed_targets * log_probs).sum(dim=-1)
            
            # Mask out padding positions (where target == 0)
            mask = (targets != 0).float()
            loss = (loss * mask).sum() / mask.sum()
        else:
            # Standard cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
                ignore_index=0  # Ignore padding index
            )
        
        return loss 