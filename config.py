"""
Configuration parameters for the Transformer model.
"""

class TransformerConfig:
    def __init__(
        self,
        vocab_size=30000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_length=512,
        dropout_rate=0.1,
        layer_norm_eps=1e-6,
        learning_rate=0.0001,
        warmup_steps=4000,
        batch_size=64,
        epochs=20,
        label_smoothing=0.1,
    ):
        """
        Transformer model configuration.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of model (embedding dimension)
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Dimension of feed forward network
            max_seq_length: Maximum sequence length
            dropout_rate: Dropout rate
            layer_norm_eps: Layer normalization epsilon
            learning_rate: Initial learning rate
            warmup_steps: Number of warmup steps for learning rate scheduler
            batch_size: Batch size for training
            epochs: Number of training epochs
            label_smoothing: Label smoothing factor
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.label_smoothing = label_smoothing
        
    def __str__(self):
        """Return string representation of config."""
        return str(self.__dict__) 