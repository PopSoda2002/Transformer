# Transformer Implementation from Scratch

This project contains a hand-written Transformer implementation for training on text data.

## Directory Structure

```
/sgl-workspace/Transformer/
├── data/               # Data directory
│   ├── raw/            # Raw datasets
│   └── processed/      # Preprocessed datasets
├── model/              # Transformer model components
│   ├── __init__.py
│   ├── attention.py    # Multi-head attention mechanism
│   ├── embedding.py    # Token and positional embeddings
│   ├── encoder.py      # Transformer encoder
│   ├── decoder.py      # Transformer decoder
│   ├── ffn.py          # Feed-forward network
│   └── transformer.py  # Complete transformer architecture
├── utils/              # Utility functions
│   ├── __init__.py
│   ├── data_utils.py   # Data loading and preprocessing
│   ├── masking.py      # Padding and look-ahead masks
│   └── metrics.py      # Performance metrics
├── config.py           # Configuration parameters
├── tokenizer.py        # Tokenization utilities
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── inference.py        # Inference script
├── requirements.txt    # Dependencies
└── Readme.md           # Project documentation
```

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Prepare your data in the `data/raw` directory

3. Run the training script:
```
python train.py
```

## Implementation Details

This implementation follows the architecture described in the "Attention Is All You Need" paper, with all components written from scratch for educational purposes.