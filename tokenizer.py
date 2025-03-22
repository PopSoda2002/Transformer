"""
Simple tokenizer implementation for text preprocessing.
"""
import os
import json
from collections import Counter

class SimpleTokenizer:
    """
    Simple tokenizer for transformer model.
    """
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.token_to_id = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.id_to_token = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.special_tokens = len(self.token_to_id)
        self.vocab_built = False
        
    def build_vocab(self, text_data):
        """
        Build vocabulary from text data.
        
        Args:
            text_data: List of text samples
        """
        # Count token frequencies
        counter = Counter()
        for text in text_data:
            tokens = self._tokenize(text)
            counter.update(tokens)
        
        # Select most common tokens (minus the special tokens we already have)
        vocab_size = min(self.vocab_size, len(counter) + self.special_tokens)
        tokens = [token for token, _ in counter.most_common(vocab_size - self.special_tokens)]
        
        # Add tokens to vocabulary
        for token in tokens:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
        
        self.vocab_built = True
        
    def _tokenize(self, text):
        """
        Tokenize text into individual tokens.
        This is a simple whitespace/punctuation tokenizer.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Simple whitespace and punctuation tokenization
        # In a real-world scenario, use a more sophisticated tokenizer
        import re
        return re.findall(r'\w+|[^\w\s]', text.lower())
    
    def encode(self, text, max_length=None, pad_to_max_length=False):
        """
        Encode text to token ids.
        
        Args:
            text: Input text string
            max_length: Maximum sequence length
            pad_to_max_length: Whether to pad sequence to max_length
            
        Returns:
            List of token ids
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary has not been built. Call build_vocab() first.")
        
        # Tokenize text
        tokens = self._tokenize(text)
        
        # Add start and end tokens
        tokens = ['<sos>'] + tokens + ['<eos>']
        
        # Convert tokens to ids
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id['<unk>'])
        
        # Truncate if necessary
        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length-1] + [self.token_to_id['<eos>']]
        
        # Pad to max length if requested
        if max_length is not None and pad_to_max_length:
            padding_length = max_length - len(ids)
            if padding_length > 0:
                ids = ids + [self.token_to_id['<pad>']] * padding_length
        
        return ids
    
    def decode(self, ids, remove_special_tokens=True):
        """
        Decode token ids to text.
        
        Args:
            ids: List of token ids
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded text string
        """
        # Convert ids to tokens
        tokens = []
        for idx in ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]
                # Skip special tokens if requested
                if remove_special_tokens and token in ['<pad>', '<sos>', '<eos>', '<unk>']:
                    continue
                tokens.append(token)
            else:
                tokens.append('<unk>')
        
        # Join tokens to form text
        text = ' '.join(tokens)
        
        # Clean up spacing around punctuation (simple rule)
        import re
        text = re.sub(r'\s([,.!?;:])', r'\1', text)
        
        return text
    
    def save(self, path):
        """
        Save tokenizer vocabulary to file.
        
        Args:
            path: Path to save the vocabulary
        """
        vocab_data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()}
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path):
        """
        Load tokenizer vocabulary from file.
        
        Args:
            path: Path to the vocabulary file
            
        Returns:
            Tokenizer instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        tokenizer = cls(vocab_size=vocab_data['vocab_size'])
        tokenizer.token_to_id = vocab_data['token_to_id']
        tokenizer.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        tokenizer.special_tokens = sum(1 for token in tokenizer.token_to_id if token.startswith('<'))
        tokenizer.vocab_built = True
        
        return tokenizer 