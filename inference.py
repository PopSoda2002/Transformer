"""
Inference script for the Transformer model.
"""
import os
import argparse
import json
import torch
import torch.nn.functional as F

from config import TransformerConfig
from model import Transformer
from tokenizer import SimpleTokenizer

def beam_search(model, src, tokenizer, max_len, beam_size=4, device="cpu"):
    """
    Perform beam search decoding.
    
    Args:
        model: Trained Transformer model
        src: Source sequence tensor of shape (1, src_seq_len)
        tokenizer: Tokenizer object
        max_len: Maximum length for generated sequence
        beam_size: Beam size for search
        device: Device to run inference on
        
    Returns:
        Best output sequence
    """
    model.eval()
    
    # Create source padding mask
    src_mask = model.create_padding_mask(src)
    
    # Get encoder output
    with torch.no_grad():
        enc_output, _ = model.encoder(src, src_mask)
    
    # Initialize beam with start token
    sos_idx = tokenizer.token_to_id['<sos>']
    eos_idx = tokenizer.token_to_id['<eos>']
    
    # Initialize with <sos> token
    beams = [(torch.tensor([[sos_idx]], device=device), 0.0)]
    completed_beams = []
    
    # Beam search loop
    for _ in range(max_len):
        candidates = []
        
        # Process all existing beams
        for seq, score in beams:
            # If the last token is <eos>, add to completed beams
            if seq[0, -1].item() == eos_idx:
                completed_beams.append((seq, score))
                continue
            
            # Create look-ahead mask for the current sequence
            look_ahead_mask = model.create_look_ahead_mask(seq.size(1))
            combined_mask = torch.max(
                look_ahead_mask, 
                torch.zeros((1, seq.size(1), seq.size(1)), device=device)
            )
            
            # Predict next token probabilities
            with torch.no_grad():
                output, _ = model.decoder(seq, enc_output, combined_mask, src_mask)
                next_token_logits = output[0, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
            
            # Get top-k tokens
            topk_probs, topk_indices = torch.topk(next_token_probs, beam_size)
            
            # Add new candidates
            for i in range(beam_size):
                token_idx = topk_indices[i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_probs[i].item()
                
                # Create new sequence by appending predicted token
                new_seq = torch.cat([seq, token_idx], dim=1)
                
                # Calculate new score (log probability)
                new_score = score + torch.log(topk_probs[i]).item()
                
                candidates.append((new_seq, new_score))
        
        # If all beams have completed, stop
        if not candidates:
            break
        
        # Select top-k candidates for next iteration
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Stop if all beams have reached the maximum length
        if all(seq.size(1) >= max_len for seq, _ in beams):
            break
    
    # Add any remaining beams to completed beams
    completed_beams.extend(beams)
    
    # Return the best completed beam
    if completed_beams:
        best_seq, _ = max(completed_beams, key=lambda x: x[1])
    else:
        best_seq = beams[0][0]
    
    return best_seq.squeeze(0).cpu().numpy()

def greedy_decode(model, src, tokenizer, max_len, device="cpu"):
    """
    Perform greedy decoding.
    
    Args:
        model: Trained Transformer model
        src: Source sequence tensor of shape (1, src_seq_len)
        tokenizer: Tokenizer object
        max_len: Maximum length for generated sequence
        device: Device to run inference on
        
    Returns:
        Output sequence
    """
    model.eval()
    
    # Create source padding mask
    src_mask = model.create_padding_mask(src)
    
    # Get encoder output
    with torch.no_grad():
        enc_output, _ = model.encoder(src, src_mask)
    
    # Initialize with <sos> token
    sos_idx = tokenizer.token_to_id['<sos>']
    eos_idx = tokenizer.token_to_id['<eos>']
    
    output_seq = torch.tensor([[sos_idx]], device=device)
    
    # Generate tokens one by one
    for i in range(max_len):
        # Create look-ahead mask for the current sequence
        look_ahead_mask = model.create_look_ahead_mask(output_seq.size(1))
        combined_mask = torch.max(
            look_ahead_mask, 
            torch.zeros((1, output_seq.size(1), output_seq.size(1)), device=device)
        )
        
        # Predict next token
        with torch.no_grad():
            output, _ = model.decoder(output_seq, enc_output, combined_mask, src_mask)
            next_token = output[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        
        # Append next token to output sequence
        output_seq = torch.cat([output_seq, next_token], dim=1)
        
        # Stop if <eos> token is generated
        if next_token.item() == eos_idx:
            break
    
    return output_seq.squeeze(0).cpu().numpy()

def load_model(model_path, config, device):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        config: Model configuration
        device: Device to load model to
        
    Returns:
        Loaded model
    """
    model = Transformer(config).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def main(args):
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = TransformerConfig(**config_dict)
    else:
        # Default configuration
        config = TransformerConfig()
    
    print("Model configuration:")
    print(config)
    
    # Load model
    model = load_model(args.model_path, config, device)
    print("Model loaded successfully")
    
    # Load tokenizer
    tokenizer = SimpleTokenizer.load(args.tokenizer_path)
    print("Tokenizer loaded successfully")
    
    # Process input text
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        input_texts = [line.strip() for line in lines]
    else:
        input_texts = [args.input_text]
    
    # Process each input text
    results = []
    for text in input_texts:
        print(f"\nInput text: {text}")
        
        # Encode input text
        src_ids = tokenizer.encode(text, max_length=config.max_seq_length, pad_to_max_length=True)
        src = torch.tensor([src_ids], device=device)
        
        # Generate output sequence
        if args.beam_size > 1:
            print(f"Using beam search with beam size {args.beam_size}")
            output_ids = beam_search(model, src, tokenizer, args.max_output_length, args.beam_size, device)
        else:
            print("Using greedy decoding")
            output_ids = greedy_decode(model, src, tokenizer, args.max_output_length, device)
        
        # Decode output sequence
        output_text = tokenizer.decode(output_ids)
        print(f"Output text: {output_text}")
        
        # Store result
        results.append({
            'input': text,
            'output': output_text
        })
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained Transformer model")
    
    # Model options
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer vocabulary")
    parser.add_argument("--config", type=str, help="Path to model configuration file")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_text", type=str, help="Input text for inference")
    group.add_argument("--input_file", type=str, help="Path to file with input texts (one per line)")
    
    # Output options
    parser.add_argument("--output_file", type=str, help="Path to save output")
    
    # Decoding options
    parser.add_argument("--max_output_length", type=int, default=128, help="Maximum output sequence length")
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size for decoding (1 for greedy)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on")
    
    args = parser.parse_args()
    main(args) 