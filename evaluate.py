"""
Evaluation script for the Transformer model.
"""
import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm

from config import TransformerConfig
from model import Transformer
from tokenizer import SimpleTokenizer
from utils.data_utils import load_data, process_data, create_data_loader
from utils.metrics import accuracy, bleu_score, perplexity

def evaluate_model(model, dataloader, device, tokenizer=None, compute_bleu=False):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Transformer model
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        tokenizer: Tokenizer for decoding predictions (required for BLEU)
        compute_bleu: Whether to compute BLEU score
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    # Track predictions and targets for BLEU
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Target for loss computation (shifted right)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            logits, _ = model(src, tgt_input)
            
            # Calculate loss
            loss = model.loss_fn(logits, tgt_output)
            
            # Calculate accuracy
            batch_acc = accuracy(logits, tgt_output)
            
            # Store predictions and targets for BLEU calculation
            if compute_bleu:
                pred_tokens = logits.argmax(dim=-1)
                all_predictions.extend(pred_tokens.cpu())
                all_targets.extend(tgt_output.cpu())
            
            # Update metrics
            batch_size = src.size(0)
            total_loss += loss.item() * batch_size
            total_acc += batch_acc * batch_size
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{batch_acc:.4f}"})
    
    # Calculate average metrics
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    avg_ppl = perplexity(avg_loss)
    
    # Compute BLEU score if requested
    bleu = None
    if compute_bleu and tokenizer:
        bleu = bleu_score(torch.stack(all_predictions), torch.stack(all_targets))
    
    metrics = {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'perplexity': avg_ppl
    }
    
    if bleu is not None:
        metrics['bleu'] = bleu
    
    return metrics

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
    if os.path.exists(os.path.join(os.path.dirname(args.model_path), 'args.json')):
        with open(os.path.join(os.path.dirname(args.model_path), 'args.json'), 'r') as f:
            saved_args = json.load(f)
        
        config = TransformerConfig(
            vocab_size=saved_args.get('vocab_size', 30000),
            d_model=saved_args.get('d_model', 512),
            num_heads=saved_args.get('num_heads', 8),
            num_encoder_layers=saved_args.get('num_encoder_layers', 6),
            num_decoder_layers=saved_args.get('num_decoder_layers', 6),
            d_ff=saved_args.get('d_ff', 2048),
            max_seq_length=saved_args.get('max_seq_length', 512),
            dropout_rate=saved_args.get('dropout_rate', 0.1)
        )
    else:
        config = TransformerConfig()
    
    print("Model configuration:")
    print(config)
    
    # Load model
    model = load_model(args.model_path, config, device)
    print("Model loaded successfully")
    
    # Load tokenizer
    tokenizer_path = args.tokenizer_path or os.path.join(os.path.dirname(args.model_path), 'tokenizer.json')
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    print("Tokenizer loaded successfully")
    
    # Load and process test data
    print("Loading and processing test data...")
    test_data = load_data(os.path.join(args.data_dir, 'raw'), split='test')
    test_src, test_tgt = process_data(test_data, tokenizer, config.max_seq_length)
    test_loader = create_data_loader(test_src, test_tgt, args.batch_size, shuffle=False)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device, tokenizer, args.compute_bleu)
    
    # Print metrics
    print("\nEvaluation results:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Save metrics to file
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer model")
    
    # Model options
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer vocabulary")
    
    # Data options
    parser.add_argument("--data_dir", type=str, default="/sgl-workspace/Transformer/data", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--compute_bleu", action="store_true", help="Compute BLEU score")
    
    # Output options
    parser.add_argument("--output_file", type=str, help="Path to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to evaluate on")
    
    args = parser.parse_args()
    main(args) 