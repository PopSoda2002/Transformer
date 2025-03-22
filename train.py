"""
Training script for the Transformer model.
"""
import os
import argparse
import time
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import TransformerConfig
from model import Transformer
from tokenizer import SimpleTokenizer
from utils.data_utils import load_data, process_data, create_data_loader, create_dummy_data
from utils.metrics import accuracy, perplexity

def get_learning_rate(step, d_model, warmup_steps):
    """
    Get learning rate with warmup and decay as described in the paper.
    
    Args:
        step: Current training step
        d_model: Model dimension
        warmup_steps: Number of warmup steps
        
    Returns:
        Learning rate value
    """
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    """
    Train the model for one epoch.
    
    Args:
        model: Transformer model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        config: Configuration object
        
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (src, tgt) in enumerate(pbar):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Target for loss computation (shifted right)
        tgt_input = tgt[:, :-1]  # Remove last token (will be predicted)
        tgt_output = tgt[:, 1:]  # Remove first token (<sos>)
        
        # Forward pass
        logits, _ = model(src, tgt_input)
        
        # Calculate loss
        loss = model.loss_fn(logits, tgt_output, config.label_smoothing)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        # Calculate accuracy
        batch_acc = accuracy(logits, tgt_output)
        
        # Update metrics
        batch_size = src.size(0)
        total_loss += loss.item() * batch_size
        total_acc += batch_acc * batch_size
        total_samples += batch_size
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}", 
            'acc': f"{batch_acc:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.7f}"
        })
    
    # Calculate average metrics
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    
    return avg_loss, avg_acc

def validate(model, dataloader, device, config):
    """
    Validate the model on validation data.
    
    Args:
        model: Transformer model
        dataloader: DataLoader for validation data
        device: Device to validate on
        config: Configuration object
        
    Returns:
        Average loss and accuracy for validation
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
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
    
    return avg_loss, avg_acc

def save_checkpoint(model, optimizer, scheduler, epoch, loss, acc, path):
    """
    Save model checkpoint.
    
    Args:
        model: Transformer model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        loss: Validation loss
        acc: Validation accuracy
        path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'acc': acc
    }
    
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, scheduler, path, device):
    """
    Load model checkpoint.
    
    Args:
        model: Transformer model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        path: Path to load checkpoint from
        device: Device to load to
        
    Returns:
        Epoch number and metrics
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['acc']

def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up config
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        label_smoothing=args.label_smoothing
    )
    
    print("Configuration:")
    print(config)
    
    # Create model
    model = Transformer(config).to(device)
    print(f"Created Transformer model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Set up optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), 
                           lr=config.learning_rate, 
                           betas=(0.9, 0.98), 
                           eps=1e-9)
    
    if args.use_custom_lr_schedule:
        # Custom learning rate schedule from the paper
        class CustomLRScheduler:
            def __init__(self, optimizer, d_model, warmup_steps):
                self.optimizer = optimizer
                self.d_model = d_model
                self.warmup_steps = warmup_steps
                self.current_step = 0
                
            def step(self):
                self.current_step += 1
                lr = get_learning_rate(self.current_step, self.d_model, self.warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    
            def state_dict(self):
                return {'current_step': self.current_step}
                
            def load_state_dict(self, state_dict):
                self.current_step = state_dict['current_step']
                
        scheduler = CustomLRScheduler(optimizer, config.d_model, config.warmup_steps)
    else:
        # PyTorch's built-in scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=args.epochs * (args.dataset_size // config.batch_size + 1),
            pct_start=0.1
        )
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Starting epoch
    start_epoch = 0
    
    # Load checkpoint if specified
    if args.checkpoint and os.path.exists(args.checkpoint):
        start_epoch, val_loss, val_acc = load_checkpoint(model, optimizer, scheduler, args.checkpoint, device)
        print(f"Loaded checkpoint from epoch {start_epoch} with validation loss {val_loss:.4f} and accuracy {val_acc:.4f}")
        start_epoch += 1
    
    # Create data directory if using real data
    if not args.use_dummy_data:
        os.makedirs(os.path.join(args.data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(args.data_dir, 'processed'), exist_ok=True)
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        
        # Load train data
        train_data = load_data(os.path.join(args.data_dir, 'raw'), split='train')
        
        # Create tokenizer
        tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        
        # Build vocabulary from training data
        print("Building vocabulary...")
        tokenizer.build_vocab(train_data['source'] + train_data['target'])
        
        # Save tokenizer
        tokenizer.save(os.path.join(args.output_dir, 'tokenizer.json'))
        
        # Process training data
        print("Processing training data...")
        train_src, train_tgt = process_data(train_data, tokenizer, config.max_seq_length)
        train_loader = create_data_loader(train_src, train_tgt, config.batch_size, shuffle=True)
        
        # Load and process validation data
        print("Processing validation data...")
        val_data = load_data(os.path.join(args.data_dir, 'raw'), split='valid')
        val_src, val_tgt = process_data(val_data, tokenizer, config.max_seq_length)
        val_loader = create_data_loader(val_src, val_tgt, config.batch_size, shuffle=False)
    else:
        # Use dummy data for testing
        print("Using dummy data...")
        
        # Create dummy training data
        train_src, train_tgt = create_dummy_data(
            batch_size=config.batch_size,
            src_seq_len=config.max_seq_length,
            tgt_seq_len=config.max_seq_length,
            vocab_size=config.vocab_size
        )
        train_loader = [(train_src, train_tgt)] * 100  # 100 batches
        
        # Create dummy validation data
        val_src, val_tgt = create_dummy_data(
            batch_size=config.batch_size,
            src_seq_len=config.max_seq_length,
            tgt_seq_len=config.max_seq_length,
            vocab_size=config.vocab_size
        )
        val_loader = [(val_src, val_tgt)] * 10  # 10 batches
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Train
        train_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, config)
        train_time = time.time() - train_start_time
        
        # Validate
        val_start_time = time.time()
        val_loss, val_acc = validate(model, val_loader, device, config)
        val_time = time.time() - val_start_time
        
        # Calculate perplexity
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} | Train Acc: {train_acc:.4f} | Time: {train_time:.2f}s")
        print(f"Valid Loss: {val_loss:.4f} | Valid PPL: {val_ppl:.2f} | Valid Acc: {val_acc:.4f} | Time: {val_time:.2f}s")
        
        # Write to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Perplexity/train', train_ppl, epoch)
        writer.add_scalar('Perplexity/val', val_ppl, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pt')
        save_checkpoint(model, optimizer, scheduler, epoch+1, val_loss, val_acc, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, scheduler, epoch+1, val_loss, val_acc, best_model_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    
    # Data options
    parser.add_argument("--data_dir", type=str, default="/sgl-workspace/Transformer/data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="/sgl-workspace/Transformer/output", help="Output directory")
    parser.add_argument("--use_dummy_data", action="store_true", help="Use dummy data for testing")
    parser.add_argument("--dataset_size", type=int, default=10000, help="Number of samples in dataset (for LR scheduler)")
    
    # Model options
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    
    # Training options
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="Number of warmup steps")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--use_custom_lr_schedule", action="store_true", help="Use custom LR schedule from paper")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    main(args) 