import os
import json
import torch
import argparse

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

from utils import *

def main():
    parser = argparse.ArgumentParser(description='Train Multi-Task Facial Attribute Classifier')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--root_dir', type=str, default='', help='Root directory for images')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for main label task')
    
    # Multi-task specific arguments
    parser.add_argument('--label_weight', type=float, default=0.7, help='Weight for main label task')
    parser.add_argument('--aux_weight', type=float, default=0.1, help='Weight for each auxiliary task')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df, val_df = load_and_preprocess_data(args.csv_path, args.root_dir)
    # Save the split DataFrames
    train_df.to_csv(os.path.join(args.save_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(args.save_dir, 'val_split.csv'), index=False)

    print(f"Saved split DataFrames to {args.save_dir}")
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_df, val_df, args.root_dir, args.batch_size, args.num_workers
    )
    
    # Initialize model
    model = MultiTaskFacialAttributeClassifier(pretrained=True)
    model = model.to(device)
    
    # Setup loss function
    task_weights = {
        0: args.aux_weight,      # glasses
        1: args.aux_weight,      # hat  
        2: args.aux_weight,      # facial_hair
        3: args.label_weight     # label (main task)
    }
    
    class_weights = {}
    if args.use_class_weights:
        # Only apply class weights to the main label task
        label_class_weight = calculate_class_weights(train_df, 'label').to(device)
        class_weights[3] = label_class_weight  # index 3 is the main label
        print(f"Using class weight for main label task: {label_class_weight.item():.3f}")
    
    criterion = MultiTaskLoss(task_weights=task_weights, class_weights=class_weights)
    
    print(f"Task weights: {task_weights}")
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_main_hter = float('inf')
    train_history = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # Training phase
        train_loss, train_main_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, optimal_metrics, standard_metrics, optimal_thresholds = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save training history
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_main_acc': train_main_acc,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        # Add metrics for all tasks
        for task in ['glasses', 'hat', 'facial_hair', 'label']:
            epoch_data[f'{task}_optimal_acc'] = optimal_metrics[task]['accuracy']
            epoch_data[f'{task}_optimal_hter'] = optimal_metrics[task]['hter']
            epoch_data[f'{task}_optimal_threshold'] = optimal_metrics[task]['optimal_threshold']
            epoch_data[f'{task}_standard_acc'] = standard_metrics[task]['accuracy']
            epoch_data[f'{task}_standard_hter'] = standard_metrics[task]['hter']
        
        train_history.append(epoch_data)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Main Acc: {train_main_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("\nOptimal Threshold Results:")
        print(f"{'Task':<12} {'Acc':<6} {'HTER':<6} {'FAR':<6} {'FRR':<6} {'Thresh':<6}")
        print("-" * 60)
        
        for task in ['glasses', 'hat', 'facial_hair', 'label']:
            m = optimal_metrics[task]
            print(f"{task:<12} {m['accuracy']:<6.3f} {m['hter']:<6.3f} {m['far']:<6.3f} {m['frr']:<6.3f} {m['optimal_threshold']:<6.3f}")
        
        print("\nStandard (0.5) Threshold Results:")
        print(f"{'Task':<12} {'Acc':<6} {'HTER':<6} {'FAR':<6} {'FRR':<6}")
        print("-" * 48)
        
        for task in ['glasses', 'hat', 'facial_hair', 'label']:
            m = standard_metrics[task]
            print(f"{task:<12} {m['accuracy']:<6.3f} {m['hter']:<6.3f} {m['far']:<6.3f} {m['frr']:<6.3f}")
        
        # Save best model based on main task HTER
        main_hter = optimal_metrics['label']['hter']
        if main_hter < best_main_hter:
            best_main_hter = main_hter
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss, main_hter, best_model_path
            )
            print(f"\n*** New best model saved with main task HTER: {main_hter:.4f} ***")
        
        # Save latest checkpoint
        latest_checkpoint_path = os.path.join(args.save_dir, 'latest_checkpoint.pth')
        save_checkpoint(
            model, optimizer, epoch + 1, val_loss, main_hter, latest_checkpoint_path
        )
    
    # Save training history
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2, default=float)  # default=float to handle numpy types
    
    print(f"\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Best Main Task HTER: {best_main_hter:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training history saved to: {history_path}")
    
    # Print final summary
    print(f"\nFinal Results Summary:")
    print(f"Task weights used: Label={args.label_weight}, Auxiliary={args.aux_weight}")
    print(f"Class weights applied to main task: {args.use_class_weights}")

if __name__ == '__main__':
    main()