import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchvision.models as models
from tqdm import tqdm
import numpy as np

class FacialAttributeDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with image paths and labels
            root_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx]['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        # Get all labels: glasses, hat, facial_hair, label
        labels = self.dataframe.iloc[idx][['glasses', 'hat', 'facial_hair', 'label']].values.astype(float)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(labels, dtype=torch.float32)

class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights=None, class_weights=None):
        """
        Multi-task loss function for facial attribute classification
        
        Args:
            task_weights (dict): Weights for each task {task_index: weight}
            class_weights (dict): Class weights for imbalanced classes {task_index: torch.tensor}
        """
        super(MultiTaskLoss, self).__init__()
        self.task_weights = task_weights or {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.7}  # Default: main label gets 0.7 weight
        self.class_weights = class_weights or {}
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions (dict): Dictionary of predictions {'glasses': tensor, 'hat': tensor, 'facial_hair': tensor, 'label': tensor}
            targets (torch.Tensor): True labels [batch_size, num_labels] - [glasses, hat, facial_hair, label]
        """
        total_loss = 0.0
        task_names = ['glasses', 'hat', 'facial_hair', 'label']
        
        for i, task_name in enumerate(task_names):
            pred_i = predictions[task_name].squeeze()
            target_i = targets[:, i]
            
            # Use class weights if available for this task
            if i in self.class_weights:
                criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights[i])
            else:
                criterion = nn.BCEWithLogitsLoss()
            
            loss_i = criterion(pred_i, target_i)
            weighted_loss = self.task_weights[i] * loss_i
            total_loss += weighted_loss
            
        return total_loss

class MultiTaskFacialAttributeClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(MultiTaskFacialAttributeClassifier, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Shared feature layers
        self.shared_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads
        self.glasses_head = nn.Linear(256, 1)
        self.hat_head = nn.Linear(256, 1)
        self.facial_hair_head = nn.Linear(256, 1)
        self.label_head = nn.Linear(256, 1)  # Main classification task
        
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Pass through shared layers
        shared_features = self.shared_layers(features)
        
        # Task-specific predictions
        glasses_out = self.glasses_head(shared_features)
        hat_out = self.hat_head(shared_features)
        facial_hair_out = self.facial_hair_head(shared_features)
        label_out = self.label_head(shared_features)
        
        return {
            'glasses': glasses_out,
            'hat': hat_out,
            'facial_hair': facial_hair_out,
            'label': label_out

        }

################## HELPER  FUNCTIONS ##################

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    task_names = ['glasses', 'hat', 'facial_hair', 'label']
    
    # Track correct predictions for main task
    correct_main = 0
    total_main = 0
    
    train_bar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (data, targets) in enumerate(train_bar):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy for main task (label)
        main_pred_probs = torch.sigmoid(outputs['label'].squeeze())
        main_predicted = (main_pred_probs >= 0.5).float()
        main_targets = targets[:, 3]  # label is at index 3
        
        total_main += main_targets.size(0)
        correct_main += (main_predicted == main_targets).sum().item()
        
        # Update progress bar
        train_bar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Main_Acc': f'{100.*correct_main/total_main:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    main_epoch_acc = correct_main / total_main
    
    return epoch_loss, main_epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate the model with multi-task metrics"""
    model.eval()
    running_loss = 0.0
    all_predictions = {'glasses': [], 'hat': [], 'facial_hair': [], 'label': []}
    all_targets = []
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc='Validation')
        
        for batch_idx, (data, targets) in enumerate(val_bar):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            # Collect predictions and targets
            for task in all_predictions.keys():
                all_predictions[task].extend(outputs[task].cpu())
            all_targets.extend(targets.cpu())
            
            val_bar.set_postfix({'Loss': f'{running_loss/(batch_idx+1):.4f}'})
    
    epoch_loss = running_loss / len(val_loader)
    
    # Convert to tensors
    for task in all_predictions.keys():
        all_predictions[task] = torch.stack(all_predictions[task])
    all_targets = torch.stack(all_targets)
    
    # Find optimal thresholds for each task
    optimal_thresholds = find_optimal_threshold_multitask(all_predictions, all_targets)
    
    # Calculate metrics with optimal thresholds
    metrics_optimal = {}
    for task, threshold_info in optimal_thresholds.items():
        threshold = threshold_info['optimal_threshold']
        task_idx = ['glasses', 'hat', 'facial_hair', 'label'].index(task)
        
        pred_probs = torch.sigmoid(all_predictions[task].squeeze()).numpy()
        pred_binary = (pred_probs >= threshold).astype(int)
        target_binary = all_targets[:, task_idx].numpy().astype(int)
        
        # Calculate metrics
        tp = ((pred_binary == 1) & (target_binary == 1)).sum()
        fp = ((pred_binary == 1) & (target_binary == 0)).sum()
        tn = ((pred_binary == 0) & (target_binary == 0)).sum()
        fn = ((pred_binary == 0) & (target_binary == 1)).sum()
        
        eps = 1e-8
        accuracy = (tp + tn) / len(target_binary)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        far = fp / (fp + tn + eps)
        frr = fn / (fn + tp + eps)
        hter = (far + frr) / 2
        
        metrics_optimal[task] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'hter': hter,
            'far': far,
            'frr': frr,
            'optimal_threshold': threshold
        }
    
    # Also calculate standard metrics with 0.5 threshold
    standard_metrics = calculate_multitask_metrics(all_predictions, all_targets, threshold=0.5)
    
    return epoch_loss, metrics_optimal, standard_metrics, optimal_thresholds

def calculate_class_weights(df, column='label'):
    """
    Calculate class weights for binary classification
    
    Args:
        df (pd.DataFrame): DataFrame with labels
        column (str): Column name for labels
    
    Returns:
        torch.Tensor: Class weights
    """
    class_counts = df[column].value_counts().sort_index()
    
    if len(class_counts) == 2:
        # For binary classification: weight for positive class
        pos_weight = class_counts[0] / class_counts[1]  # neg_count / pos_count
        return torch.tensor(pos_weight, dtype=torch.float32)
    else:
        # For multi-class
        total_samples = len(df)
        weights = total_samples / (len(class_counts) * class_counts.values)
        return torch.FloatTensor(weights)

def calculate_multitask_metrics(predictions, targets, threshold=0.5):
    """
    Calculate metrics for multi-task classification
    
    Args:
        predictions (dict): Dictionary of predictions for each task
        targets (torch.Tensor): True labels [batch_size, num_labels]
        threshold (float): Decision threshold for classification
    
    Returns:
        dict: Metrics for each task
    """
    task_names = ['glasses', 'hat', 'facial_hair', 'label']
    results = {}
    
    for i, task_name in enumerate(task_names):
        pred_probs = torch.sigmoid(predictions[task_name].squeeze()).cpu().numpy()
        pred_binary = (pred_probs >= threshold).astype(int)
        target_binary = targets[:, i].cpu().numpy().astype(int)
        
        # Calculate confusion matrix components
        tp = ((pred_binary == 1) & (target_binary == 1)).sum()
        fp = ((pred_binary == 1) & (target_binary == 0)).sum()
        tn = ((pred_binary == 0) & (target_binary == 0)).sum()
        fn = ((pred_binary == 0) & (target_binary == 1)).sum()
        
        # Calculate rates
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        accuracy = (tp + tn) / len(target_binary)
        
        far = fp / (fp + tn + eps)
        frr = fn / (fn + tp + eps)
        hter = (far + frr) / 2
        
        results[task_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'hter': hter,
            'far': far,
            'frr': frr,
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        }
    
    return results

def find_optimal_threshold_multitask(predictions, targets, num_thresholds=100):
    """
    Find optimal thresholds for each task that minimize HTER
    
    Args:
        predictions (dict): Dictionary of predictions for each task
        targets (torch.Tensor): True labels [batch_size, num_labels]
        num_thresholds (int): Number of thresholds to try
    
    Returns:
        dict: Optimal thresholds and best HTERs for each task
    """
    task_names = ['glasses', 'hat', 'facial_hair', 'label']
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    results = {}
    
    for i, task_name in enumerate(task_names):
        pred_probs = torch.sigmoid(predictions[task_name].squeeze()).cpu().numpy()
        target_binary = targets[:, i].cpu().numpy().astype(int)
        
        best_hter = float('inf')
        optimal_threshold = 0.5
        
        for threshold in thresholds:
            pred_binary = (pred_probs >= threshold).astype(int)
            
            # Calculate HTER for this threshold
            tp = ((pred_binary == 1) & (target_binary == 1)).sum()
            fp = ((pred_binary == 1) & (target_binary == 0)).sum()
            tn = ((pred_binary == 0) & (target_binary == 0)).sum()
            fn = ((pred_binary == 0) & (target_binary == 1)).sum()
            
            eps = 1e-8
            far = fp / (fp + tn + eps)
            frr = fn / (fn + tp + eps)
            hter = (far + frr) / 2
            
            if hter < best_hter:
                best_hter = hter
                optimal_threshold = threshold
        
        results[task_name] = {
            'optimal_threshold': optimal_threshold,
            'best_hter': best_hter
        }
    
    return results

def create_data_loaders(train_df, val_df, root_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for train, validation, and test sets
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FacialAttributeDataset(train_df, root_dir, train_transform)
    val_dataset = FacialAttributeDataset(val_df, root_dir, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

def load_and_preprocess_data(csv_path, root_dir='', val_size=0.2, random_state=42):
    """
    Load the dataset and split into train/validation/test sets
    """
    df = pd.read_csv(csv_path)
    
    print(f"Dataset loaded with {len(df)} samples")
    print("Label distributions:")
    for col in ['glasses', 'hat', 'facial_hair', 'label']:
        if col in df.columns:
            print(f"  {col}: {df[col].value_counts().sort_index().to_dict()}")

    # Second split: separate train and validation
    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df['label'], 
        random_state=random_state
    )
    
    return train_df, val_df

def save_checkpoint(model, optimizer, epoch, loss, metric, filepath):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metric': metric
    }
    torch.save(checkpoint, filepath)

def accuracy_per_class(predictions, targets, num_classes=2):
    """
    Calculate accuracy for each class (for compatibility with existing code)
    """
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    accuracies = {}
    for i in range(num_classes):
        class_mask = (target_np == i)
        if class_mask.sum() > 0:
            class_acc = (pred_np[class_mask] == target_np[class_mask]).mean()
            accuracies[f'class_{i}'] = class_acc
        else:
            accuracies[f'class_{i}'] = 0.0
    
    return accuracies

def calculate_hter_metrics(predictions, targets):
    """
    Calculate HTER metrics (for compatibility with existing code)
    """
    tp = ((predictions == 1) & (targets == 1)).sum()
    fp = ((predictions == 1) & (targets == 0)).sum()
    tn = ((predictions == 0) & (targets == 0)).sum()
    fn = ((predictions == 0) & (targets == 1)).sum()
    
    eps = 1e-8
    far = fp / (fp + tn + eps)
    frr = fn / (fn + tp + eps)
    hter = (far + frr) / 2
    
    return {
        'hter': hter,
        'far': far,
        'frr': frr
    }

def find_optimal_threshold(probabilities, targets, num_thresholds=100):
    """
    Find optimal threshold that minimizes HTER (for compatibility)
    """
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_hter = float('inf')
    optimal_threshold = 0.5
    
    for threshold in thresholds:
        pred_binary = (probabilities >= threshold).astype(int)
        
        tp = ((pred_binary == 1) & (targets == 1)).sum()
        fp = ((pred_binary == 1) & (targets == 0)).sum()
        tn = ((pred_binary == 0) & (targets == 0)).sum()
        fn = ((pred_binary == 0) & (targets == 1)).sum()
        
        eps = 1e-8
        far = fp / (fp + tn + eps)
        frr = fn / (fn + tp + eps)
        hter = (far + frr) / 2
        
        if hter < best_hter:
            best_hter = hter
            optimal_threshold = threshold
    
    return optimal_threshold, best_hter, None