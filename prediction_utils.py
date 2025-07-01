import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Utils.utils import TimingCallback, plot_alarmDF
from Utils.metrics_errors import point_metrics_with_nans, picp, mpiw, pinaw

# === Environment Setup === #
# Add root path for relative imports if running from subdirectories
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# === Debugging Utility === #
def log_gradient_norms(model):
    """
    Log the L2 norm of gradients for all model parameters.
    Useful for debugging vanishing or exploding gradients.
    """
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            print(f'{name} gradient norm: {param_norm.item():.6f}')
    print(f'Total gradient norm: {total_norm ** 0.5:.6f}\n')


# === Loss Function === #
def nll_loss(mu, log_var, y):
    """
    Compute Negative Log-Likelihood loss assuming Gaussian uncertainty.
    
    Args:
        mu (Tensor): Predicted mean
        log_var (Tensor): Predicted log-variance
        y (Tensor): Ground truth targets
    
    Returns:
        Tensor: Total NLL loss across batch
    """
    mu = mu.to(y.device)
    log_var = log_var.to(y.device)
    loss = 0.5 * torch.exp(-log_var) * (y - mu) ** 2 + 0.5 * log_var
    return loss.sum()  # <- this ensures it's a scalar


# === Evaluation === #
def test_trained_model(model, test_loader):
    """
    Evaluate a trained probabilistic model on a test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
    
    Returns:
        pd.DataFrame: Combined pointwise and probabilistic metrics
    """
    model.eval()
    lower_preds, mu_preds, upper_preds, y_true = [], [], [], []
    batches_skipped = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Skip batch if it contains NaNs to avoid invalid computations
            if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                batches_skipped += 1
                nan_tensor = torch.full((y_batch.shape[0],), float('nan'))
                y_true.extend(nan_tensor.tolist())
                lower_preds.extend(nan_tensor.tolist())
                mu_preds.extend(nan_tensor.tolist())
                upper_preds.extend(nan_tensor.tolist())
                continue

            # Forward pass
            lower, mu, upper, log_var = model(X_batch)
            y_true.extend(y_batch.view(-1).tolist())
            lower_preds.extend(lower.view(-1).tolist())
            mu_preds.extend(mu.view(-1).tolist())
            upper_preds.extend(upper.view(-1).tolist())

    # Convert to numpy arrays for metric computation
    y_true = np.asarray(y_true)
    lower_preds = np.asarray(lower_preds)
    mu_preds = np.asarray(mu_preds)
    upper_preds = np.asarray(upper_preds)

    # Pointwise metrics (e.g. MAE, RMSE, etc.)
    point_metrics = point_metrics_with_nans(mu_preds, y_true, total_p=100, norm=True)

    # Probabilistic metrics
    pr_metrics = pd.DataFrame({
        'METRIC': ['PICP', 'MPIW'],
        'VALUE': [
            picp(y_true, lower_preds, upper_preds),
            mpiw(y_true, lower_preds, upper_preds, False, 100, 0)
        ]
    })

    return pd.concat([point_metrics, pr_metrics], ignore_index=True)

# === Prediction === #
def pred_trained_model(model, test_loader):
    """
    make predictions based on trained mdoel
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
    
    Returns:
        pd.DataFrame: y_true, lower_preds,mu_preds,upper_preds
    """
    model.eval()
    lower_preds, mu_preds, upper_preds, y_true = [], [], [], []
    batches_skipped = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Skip batch if it contains NaNs to avoid invalid computations
            if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                batches_skipped += 1
                nan_tensor = torch.full((y_batch.shape[0],), float('nan'))
                y_true.extend(nan_tensor.tolist())
                lower_preds.extend(nan_tensor.tolist())
                mu_preds.extend(nan_tensor.tolist())
                upper_preds.extend(nan_tensor.tolist())
                continue

            # Forward pass
            lower, mu, upper, log_var = model(X_batch)
            y_true.extend(y_batch.view(-1).tolist())
            lower_preds.extend(lower.view(-1).tolist())
            mu_preds.extend(mu.view(-1).tolist())
            upper_preds.extend(upper.view(-1).tolist())

    # Convert to numpy arrays for metric computation
    y_true = np.asarray(y_true)
    lower_preds = np.asarray(lower_preds)
    mu_preds = np.asarray(mu_preds)
    upper_preds = np.asarray(upper_preds)


    return y_true, lower_preds, mu_preds, upper_preds
    

# === Training and Validation === #
def train_valid_model_fast(model, criterion, optimizer, train_loader, valid_loader, num_epochs, clipping=False):
    """
    Train a probabilistic model and evaluate on validation set after each epoch.

    Args:
        model: PyTorch model
        criterion: Loss function
        optimizer: Optimizer
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        num_epochs (int): Number of training epochs
        clipping (bool): If True, apply gradient clipping
    
    Returns:
        tuple: (Best model weights, best validation loss)
    """
    callback = TimingCallback()
    best_valid_loss = float('inf')
    best_model_valid = None

    for epoch in range(num_epochs):
        model.train()
        callback.on_epoch_begin(epoch)

        # === Training === #
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)

            # Skip NaN-containing batches
            if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                continue

            # Forward + backward pass
            lower, mu, upper, log_var = model(X_batch)
            loss = nll_loss(mu, log_var, y_batch.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()

            if clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # === Validation === #
        model.eval()
        total_val_loss = 0
        total_abs_error = 0
        processed_samples = 0
        y_true_valid, lower_preds_valid, mu_preds_valid, upper_preds_valid = [], [], [], []

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)

                if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                    nan_tensor = torch.full((y_batch.shape[0],), float('nan'))
                    y_true_valid.extend(nan_tensor.tolist())
                    lower_preds_valid.extend(nan_tensor.tolist())
                    mu_preds_valid.extend(nan_tensor.tolist())
                    upper_preds_valid.extend(nan_tensor.tolist())
                    continue

                lower, mu, upper, log_var = model(X_batch)
                val_loss = nll_loss(mu, log_var, y_batch.unsqueeze(1))
                total_val_loss += val_loss.item()

                total_abs_error += torch.sum(torch.abs(y_batch - mu.squeeze(1)))
                processed_samples += len(X_batch)

                y_true_valid.extend(y_batch.view(-1).tolist())
                lower_preds_valid.extend(lower.view(-1).tolist())
                mu_preds_valid.extend(mu.view(-1).tolist())
                upper_preds_valid.extend(upper.view(-1).tolist())

        # Avoid division by zero
        mean_val_loss = total_val_loss / processed_samples if processed_samples else float('inf')

        # Periodic logging
        if (epoch + 1) % 20 == 0 or epoch == 1:
            callback.on_epoch_end(epoch)
            print(f'Epoch {epoch+1}/{num_epochs}: best validation loss so far: {best_valid_loss:.4f}')

        # Save best model
        if mean_val_loss < best_valid_loss:
            best_valid_loss = mean_val_loss
            best_model_valid = model.state_dict()

    print(f'Training completed. Best validation loss: {best_valid_loss:.4f}')
    return best_model_valid, best_valid_loss
