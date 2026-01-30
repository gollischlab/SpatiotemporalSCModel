"""Training functions for subunit models.

This module provides PyTorch training utilities for the Logical OR and subunit
models, including:
- Dataset class for stimulus-response pairs
- Pearson correlation metric
- Pruning functions for subunit/weight selection
- Training loops with Group Lasso regularization
"""

import copy
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class SpatialDataset(Dataset):
    """
    Simple PyTorch dataset for (stimulus, response) pairs.

    :param stimulus: Input stimulus array. Shape (n_samples, n_features).
    :type stimulus: np.ndarray
    :param response: Response array (spike counts or binarized responses).
        Shape (n_samples,).
    :type response: np.ndarray
    """

    def __init__(self, stimulus: np.ndarray, response: np.ndarray):
        self.stimulus = torch.tensor(stimulus, dtype=torch.float32)
        self.response = torch.tensor(response, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.response)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.stimulus[idx], self.response[idx]


def pearson_correlation(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Pearson correlation coefficient between two 1D tensors.

    :param x: First tensor. Shape (n,).
    :type x: torch.Tensor
    :param y: Second tensor. Shape (n,).
    :type y: torch.Tensor
    :param eps: Small value for numerical stability. Default is 1e-8.
    :type eps: float

    :return: Scalar correlation coefficient.
    :rtype: torch.Tensor
    """
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    xm = x - x_mean
    ym = y - y_mean
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2).clamp(min=eps))
    return r_num / r_den


def prune_functional_subunits(
    functional_subunits: np.ndarray,
    A_matrices: np.ndarray,
    thresholds: np.ndarray,
    prune_threshold: float = 0.1,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prune functional subunits based on L2 norm threshold.

    After training with Group Lasso regularization, some subunits may have
    very small norms. This function removes subunits whose L2 norm is below
    a threshold relative to the largest subunit.

    :param functional_subunits: Functional subunits computed as A_matrices @ stc_filters.
        Shape (n_subunits, n_features).
    :type functional_subunits: np.ndarray
    :param A_matrices: Learned mixing matrices. Shape (n_subunits, n_filters).
    :type A_matrices: np.ndarray
    :param thresholds: Learned threshold parameters. Shape (n_subunits,).
    :type thresholds: np.ndarray
    :param prune_threshold: Subunits with L2 norm < prune_threshold * max_norm are
        discarded. Default is 0.1 (10% of largest norm).
    :type prune_threshold: float
    :param normalize: If True, normalize pruned subunits to unit L2 norm. Default is True.
    :type normalize: bool

    :return: Tuple of (pruned_subunits, pruned_A_matrices, pruned_thresholds, keep_mask, norms):
        - pruned_subunits: shape (n_kept, n_features)
        - pruned_A_matrices: shape (n_kept, n_filters)
        - pruned_thresholds: shape (n_kept,)
        - keep_mask: shape (n_subunits,), boolean mask of kept subunits
        - norms: shape (n_subunits,), L2 norms of all subunits
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

    :raises ValueError: If all functional subunits have near-zero norm.
    """
    # Compute L2 norm of each subunit
    l2_norms = np.linalg.norm(functional_subunits, axis=1)
    max_norm = np.max(l2_norms)

    # Handle edge case: all norms are zero
    if max_norm < 1e-10:
        raise ValueError(
            "All functional subunits have near-zero norm. "
            "Model may not have converged or regularization is too strong."
        )

    # Determine which subunits to keep
    threshold_value = prune_threshold * max_norm
    keep_mask = l2_norms >= threshold_value

    # Edge case: all subunits pruned - keep the one with largest norm
    if not np.any(keep_mask):
        keep_mask[np.argmax(l2_norms)] = True
        print(
            f"Warning: All subunits below threshold. Keeping the one with "
            f"largest norm (norm={max_norm:.6f})."
        )

    # Apply pruning
    pruned_subunits = functional_subunits[keep_mask].copy()
    pruned_A_matrices = A_matrices[keep_mask].copy()
    pruned_thresholds = thresholds[keep_mask].copy()

    n_original = functional_subunits.shape[0]
    n_kept = pruned_subunits.shape[0]

    # Normalize if requested
    if normalize:
        norms_kept = np.linalg.norm(pruned_subunits, axis=1, keepdims=True)
        pruned_subunits = pruned_subunits / norms_kept

    print(
        f"Pruning: {n_original} -> {n_kept} subunits "
        f"(threshold={threshold_value:.6f}, max_norm={max_norm:.6f})"
    )

    return pruned_subunits, pruned_A_matrices, pruned_thresholds, keep_mask, l2_norms


def prune_subunit_weights(
    weights: np.ndarray,
    functional_subunits: np.ndarray,
    prune_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prune subunit model weights based on relative magnitude.

    For the Liu model with ReLU nonlinearity, pruning is based on scaled
    weights (weight * ||subunit||) since ReLU is homogeneous of degree 1.

    :param weights: Learned subunit weights (non-negative). Shape (n_subunits,).
    :type weights: np.ndarray
    :param functional_subunits: Functional subunits. Shape (n_subunits, n_features).
    :type functional_subunits: np.ndarray
    :param prune_threshold: Weights with scaled value < prune_threshold * max are
        pruned. Default is 0.1 (10% of largest).
    :type prune_threshold: float

    :return: Tuple of (pruned_weights, pruned_subunits, keep_mask):
        - pruned_weights: shape (n_kept,)
        - pruned_subunits: shape (n_kept, n_features)
        - keep_mask: shape (n_subunits,), boolean mask
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]

    :raises ValueError: If all weights are near-zero.
    """
    # Use scaled weights for Liu model (weight * ||subunit||)
    subunit_norms = np.linalg.norm(functional_subunits, axis=1)
    effective_weights = weights * subunit_norms

    max_weight = np.max(effective_weights)

    # Handle edge case: all weights are zero
    if max_weight < 1e-10:
        raise ValueError(
            "All subunit weights are near-zero. "
            "Model may not have converged or L1 regularization is too strong."
        )

    # Determine which weights to keep
    threshold_value = prune_threshold * max_weight
    keep_mask = effective_weights >= threshold_value

    # Edge case: all weights pruned - keep the one with largest weight
    if not np.any(keep_mask):
        keep_mask[np.argmax(effective_weights)] = True
        print(
            f"Warning: All subunit weights below threshold. "
            f"Keeping the one with largest weight (weight={max_weight:.6f})."
        )

    # Apply pruning
    pruned_weights = weights[keep_mask].copy()
    pruned_subunits = functional_subunits[keep_mask].copy()

    n_original = weights.shape[0]
    n_kept = pruned_weights.shape[0]

    print(
        f"Subunit Weight Pruning: {n_original} -> {n_kept} subunits "
        f"(threshold={threshold_value:.6f}, max_weight={max_weight:.6f})"
    )

    return pruned_weights, pruned_subunits, keep_mask


def train_logical_or_model(
    model: nn.Module,
    train_stim: np.ndarray,
    train_resp: np.ndarray,
    val_stim: np.ndarray,
    val_resp: np.ndarray,
    batch_size: int = 512,
    max_epochs: int = 500,
    patience: int = 50,
    learning_rate: float = 5e-3,
    reg_lambda: float = 1e-4,
    warmup_epochs: int = 50,
    device: str = "cuda",
) -> dict:
    """
    Train Logical OR model with Group Lasso regularization.

    Group Lasso regularization applies the sum of L2 norms of functional
    subunits, encouraging sparse subunit selection.

    :param model: LogicalORModel instance.
    :type model: nn.Module
    :param train_stim: Training stimulus. Shape (n_train, n_features).
    :type train_stim: np.ndarray
    :param train_resp: Training responses (binarized for BCE loss). Shape (n_train,).
    :type train_resp: np.ndarray
    :param val_stim: Validation stimulus. Shape (n_val, n_features).
    :type val_stim: np.ndarray
    :param val_resp: Validation responses. Shape (n_val,).
    :type val_resp: np.ndarray
    :param batch_size: Batch size. Default is 512.
    :type batch_size: int
    :param max_epochs: Maximum training epochs. Default is 500.
    :type max_epochs: int
    :param patience: Early stopping patience. Default is 50.
    :type patience: int
    :param learning_rate: Learning rate. Default is 5e-3.
    :type learning_rate: float
    :param reg_lambda: Group Lasso regularization strength. Default is 1e-4.
    :type reg_lambda: float
    :param warmup_epochs: Epochs before regularization kicks in. Default is 50.
    :type warmup_epochs: int
    :param device: Device for training. Default is 'cuda'.
    :type device: str

    :return: Dictionary containing 'model' (trained model with best state restored),
        'history' (training history dict), and 'best_epoch' (best epoch number).
    :rtype: dict
    """
    model.to(device)

    # Create data loaders
    train_dataset = SpatialDataset(train_stim, train_resp)
    val_dataset = SpatialDataset(val_stim, val_resp)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    criterion = nn.BCELoss()

    # Scale lambda by sqrt(n_features / 400)
    n_features = model.n_features
    effective_lambda = reg_lambda * np.sqrt(n_features / 400.0)

    # Tracking
    best_val_loss = np.inf
    best_epoch = -1
    epochs_no_improve = 0
    best_state_dict = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_corr": [],
        "reg_penalty": [],
        "n_active_subunits": [],
    }

    with tqdm(range(max_epochs), unit="epoch") as epoch_bar:
        for epoch in epoch_bar:
            # Training
            model.train()
            epoch_train_loss = []
            epoch_reg_penalty = []

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                y_pred = model(x)
                bce_loss = criterion(y_pred, y)

                # Group Lasso regularization (after warmup)
                if epoch >= warmup_epochs:
                    functional_subunits = model.A_matrices @ model.stc_filters
                    l2_norms = torch.norm(functional_subunits, p=2, dim=1)
                    reg_penalty = torch.sum(l2_norms)
                    total_loss = bce_loss + effective_lambda * reg_penalty
                else:
                    reg_penalty = torch.tensor(0.0, device=device)
                    total_loss = bce_loss

                total_loss.backward()
                optimizer.step()

                epoch_train_loss.append(total_loss.item())
                epoch_reg_penalty.append(reg_penalty.item())

            avg_train_loss = float(np.mean(epoch_train_loss))
            avg_reg_penalty = float(np.mean(epoch_reg_penalty))

            # Validation
            model.eval()
            epoch_val_loss = []
            epoch_val_preds = []
            epoch_val_targets = []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    val_loss = criterion(y_pred, y)

                    epoch_val_loss.append(val_loss.item())
                    epoch_val_preds.append(y_pred)
                    epoch_val_targets.append(y)

                all_preds = torch.cat(epoch_val_preds, dim=0)
                all_targets = torch.cat(epoch_val_targets, dim=0)
                val_corr = pearson_correlation(all_preds, all_targets).item()
                avg_val_loss = float(np.mean(epoch_val_loss))

            scheduler.step(avg_val_loss)

            # Count active subunits
            with torch.no_grad():
                current_subunits = model.A_matrices @ model.stc_filters
                subunit_norms = torch.norm(current_subunits, p=2, dim=1).cpu().numpy()
                n_active = int(np.sum(subunit_norms > 1e-4))

            # Record history
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_corr"].append(val_corr)
            history["reg_penalty"].append(avg_reg_penalty)
            history["n_active_subunits"].append(n_active)

            # Best model tracking (only after warmup)
            if epoch >= warmup_epochs:
                if avg_val_loss < best_val_loss - 1e-6:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    epochs_no_improve = 0
                    best_state_dict = copy.deepcopy(model.state_dict())
                else:
                    epochs_no_improve += 1

            # Progress bar
            warmup_str = " [WARMUP]" if epoch < warmup_epochs else ""
            epoch_bar.set_description(
                f"Epoch {epoch+1}/{max_epochs}{warmup_str} | "
                f"Train: {avg_train_loss:.4f} | "
                f"Val: {avg_val_loss:.4f} | "
                f"Corr: {val_corr:.3f} | "
                f"Active: {n_active} | "
                f"Best Epoch: {best_epoch}"
            )

            # Early stopping
            if epoch >= warmup_epochs and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    else:
        best_epoch = epoch
        print("Warning: No improvement found post-warmup. Using final model state.")

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
    }


def train_subunit_model(
    model: nn.Module,
    train_stim: np.ndarray,
    train_resp: np.ndarray,
    val_stim: np.ndarray,
    val_resp: np.ndarray,
    batch_size: int = 512,
    max_epochs: int = 500,
    patience: int = 50,
    learning_rate: float = 1e-3,
    l1_lambda: float = 1e-4,
    prune_threshold: float = 0.1,
    device: str = "cuda",
) -> dict:
    """
    Two-stage training for subunit model.

    Stage 1: Train with L1 regularization on scaled weights (weight * ||subunit||).
    Pruning: Remove weights below prune_threshold * max_weight.
    Stage 2: Retrain pruned model without regularization.

    :param model: SubunitModel instance with all subunits.
    :type model: nn.Module
    :param train_stim: Training stimulus. Shape (n_train, n_features).
    :type train_stim: np.ndarray
    :param train_resp: Training responses (spike counts). Shape (n_train,).
    :type train_resp: np.ndarray
    :param val_stim: Validation stimulus. Shape (n_val, n_features).
    :type val_stim: np.ndarray
    :param val_resp: Validation responses. Shape (n_val,).
    :type val_resp: np.ndarray
    :param batch_size: Batch size. Default is 512.
    :type batch_size: int
    :param max_epochs: Maximum epochs per stage. Default is 500.
    :type max_epochs: int
    :param patience: Early stopping patience. Default is 50.
    :type patience: int
    :param learning_rate: Learning rate. Default is 1e-3.
    :type learning_rate: float
    :param l1_lambda: L1 regularization strength on weights. Default is 1e-4.
    :type l1_lambda: float
    :param prune_threshold: Pruning threshold (fraction of max weight). Default is 0.1.
    :type prune_threshold: float
    :param device: Device for training. Default is 'cuda'.
    :type device: str

    :return: Dictionary containing 'model' (pruned model with best state),
        'history' (combined training history), 'best_epoch' (global best),
        'stage1_best_epoch', 'stage2_best_epoch', 'prune_epoch',
        'keep_mask' (boolean mask), and 'n_pruned' (number pruned).
    :rtype: dict
    """
    # Import here to avoid circular imports
    from sc_model.models.subunit_model import SubunitModel

    model.to(device)

    # Compute subunit L2 norms (for scaled weight regularization)
    with torch.no_grad():
        subunit_norms = torch.norm(model.functional_subunits, dim=1)

    print("Using scaled weight regularization (weight * ||subunit||)")

    # Create data loaders
    train_dataset = SpatialDataset(train_stim, train_resp)
    val_dataset = SpatialDataset(val_stim, val_resp)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    # Loss function
    criterion = nn.PoissonNLLLoss(log_input=True, full=False)

    # Combined history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_corr": [],
        "weights": [],
        "a": [],
        "b": [],
        "reg_penalty": [],
        "n_active_weights": [],
        "stage": [],
    }

    # =========================================================================
    # STAGE A: Train with L1 regularization
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STAGE A: Training with L1 regularization (lambda={l1_lambda:.0e})")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = np.inf
    best_epoch = -1
    epochs_no_improve = 0
    best_state_dict = None

    with tqdm(range(max_epochs), unit="epoch", desc="Stage 1") as epoch_bar:
        for epoch in epoch_bar:
            model.train()
            epoch_train_loss = []
            epoch_reg_penalty = []

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                y_pred = model(x)
                base_loss = criterion(y_pred.log(), y)

                # L1 on scaled weights
                if l1_lambda > 0:
                    scaled_weights = model.weights * subunit_norms
                    reg_penalty = l1_lambda * torch.sum(scaled_weights)
                else:
                    reg_penalty = torch.tensor(0.0, device=device)

                loss = base_loss + reg_penalty
                loss.backward()
                optimizer.step()

                # Clamp weights to be non-negative
                with torch.no_grad():
                    model.weights.data.clamp_(min=0.0)

                epoch_train_loss.append(loss.item())
                epoch_reg_penalty.append(reg_penalty.item())

            avg_train_loss = float(np.mean(epoch_train_loss))
            avg_reg_penalty = float(np.mean(epoch_reg_penalty))

            # Validation
            model.eval()
            epoch_val_loss = []
            epoch_val_preds = []
            epoch_val_targets = []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    val_loss = criterion(y_pred.log(), y)
                    epoch_val_loss.append(val_loss.item())
                    epoch_val_preds.append(y_pred)
                    epoch_val_targets.append(y)

                all_preds = torch.cat(epoch_val_preds, dim=0)
                all_targets = torch.cat(epoch_val_targets, dim=0)
                val_corr = pearson_correlation(all_preds, all_targets).item()
                avg_val_loss = float(np.mean(epoch_val_loss))

            scheduler.step(avg_val_loss)

            # Count active weights
            effective_weights = (model.weights.detach() * subunit_norms).cpu().numpy()
            max_weight = np.max(effective_weights) if np.max(effective_weights) > 1e-10 else 1e-10
            n_active = int((effective_weights >= prune_threshold * max_weight).sum())

            # Record history
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_corr"].append(val_corr)
            history["weights"].append(model.weights.detach().cpu().numpy().copy())
            history["a"].append(model.a.item())
            history["b"].append(model.b.item())
            history["reg_penalty"].append(avg_reg_penalty)
            history["n_active_weights"].append(n_active)
            history["stage"].append(1)

            # Best model tracking
            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1

            epoch_bar.set_postfix({
                "loss": f"{avg_val_loss:.4f}",
                "corr": f"{val_corr:.3f}",
                "active": n_active,
                "best": best_epoch
            })

            if epochs_no_improve >= patience:
                print(f"Stage A early stopping at epoch {epoch}")
                break

    # Restore best model from stage 1
    stage1_best_epoch = best_epoch
    prune_epoch = len(history["train_loss"]) - 1
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"Stage A complete. Best epoch: {stage1_best_epoch}")

    # =========================================================================
    # PRUNING
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"PRUNING: Removing weights below {prune_threshold*100:.0f}% of max")
    print(f"{'='*60}")

    current_weights = model.weights.detach().cpu().numpy()
    current_subunits = model.functional_subunits.detach().cpu().numpy()

    pruned_weights, pruned_subunits, keep_mask = prune_subunit_weights(
        weights=current_weights,
        functional_subunits=current_subunits,
        prune_threshold=prune_threshold,
    )

    n_original = len(current_weights)
    n_kept = len(pruned_weights)
    n_pruned = n_original - n_kept

    # =========================================================================
    # STAGE B: Retrain with pruned subunits
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STAGE B: Retraining with {n_kept} subunits (no regularization)")
    print(f"{'='*60}")

    # Create new model with pruned subunits
    current_a = model.a.item()
    current_b = model.b.item()

    model = SubunitModel(
        functional_subunits=pruned_subunits,
        a_init=current_a,
        b_init=current_b,
        w_inits=pruned_weights,
        device=device,
    )
    model.to(device)

    # Reset optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = np.inf
    best_epoch_stage2 = -1
    epochs_no_improve = 0
    best_state_dict = None

    with tqdm(range(max_epochs), unit="epoch", desc="Stage 2") as epoch_bar:
        for epoch in epoch_bar:
            model.train()
            epoch_train_loss = []

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred.log(), y)  # No regularization
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.weights.data.clamp_(min=0.0)

                epoch_train_loss.append(loss.item())

            avg_train_loss = float(np.mean(epoch_train_loss))

            # Validation
            model.eval()
            epoch_val_loss = []
            epoch_val_preds = []
            epoch_val_targets = []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    val_loss = criterion(y_pred.log(), y)
                    epoch_val_loss.append(val_loss.item())
                    epoch_val_preds.append(y_pred)
                    epoch_val_targets.append(y)

                all_preds = torch.cat(epoch_val_preds, dim=0)
                all_targets = torch.cat(epoch_val_targets, dim=0)
                val_corr = pearson_correlation(all_preds, all_targets).item()
                avg_val_loss = float(np.mean(epoch_val_loss))

            scheduler.step(avg_val_loss)

            # Record history (pad weights to original size)
            stage2_weights = model.weights.detach().cpu().numpy().copy()
            padded_weights = np.zeros(n_original)
            padded_weights[keep_mask] = stage2_weights

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_corr"].append(val_corr)
            history["weights"].append(padded_weights)
            history["a"].append(model.a.item())
            history["b"].append(model.b.item())
            history["reg_penalty"].append(0.0)
            history["n_active_weights"].append(n_kept)
            history["stage"].append(2)

            # Best model tracking
            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                best_epoch_stage2 = epoch
                epochs_no_improve = 0
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1

            epoch_bar.set_postfix({
                "loss": f"{avg_val_loss:.4f}",
                "corr": f"{val_corr:.3f}",
                "best": best_epoch_stage2
            })

            if epochs_no_improve >= patience:
                print(f"Stage B early stopping at epoch {epoch}")
                break

    # Restore best model from stage 2
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"Stage B complete. Best epoch: {best_epoch_stage2}")

    # Compute global best epoch
    global_best_epoch = (prune_epoch + 1) + best_epoch_stage2

    return {
        "model": model,
        "history": history,
        "best_epoch": global_best_epoch,
        "stage1_best_epoch": stage1_best_epoch,
        "stage2_best_epoch": best_epoch_stage2,
        "prune_epoch": prune_epoch,
        "keep_mask": keep_mask,
        "n_pruned": n_pruned,
    }
