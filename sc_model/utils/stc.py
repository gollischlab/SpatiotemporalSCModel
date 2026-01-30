"""Spike-Triggered Covariance (STC) computation using GPU acceleration.

This module provides PyTorch/GPU implementations of STC methods for computing
spike-triggered covariance and extracting significant eigenvectors. The
implementation uses the Kaardal et al. method which operates on temporally-
convolved stimulus directly.
"""

from typing import Tuple, Optional, Union

import numpy as np
import torch


def kaardal_stc(
    stimulus: Union[np.ndarray, torch.Tensor],
    spike_counts: Union[np.ndarray, torch.Tensor],
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spike-triggered average and covariance using Kaardal et al. 2013 method.

    This method computes STC on temporally-convolved stimulus (i.e., stimulus
    that has already been filtered by the temporal kernel). The eigenvalues
    are sorted by absolute deviation from the raw stimulus covariance.

    :param stimulus: Temporally-convolved stimulus where each row is the flattened
        spatial stimulus at one time point. Shape (n_times, n_features).
    :type stimulus: np.ndarray or torch.Tensor
    :param spike_counts: Spike counts per time bin. Shape (n_times,).
    :type spike_counts: np.ndarray or torch.Tensor
    :param device: Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
    :type device: str

    :return: Tuple of (eigenvalues, eigenvectors, st_avg, raw_cov) where:
        - eigenvalues: shape (n_features,), sorted by absolute value
        - eigenvectors: shape (n_features, n_features), columns match eigenvalues
        - st_avg: shape (n_features,), spike-triggered average
        - raw_cov: shape (n_features, n_features), raw stimulus covariance
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    # Convert to torch tensors
    if isinstance(stimulus, np.ndarray):
        stim_t = torch.from_numpy(stimulus).float().to(device)
    else:
        stim_t = stimulus.float().to(device)

    if isinstance(spike_counts, np.ndarray):
        spikes_t = torch.from_numpy(spike_counts).float().to(device)
    else:
        spikes_t = spike_counts.float().to(device)

    n_timebins = stim_t.shape[0]

    # Spike-triggered ensemble
    spike_mask = spikes_t > 0
    st_ensemble = stim_t[spike_mask]
    spike_weights = spikes_t[spike_mask]
    n_spikes = spike_weights.sum()

    # Spike-triggered average (weighted by spike counts)
    weighted_st_ensemble = st_ensemble * spike_weights.unsqueeze(1)
    st_avg = weighted_st_ensemble.sum(dim=0) / n_spikes

    # Spike-triggered covariance with Bessel correction
    second_moment = (st_ensemble.T @ weighted_st_ensemble) / (n_spikes - 1)
    st_cov = second_moment - torch.outer(st_avg, st_avg) * (n_spikes / (n_spikes - 1))

    # Raw stimulus statistics
    raw_avg = stim_t.mean(dim=0)
    centered_raw = stim_t - raw_avg
    raw_cov = (centered_raw.T @ centered_raw) / (n_timebins - 1)

    # Eigendecomposition of (STC - raw_cov)
    diff_cov = st_cov - raw_cov
    eigenvalues, eigenvectors = torch.linalg.eigh(diff_cov)

    # Sort by absolute eigenvalue (descending)
    abs_eigenvalues = torch.abs(eigenvalues)
    sort_idx = torch.argsort(abs_eigenvalues, descending=True)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Convert to numpy
    eigenvalues_np = eigenvalues.cpu().numpy()
    eigenvectors_np = eigenvectors.cpu().numpy()
    st_avg_np = st_avg.cpu().numpy()
    raw_cov_np = raw_cov.cpu().numpy()

    return eigenvalues_np, eigenvectors_np, st_avg_np, raw_cov_np


def get_top_eigenvectors(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_top: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top-N eigenvectors by absolute eigenvalue magnitude.

    This function selects the eigenvectors corresponding to the eigenvalues
    with the largest absolute deviation from the raw covariance (already
    sorted if using kaardal_stc output).

    :param eigenvalues: Eigenvalues (should be sorted by absolute value if from
        kaardal_stc). Shape (n_features,).
    :type eigenvalues: np.ndarray
    :param eigenvectors: Corresponding eigenvectors (columns). Shape (n_features, n_features).
    :type eigenvectors: np.ndarray
    :param n_top: Number of top eigenvectors to select.
    :type n_top: int

    :return: Tuple of (top_eigenvectors, top_eigenvalues) where:
        - top_eigenvectors: shape (n_top, n_features), selected eigenvectors as rows
        - top_eigenvalues: shape (n_top,), corresponding eigenvalues
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    if n_top > len(eigenvalues):
        raise ValueError(
            f"Requested n_top={n_top} but only {len(eigenvalues)} eigenvalues available."
        )

    # Take top N (assuming already sorted by absolute value)
    top_eigenvalues = eigenvalues[:n_top]
    top_eigenvectors = eigenvectors[:, :n_top].T  # (n_top, n_features)

    return top_eigenvectors, top_eigenvalues


def compute_stc_for_cell(
    stimulus: np.ndarray,
    spike_counts: np.ndarray,
    n_top: int = 16,
    device: str = "cuda",
) -> dict:
    """
    Compute STC and extract top eigenvectors for a single cell.

    This is a convenience function that combines kaardal_stc and
    get_top_eigenvectors for typical usage.

    :param stimulus: Temporally-convolved stimulus. Shape (n_times, n_features).
    :type stimulus: np.ndarray
    :param spike_counts: Spike counts per time bin. Shape (n_times,).
    :type spike_counts: np.ndarray
    :param n_top: Number of top eigenvectors to extract. Default is 16.
    :type n_top: int
    :param device: Device for computation. Default is 'cuda'.
    :type device: str

    :return: Dictionary containing 'eigenvectors' (n_top, n_features),
        'eigenvalues' (n_top,), 'st_avg' (n_features,), 'all_eigenvalues'
        (n_features,), and 'all_eigenvectors' (n_features, n_features).
    :rtype: dict
    """
    # Compute full STC
    eigenvalues, eigenvectors, st_avg, raw_cov = kaardal_stc(
        stimulus, spike_counts, device=device
    )

    # Extract top eigenvectors
    top_eigenvectors, top_eigenvalues = get_top_eigenvectors(
        eigenvalues, eigenvectors, n_top
    )

    return {
        "eigenvectors": top_eigenvectors,
        "eigenvalues": top_eigenvalues,
        "st_avg": st_avg,
        "all_eigenvalues": eigenvalues,
        "all_eigenvectors": eigenvectors,
    }
