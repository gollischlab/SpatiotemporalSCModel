"""Logical OR model for learning functional subunits from STC filters.

This module implements the Logical OR model that learns to combine STC
eigenvectors into functional subunits. The model uses a soft OR operation
implemented via: 1 - prod(1 - sigmoid(A @ STC @ x + threshold))

The learned mixing matrices (A_matrices) combine STC filters to form
functional subunits that can then be used in the LNLN model.
"""

from typing import Union

import numpy as np
import torch
import torch.nn as nn


class LogicalORModel(nn.Module):
    """
    Logical OR model that learns to combine STC filters into functional subunits.

    The model computes:
        1 - prod(1 - sigmoid(A @ STC @ x + threshold))

    This implements a soft logical OR operation where each subunit can fire
    if any of its constituent STC filters is activated.

    :param stc_filters: Fixed STC eigenvectors (not trainable).
        Shape (n_filters, n_features).
    :type stc_filters: np.ndarray or torch.Tensor
    :param n_subunits: Number of functional subunits to learn.
    :type n_subunits: int
    :param device: Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
    :type device: str

    :ivar stc_filters: Fixed STC eigenvectors, shape (n_filters, n_features).
    :ivar n_filters: Number of STC filters.
    :ivar n_features: Number of features per filter.
    :ivar n_subunits: Number of subunits to learn.
    :ivar A_matrices: Trainable mixing coefficients, shape (n_subunits, n_filters).
    :ivar thresholds: Trainable thresholds, shape (n_subunits,).
    """

    def __init__(
        self,
        stc_filters: Union[np.ndarray, torch.Tensor],
        n_subunits: int,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # Store fixed STC filters (not trainable)
        self.stc_filters = torch.tensor(
            stc_filters, dtype=torch.float32, device=device
        )
        self.n_filters = self.stc_filters.shape[0]
        self.n_features = self.stc_filters.shape[1]
        self.n_subunits = n_subunits

        # Trainable parameters
        # A_matrices: mixing coefficients to combine STC filters
        self.A_matrices = nn.Parameter(
            torch.randn(n_subunits, self.n_filters, device=device)
        )

        # Thresholds for each subunit
        self.thresholds = nn.Parameter(
            torch.randn(n_subunits, device=device)
        )

        # Sigmoid activation
        self.logistic = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Logical OR model.

        :param x: Input stimulus (temporally-convolved spatial features).
            Shape (batch_size, n_features).
        :type x: torch.Tensor

        :return: Output probability (between 0 and 1). Shape (batch_size,).
        :rtype: torch.Tensor
        """
        # Compute functional subunits: A_matrices @ stc_filters
        # Shape: (n_subunits, n_features)
        functional_subunits = self.A_matrices @ self.stc_filters

        # Linear activation: functional_subunits @ x^T + thresholds
        # Shape: (batch_size, n_subunits)
        lin_activation = (
            torch.inner(functional_subunits, x) + self.thresholds.unsqueeze(1)
        ).T

        # Apply soft OR: 1 - prod(1 - sigmoid(activation))
        nonlin_activation = 1.0 - self.logistic(lin_activation)  # (batch, n_subunits)
        output = 1.0 - torch.prod(nonlin_activation, dim=1)  # (batch,)

        return output

    def get_thresholds(self) -> np.ndarray:
        """
        Get current threshold values as numpy array.

        :return: Current threshold values. Shape (n_subunits,).
        :rtype: np.ndarray
        """
        return self.thresholds.detach().cpu().numpy()

    def get_A_matrices(self) -> np.ndarray:
        """
        Get current mixing coefficients as numpy array.

        :return: Current mixing coefficient values. Shape (n_subunits, n_filters).
        :rtype: np.ndarray
        """
        return self.A_matrices.detach().cpu().numpy()

    def get_functional_subunits(self) -> np.ndarray:
        """
        Compute and return functional subunits as numpy array.

        The functional subunits are computed as A_matrices @ stc_filters.

        :return: Functional subunit filters. Shape (n_subunits, n_features).
        :rtype: np.ndarray
        """
        return (self.A_matrices @ self.stc_filters).detach().cpu().numpy()
