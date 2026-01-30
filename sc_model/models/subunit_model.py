"""Subunit model with ReLU subunit nonlinearity and parametric softplus output.

This module implements the subunit model variant from Liu et al., which uses:
- ReLU as the subunit nonlinearity
- Parametric softplus as the output nonlinearity

The ReLU nonlinearity is homogeneous of degree 1 (ReLU(c*x) = c*ReLU(x)),
which makes weight-norm scaling exact and resolves scaling issues when
subunits are not normalized.
"""

from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn


class SubunitModel(nn.Module):
    """
    Subunit model with ReLU subunit nonlinearity and parametric softplus output.

    Model architecture:
        input -> (W @ subunits)^T -> ReLU -> weighted sum -> softplus output

    Subunit nonlinearity: ReLU(x) = max(0, x)
    Output nonlinearity: a * log(1 + exp(x - b)) (parametric softplus)

    :param functional_subunits: Fixed spatial subunit filters (not trainable).
        Shape (n_subunits, n_features). Typically derived from STC eigenvectors
        via Logical OR model training.
    :type functional_subunits: np.ndarray or torch.Tensor
    :param a_init: Initial gain for output softplus. Default is 1.0.
    :type a_init: float
    :param b_init: Initial threshold for output softplus. Default is 0.5.
    :type b_init: float
    :param w_inits: Initial weights for each subunit. Shape (n_subunits,).
        Defaults to ones if not provided.
    :type w_inits: np.ndarray or torch.Tensor, optional
    :param tune_weights: Whether to make subunit weights trainable. Default is True.
    :type tune_weights: bool
    :param device: Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
    :type device: str

    :ivar functional_subunits: Fixed subunit filters, shape (n_subunits, n_features).
    :ivar n_subunits: Number of subunits.
    :ivar n_features: Number of features per subunit.
    :ivar weights: Trainable weights for each subunit.
    :ivar a: Trainable gain for output nonlinearity.
    :ivar b: Trainable threshold for output nonlinearity.
    """

    def __init__(
        self,
        functional_subunits: Union[np.ndarray, torch.Tensor],
        a_init: float = 1.0,
        b_init: float = 0.5,
        w_inits: Optional[Union[np.ndarray, torch.Tensor]] = None,
        tune_weights: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # Store fixed subunits (not trainable)
        self.functional_subunits = torch.tensor(
            functional_subunits, dtype=torch.float32, device=device, requires_grad=False
        )
        self.n_subunits = self.functional_subunits.shape[0]
        self.n_features = self.functional_subunits.shape[1]

        # Trainable parameters
        if w_inits is not None:
            w_init_values = np.asarray(w_inits)
        else:
            w_init_values = np.ones(self.n_subunits)

        self.weights = nn.Parameter(
            torch.tensor(
                w_init_values,
                device=device,
                requires_grad=tune_weights,
                dtype=torch.float32,
            )
        )

        self.a = nn.Parameter(
            torch.tensor([a_init], device=device, requires_grad=True, dtype=torch.float32)
        )
        self.b = nn.Parameter(
            torch.tensor([b_init], device=device, requires_grad=True, dtype=torch.float32)
        )

    def subunit_nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ReLU nonlinearity to subunit activations.

        :param x: Input tensor.
        :type x: torch.Tensor

        :return: ReLU(x) = max(0, x)
        :rtype: torch.Tensor
        """
        return torch.relu(x)

    def output_nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply parametric softplus output nonlinearity.

        :param x: Input tensor (summed subunit activations).
        :type x: torch.Tensor

        :return: a * softplus(x - b) = a * log(1 + exp(x - b))
        :rtype: torch.Tensor
        """
        return self.a * torch.nn.functional.softplus(x - self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the subunit model.

        :param x: Input stimulus (temporally-convolved spatial features).
            Shape (batch_size, n_features).
        :type x: torch.Tensor

        :return: Predicted firing rates. Shape (batch_size,).
        :rtype: torch.Tensor
        """
        # Linear projection onto subunits
        lin_activations = torch.matmul(x, self.functional_subunits.T)  # (batch, n_subunits)

        # Apply subunit nonlinearity and weights
        subunit_activations = self.weights[None, :] * self.subunit_nonlinearity(lin_activations)

        # Sum over subunits
        subunit_sum = torch.sum(subunit_activations, dim=1)  # (batch,)

        # Apply output nonlinearity
        output = self.output_nonlinearity(subunit_sum)

        return output

    def get_weights(self) -> np.ndarray:
        """
        Get current subunit weights as numpy array.

        :return: Current weight values. Shape (n_subunits,).
        :rtype: np.ndarray
        """
        return self.weights.detach().cpu().numpy()

    def get_subunits(self) -> np.ndarray:
        """
        Get functional subunits as numpy array.

        :return: Functional subunit filters. Shape (n_subunits, n_features).
        :rtype: np.ndarray
        """
        return self.functional_subunits.detach().cpu().numpy()

    def get_output_params(self) -> dict:
        """
        Get output nonlinearity parameters.

        :return: Dictionary with 'a' (gain) and 'b' (threshold) values.
        :rtype: dict
        """
        return {
            "a": self.a.item(),
            "b": self.b.item(),
        }
