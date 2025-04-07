from typing import Union

import numpy as np


def vectorized_softplus(
    x: np.ndarray,
    params: Union[list, np.ndarray],
):
    a = params[0]
    b = params[1]
    w = np.array([params[2:]])
    return a * np.log(1. + np.exp(w @ x + b))[0]


def vectorized_softplus_derivative(
    x: np.ndarray,
    params: Union[list, np.ndarray],
):
    a = params[0]
    b = params[1]
    w = np.array([params[2:]])

    inner_exp = np.exp(w @ x + b)
    inner_exp_p1 = 1. + inner_exp
    inner_log = np.log(inner_exp_p1)

    der_a = inner_log

    der_b = a * inner_exp / inner_exp_p1

    der_w = der_b * x

    return np.vstack([der_a, der_b, der_w]).T
