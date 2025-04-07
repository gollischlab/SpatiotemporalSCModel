from typing import Optional

import numpy as np
from numba import njit
from scipy.optimize import curve_fit


@njit(parallel=True)
def set_pixels_outside_ellipse_to_zero(
    fitted_gaussian: np.ndarray,
    fit_parameters: np.ndarray,
    sigma_window: Optional[float] = 3.0,
) -> np.ndarray:
    """
    Set pixels outside the n-sigma window of a fitted Gaussian to zero.
    This function modifies the input array in place.

    :param fitted_gaussian: The array representing the fitted Gaussian. This array is modified in place.
    :type fitted_gaussian: np.ndarray
    :param fit_parameters: The parameters of the fitted Gaussian (amplitude, xo, yo, sigma_x, sigma_y, theta).
    :type fit_parameters: np.ndarray
    :param sigma_window: The number of standard deviations to use for the contour.
    :type sigma_window: float, optional

    :return: The modified fitted Gaussian array with pixels outside the n-sigma window set to zero.
    :rtype: np.ndarray
    """
    amplitude, xo, yo, sigma_x, sigma_y, theta = fit_parameters

    # Create a mask for pixels outside the n-sigma window
    sigma_x_contour = sigma_x * sigma_window
    sigma_y_contour = sigma_y * sigma_window

    # Coordinates in the original image
    x, y = meshgrid(np.arange(fitted_gaussian.shape[1]), np.arange(fitted_gaussian.shape[0]))

    # Rotate coordinates to align with ellipse orientation
    x_rot = (x - xo) * np.cos(-theta) + (y - yo) * np.sin(-theta)
    y_rot = -(x - xo) * np.sin(-theta) + (y - yo) * np.cos(-theta)

    # Apply the mask to set pixels outside the n-sigma window to zero
    for i in range(fitted_gaussian.shape[0]):
        for j in range(fitted_gaussian.shape[1]):
            if (x_rot[i, j]**2 / sigma_x_contour**2 + y_rot[i, j]**2 / sigma_y_contour**2) > 1:
                fitted_gaussian[i, j] = 0

    return fitted_gaussian


@njit
def gaussian_2d(
    xy: np.ndarray,
    amplitude: float,
    xo: float,
    yo: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
) -> np.ndarray:
    """
    2D Gaussian function.
    This function computes the value of a 2D Gaussian function at given coordinates.
    The Gaussian is defined by its amplitude, center coordinates (xo, yo),
    standard deviations (sigma_x, sigma_y), and rotation angle (theta).

    :param xy: The coordinates at which to evaluate the Gaussian function.
        The coordinates should be a 2D array with shape (2, N), where N is the number of points.
        The first row contains the x-coordinates, and the second row contains the y-coordinates.
    :type xy: np.ndarray
    :param amplitude: The amplitude of the Gaussian function.
    :type amplitude: float
    :param xo: The x-coordinate of the center of the Gaussian.
    :type xo: float
    :param yo: The y-coordinate of the center of the Gaussian.
    :type yo: float
    :param sigma_x: The standard deviation of the Gaussian in the x-direction.
    :type sigma_x: float
    :param sigma_y: The standard deviation of the Gaussian in the y-direction.
    :type sigma_y: float
    :param theta: The rotation angle of the Gaussian in radians.
    :type theta: float

    :return: The value of the Gaussian function at the given coordinates.
    :rtype: np.ndarray
    """
    x, y = xy[0], xy[1]
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = amplitude * np.exp(-(a * (x - xo) ** 2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo) ** 2))
    return g.ravel()


def fit_gauss2d(
    spatial_kernel: np.ndarray,
    sigma_window: (float, None) = None,
    initial_guesses: (tuple, None) = None,
    bounds: (tuple, None) = None,
) -> tuple:
    """
    Fit a 2D Gaussian to the given spatial kernel using non-linear least squares optimization.
    This function uses the `curve_fit` function from `scipy.optimize` to perform the fitting.
    The Gaussian function is defined by its amplitude, center coordinates (xo, yo),
    standard deviations (sigma_x, sigma_y), and rotation angle (theta).
    The function also allows for setting pixels outside a specified n-sigma window to zero.

    :param spatial_kernel: The spatial kernel to which the Gaussian fit is applied. This should be
        a 2D array representing the spatial kernel.
    :type spatial_kernel: np.ndarray
    :param sigma_window: The number of standard deviations to use for the contour.
        If None, no contouring is applied.
    :type sigma_window: float, optional
    :param initial_guesses: Initial guesses for the Gaussian parameters
        (amplitude, xo, yo, sigma_x, sigma_y, theta). If None, default values are used.
    :type initial_guesses: tuple, optional
    :param bounds: Bounds for the Gaussian parameters. If None, default bounds are used.
        The bounds should be specified as a tuple of two lists, where each list contains the lower
        and upper bounds for each parameter.
        For example, ([-np.inf, -5, -5, 0, 0, 0], [np.inf, 5, 5, np.inf, np.inf, np.pi])
        specifies that the first parameter is unbounded, the second and third parameters are bounded
        between -5 and 5, the fourth and fifth parameters are unbounded, and the sixth parameter is
        bounded between 0 and pi.
    :type bounds: tuple, optional

    :return: A tuple containing the fitted Gaussian parameters, the covariance matrix of the fit,
        and the fitted Gaussian array. The fitted Gaussian array is reshaped to match the shape of
        the input spatial kernel.
    :rtype: tuple
    """
    x_dim = spatial_kernel.shape[1]
    y_dim = spatial_kernel.shape[0]
    x = np.arange(0, x_dim)
    y = np.arange(0, y_dim)
    xx, yy = np.meshgrid(x, y)
    xdata = np.vstack((xx.ravel(), yy.ravel()))

    if initial_guesses is None:
        initial_guesses = (np.abs(spatial_kernel).max(), x_dim // 2, y_dim // 2, 5.0, 5.0, np.pi / 4)

    if bounds is None:
        bounds = (
            [-np.inf, (x_dim // 2) - 5, (y_dim // 2) - 5, 0        , 0        , 0        ],
            [np.inf , (x_dim // 2) + 5, (y_dim // 2) + 5, x_dim / 2, y_dim / 2, np.pi],
        )

    to_fit = spatial_kernel.ravel()

    # Do the fit!
    # noinspection PyTupleAssignmentBalance
    fit_params, covmat = curve_fit(
        gaussian_2d,
        xdata,
        to_fit,
        p0=initial_guesses,
        bounds=bounds,
    )
    fitted_gaussian = gaussian_2d(xdata, *fit_params)
    if sigma_window is not None:
        fitted_gaussian = set_pixels_outside_ellipse_to_zero(
            fitted_gaussian=fitted_gaussian.reshape((y_dim, x_dim)),
            fit_parameters=np.asarray(fit_params),
            sigma_window=sigma_window,
        )

    return fit_params, covmat, fitted_gaussian.reshape((y_dim, x_dim))


@njit
def meshgrid(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    2D meshgrid function.

    2D meshgrid function that creates a grid of coordinates from the given x and y arrays.
    This function is a Numba-compatible version of the numpy.meshgrid function.
    It creates a grid of coordinates for 2D plotting or computations.

    :param x: The x-coordinates of the grid.
    :type x: np.ndarray
    :param y: The y-coordinates of the grid.
    :type y: np.ndarray

    :return: A tuple containing two 2D arrays:
        - xx: The x-coordinates of the grid.
        - yy: The y-coordinates of the grid.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    xx = np.empty(shape=(y.size, x.size), dtype=x.dtype)
    yy = np.empty(shape=(y.size, x.size), dtype=y.dtype)
    for i in range(y.size):
        for j in range(x.size):
            xx[i, j] = x[j]
            yy[i, j] = y[i]
    return xx, yy
