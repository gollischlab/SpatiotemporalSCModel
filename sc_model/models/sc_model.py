from typing import Union, Optional

import numpy as np
from scipy.stats import pearsonr

from sc_model.dataio import WhitenoiseLoader, NaturalMovieLoader
from sc_model.utils.gaussian import fit_gauss2d
from sc_model.utils.receptive_fields import get_spat_temp_filt
from sc_model.utils.convolutions import convolve_stimulus_with_kernels_for_sc
from sc_model.utils.minimization import fit_parameters_mle


def sc_model_single_cell(
    train_stimuli: Union[WhitenoiseLoader, NaturalMovieLoader, np.ndarray],
    test_stimuli: np.ndarray,
    train_responses: np.ndarray,
    ground_truth: np.ndarray,
    cell_sta: np.ndarray,
    temporal_crop: int,
    spatial_crop: int,
    fit_func: callable,
    fit_func_der: callable,
    starter_params: Union[np.ndarray, list],
    fit_bounds: Optional[list] = None,
    mle_method: str = "L-BFGS-B",
    sigma_window: Optional[float] = 3.0,
    check_gradient: bool = False,
    stimulus_smoothing: Optional[float] = None,
    sta_method: str = "sigpix",
    sigpix_threshold: float = 6.0,
    z_score_signals: bool = True,
):
    """
    Fit the spatial contrast model to the data using maximum likelihood estimation.

    This function fits the spatial contrast model to data using maximum likelihood estimation (MLE).
    The model computes the mean luminosity signal (imean) and the local spatial contrast (lsc) signal
    within the receptive field of the cell and linearly combines them before applying a non-linear
    activation function. The receptive field is obtained from the cell's spike-triggered average (STA)
    and is represented by a spatial and temporal filter. A Gaussian function is fitted to the spatial
    filter, and the model parameters are optimized using MLE. The model is trained on the training
    stimuli and responses, and the performance is evaluated on the test stimuli.

    The function returns the fitted parameters, the r-value of the fit, and the convolved signals for
    both the training and test stimuli. The model also allows for the option to z-score the convolved
    signals and smooth the stimulus by convolving it with a Gaussian kernel of a specified smoothing
    radius (`stimulus_smoothing`). The function also allows for the specification of the method used to
    fit the parameters, the bounds for the parameters, and whether to check the gradient during optimization.

    :param train_stimuli: The stimuli used for training the model. This can be a WhitenoiseLoader or
        NaturalMovieLoader object, or a numpy array.
    :type train_stimuli: WhitenoiseLoader, NaturalMovieLoader, np.ndarray
    :param test_stimuli: The stimuli used for testing the model.
    :type test_stimuli: np.ndarray
    :param train_responses: The responses to the training stimuli.
    :type train_responses: np.ndarray
    :param ground_truth: The responses to the test stimuli averaged across trials, used for evaluation.
    :type ground_truth: np.ndarray
    :param cell_sta: The spike-triggered average (STA) of the cell.
    :type cell_sta: np.ndarray
    :param temporal_crop: The size of the temporal crop.
    :type temporal_crop: int
    :param spatial_crop: The size of the square spatial crop, specified as the number of pixels around
        the receptive field center.
    :type spatial_crop: int
    :param fit_func: The non-linear activation function to be used in the model. The function should take
        the input signal and parameters as arguments and return the output signal. It should be vectorized
        to handle multiple inputs.
    :type fit_func: callable
    :param fit_func_der: The derivative of the non-linear activation function with respect to the
        parameters to be optimized. The function should take the input signal and parameters as arguments
        and return the derivative of the output signal with respect to the parameters. It should be vectorized
        to handle multiple inputs.
    :type fit_func_der: callable
    :param starter_params: The initial parameters for the optimization. This can be a numpy array or a list.
    :type starter_params: np.ndarray, list
    :param fit_bounds: The bounds for the parameters to be optimized. If None, no bounds are applied.
        The bounds should be specified as a list of tuples, where each tuple contains the lower and upper
        bounds for each parameter. For example, [(0, 1), (0, None), (0, None)] specifies that the first
        parameter is bounded between 0 and 1, while the second and third parameters are unbounded.
    :type fit_bounds: tuple, optional
    :param mle_method: The method to be used for optimization. Default is "L-BFGS-B", which is a
        limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm with bounds. Other methods can be
        specified as needed.
    :type mle_method: str
    :param sigma_window: The number of standard deviations to use for the contour when fitting the
        Gaussian function to the spatial filter. Default is 3.0.
    :type sigma_window: float, optional
    :param check_gradient: Whether to check the gradient during optimization. Default is False.
        If True, the gradient is checked using finite differences. This can be useful for debugging
        purposes, but it may slow down the optimization process.
    :type check_gradient: bool
    :param stimulus_smoothing: The standard deviation for the Gaussian kernel used to smooth the stimulus.
        If None, no smoothing is applied. Default is None.
    :type stimulus_smoothing: float, optional
    :param sta_method: The method to use for computing the spatial and temporal kernels from the STA.
        Can be 'mis' (most important stixel) or 'sigpix' (significant pixels). Default is 'sigpix'.
    :type sta_method: str
    :param sigpix_threshold: The threshold for selecting significant pixels when using the 'sigpix' method.
        Default is 6.0.
    :type sigpix_threshold: float
    :param z_score_signals: Whether to z-score the convolved signals. Default is True. If True, the mean
        and standard deviation of the training signals are used to z-score both the training and test signals.
    :type z_score_signals: bool

    :return: A dictionary containing the following keys:
            - 'fit_parameters': The fitted parameters of the model.
            - 'r': The r-value of the fit to the test stimuli.
            - 'train_imean': The mean luminosity signal for the training stimuli.
            - 'train_lsc': The local spatial contrast signal for the training stimuli.
            - 'test_imean': The mean luminosity signal for the test stimuli.
            - 'test_lsc': The local spatial contrast signal for the test stimuli.
            - 'fitted_gaussian': The fitted Gaussian function to the spatial filter.
            - 'prediction': The predicted responses to the test stimuli.
            - 'stimulus_smoothing': The standard deviation for the Gaussian kernel used to smooth the stimulus.
            - 'spatial_crop': The size of the spatial crop.
            - 'temporal_crop': The size of the temporal crop.
            - 'mean_imean': The mean of the training mean luminosity signal (if z-scoring is applied).
            - 'mean_lsc': The mean of the training local spatial contrast signal (if z-scoring is applied).
            - 'std_imean': The standard deviation of the training mean luminosity signal (if z-scoring is applied).
            - 'std_lsc': The standard deviation of the training local spatial contrast signal (if z-scoring is applied).
    :rtype: dict
    """
    # get the spatial and temporal kernels from cell STA
    spatial_filter, temporal_filter = get_spat_temp_filt(
        sta=cell_sta,
        spatial_crop_size=spatial_crop,
        temporal_crop_size=temporal_crop,
        force_crop_size=False,
        method=sta_method,
        sigpix_threshold=sigpix_threshold,
    )

    # compute the gaussian fit on the spatial kernel
    gaussian_params, _, fitted_gaussian = fit_gauss2d(
        spatial_kernel=spatial_filter,
        sigma_window=sigma_window,
    )

    # convolve running noise with spatial and temporal stas
    train_i_mean, train_lsc = convolve_stimulus_with_kernels_for_sc(
        stimulus=train_stimuli,
        spatial_filter=fitted_gaussian,
        temporal_filter=temporal_filter,
        total_trials=train_responses.shape[0],
        stimulus_smoothing=stimulus_smoothing,
    )

    # convolve frozen noise with spatial and temporal kernels
    test_i_mean, test_lsc = convolve_stimulus_with_kernels_for_sc(
        stimulus=test_stimuli,
        spatial_filter=fitted_gaussian,
        temporal_filter=temporal_filter,
        total_trials=test_stimuli.shape[0],
        stimulus_smoothing=stimulus_smoothing,
    )

    if z_score_signals:
        mean_imean, mean_lsc = train_i_mean.mean(), train_lsc.mean()
        std_imean, std_lsc = train_i_mean.std(), train_lsc.std()
        train_i_mean = (train_i_mean - mean_imean) / std_imean
        train_lsc = (train_lsc - mean_lsc) / std_lsc
        test_i_mean = (test_i_mean - mean_imean) / std_imean
        test_lsc = (test_lsc - mean_lsc) / std_lsc

    # fit model parameters (a, b, c and w) to data
    # this step minimizes the Log-Likelihood function
    # of the model
    # the Log-Likelihood function is determined analytically
    # from the activation function (output non-linearity)
    fit_params = fit_parameters_mle(
        input_signal=np.vstack([train_i_mean.ravel(), train_lsc.ravel()]),
        response_counts=train_responses[:, temporal_crop - 1:].ravel(),
        fit_func=fit_func,
        fit_func_der=fit_func_der,
        starter_params=starter_params,
        method=mle_method,
        bounds=fit_bounds,
        check_gradient=check_gradient,
    )

    # obtain the rvalue of the fit to frozen noise using the
    # params obtained from fitting
    predicted = fit_func(
        np.vstack([test_i_mean.ravel(), test_lsc.ravel()]),
        fit_params,
    )

    predict_rvalue, _ = pearsonr(ground_truth, predicted)

    output_dict = {
        "fit_parameters": fit_params,
        "r": predict_rvalue.item() if not isinstance(predict_rvalue, float) else predict_rvalue,
        "running_imean": train_i_mean,
        "train_lsc": train_lsc,
        "frozen_imean": test_i_mean,
        "test_lsc": test_lsc,
        "fitted_gaussian": fitted_gaussian,
        "prediction": predicted.ravel(),
        "stimulus_smoothing": stimulus_smoothing,
        "spatial_crop": spatial_crop,
        "temporal_crop": temporal_crop,
    }
    if z_score_signals:
        output_dict.update({
            "mean_imean": mean_imean,
            "mean_lsc": mean_lsc,
            "std_imean": std_imean,
            "std_lsc": std_lsc,
        })

    return output_dict
