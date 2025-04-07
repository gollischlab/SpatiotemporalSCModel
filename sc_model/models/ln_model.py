from typing import Union, Optional, Iterable

import numpy as np
from scipy.stats import pearsonr

from sc_model.dataio import WhitenoiseLoader, NaturalMovieLoader
from sc_model.utils.receptive_fields import get_spat_temp_filt
from sc_model.utils.convolutions import convolve_stimulus_with_kernels
from sc_model.utils.gaussian import fit_gauss2d
from sc_model.utils.minimization import fit_parameters_mle


def ln_model_single_cell(
    train_stimuli: Union[WhitenoiseLoader, NaturalMovieLoader, np.ndarray],
    test_stimuli: np.ndarray,
    train_responses: np.ndarray,
    ground_truth: np.ndarray,
    cell_sta: np.ndarray,
    temporal_crop: int,
    spatial_crop: int,
    fit_func: callable,
    fit_func_der: Optional[callable],
    starter_params,
    fit_bounds: Optional[Iterable] = None,
    mle_method: str = "L-BFGS-B",
    sigma_window: float = 3.0,
    check_gradient: bool = False,
    stimulus_smoothing: Optional[float] = None,
    sta_method: str = "sigpix",
    sigpix_threshold: float = 4.5,
    z_score_signals: bool = True,
):
    """
    Fit a linear-nonlinear model to the data using maximum likelihood estimation.

    This function fits a linear-nonlinear model to the data using maximum likelihood estimation (MLE).
    The model consists of a linear filter followed by a non-linear activation function.
    The linear filter is represented by a spatial and temporal kernel, which are obtained
    from the cell's spike-triggered average (STA). The non-linear activation function is represented
    by a set of parameters that are optimized using the MLE method.

    The function returns the fitted parameters, the r-value of the fit, and the convolved
    signals for both the running and frozen noise. It allows for the option to z-score the convolved
    signals, to use a Gaussian function fit to the spatial filter and to smooth the stimulus by convolving it
    with a Gaussian kernel of a specified smoothing radius (`stimulus_smoothing`). It also allows for the
    specification of the method used to fit the parameters, the bounds for the parameters, and whether to
    check the gradient during optimization. If z-scoring is applied, the function also returns the mean and
    standard deviation of the convolved signals.

    :param train_stimuli:
        The  stimuli used for training the model.
    :param test_stimuli:
        The stimuli used for testing the model.
    :param train_responses:
        The responses to the training stimuli.
    :param ground_truth:
        The responses to the test stimuli averaged across trials, used for evaluation.
    :param cell_sta:
        The spike-triggered average (STA) of the cell.
    :param temporal_crop:
        The size of the temporal crop.
    :param spatial_crop:
        The size of the square spatial crop, specified as the number of pixels around the receptive field center.
    :param fit_func:
        The non-linear activation function to be used in the model.
    :param fit_func_der:
        The derivative of the non-linear activation function with respect to the parameters to be optimized.
    :param starter_params:
        The initial parameters for the optimization.
    :param fit_bounds:
        The bounds for the parameters to be optimized.
        The bounds should be specified as a list of tuples, where each tuple contains the lower and upper
        bounds for each parameter. For example, [(0, 1), (0, None), (0, None)] specifies that the first
        parameter is bounded between 0 and 1, while the second and third parameters are unbounded.
    :param mle_method:
        The method to be used for maximum likelihood estimation passed to the `minimize` function of `scipy`.
    :param sigma_window:
        The standard deviation of the Gaussian function used to fit the spatial filter.
        If set to 0, the unsmoothed spatial filter is used.
    :param check_gradient:
        Whether to check the gradient during optimization.
    :param stimulus_smoothing:
        The radius of the Gaussian kernel used to smooth the stimulus.
        If set to None, no smoothing is applied.
    :param sta_method:
        The method used to compute the spatial and temporal kernels from the cell STA.
        Options are "sigpix" or "mis_pos". The "sigpix" method uses the significant pixels
        in the STA to compute the kernels, while the "mis_pos" method uses the position
        of the maximum value in the STA to compute the kernels.
    :param sigpix_threshold:
        The threshold for significant pixels in the STA.
    :param z_score_signals:
        Whether to z-score the convolved signals.

    :return:
        A dictionary containing the fitted parameters, the r-value of the fit, the convolved signals,
        the spatial filter, the prediction, and the mean and standard deviation of the convolved signals
        if z-scoring is applied. The dictionary also contains flags indicating whether the spatial filter
        was smoothed and the sizes of the spatial and temporal crops.
        The keys in the dictionary are:
        - 'fit_parameters': The fitted parameters of the model.
        - 'r': The r-value of the fit.
        - 'train_convolved': The convolved signals for the training stimuli.
        - 'test_convolved': The convolved signals for the test stimuli.
        - 'prediction': The predicted responses to the test stimuli.
        - 'stimulus_smoothing': The standard deviation of the Gaussian kernel used for smoothing.
        - 'filter_smoothing': The standard deviation of the Gaussian function used to fit the spatial filter.
        - 'spatial_crop': The size of the spatial crop.
        - 'temporal_crop': The size of the temporal crop.
        - 'spatial_filter': The spatial filter obtained from the STA.
        - 'mean_convolved': The mean of the convolved signals for the training stimuli (if z-scoring is applied).
        - 'std_convolved': The standard deviation of the convolved signals for the training stimuli (if z-scoring is applied).

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

    if sigma_window > 0.0:
        # compute the gaussian fit on the spatial kernel
        gaussian_params, _, spatial_filter = fit_gauss2d(
            spatial_kernel=spatial_filter,
            sigma_window=sigma_window,
        )

    # convolve running noise with spatial and temporal stas
    train_convolved = convolve_stimulus_with_kernels(
        stimulus=train_stimuli,
        spatial_filter=spatial_filter,
        temporal_filter=temporal_filter,
        total_trials=train_responses.shape[0],
        stimulus_smoothing=stimulus_smoothing,
    )

    # convolve frozen noise with spatial and temporal kernels
    test_convolved = convolve_stimulus_with_kernels(
        stimulus=test_stimuli,
        spatial_filter=spatial_filter,
        temporal_filter=temporal_filter,
        total_trials=test_stimuli.shape[0],
        stimulus_smoothing=stimulus_smoothing,
    )

    if z_score_signals:
        mean_convolved = train_convolved.mean()
        std_convolved = train_convolved.std()
        train_convolved = (train_convolved - mean_convolved) / std_convolved
        test_convolved = (test_convolved - mean_convolved) / std_convolved

    # fit model parameters (a, b, c and w) to data
    # this step minimizes the Log-Likelihood function
    # of the model
    # the Log-Likelihood function is determined analytically
    # from the activation function (output non-linearity)
    fit_params = fit_parameters_mle(
        input_signal=np.expand_dims(train_convolved.ravel(), axis=0),
        response_counts=train_responses[:, temporal_crop - 1:].ravel(),
        fit_func=fit_func,
        fit_func_der=fit_func_der,
        starter_params=starter_params,
        method=mle_method,
        check_gradient=check_gradient,
        bounds=fit_bounds,
        options={"disp": True},
    )

    # obtain the rvalue of the fit to frozen noise using the
    # params obtained from fitting
    predicted = fit_func(
        np.expand_dims(test_convolved.ravel(), axis=0),
        fit_params,
    ).ravel()

    predict_rvalue, _ = pearsonr(ground_truth, predicted)

    output_dict = {
        "fit_parameters": fit_params,
        "r": predict_rvalue.item() if not isinstance(predict_rvalue, float) else predict_rvalue,
        "train_convolved": train_convolved,
        "test_convolved": test_convolved,
        "spatial_filter": spatial_filter,
        "prediction": predicted.ravel(),
        "stimulus_smoothing": stimulus_smoothing,
        "filter_smoothing": sigma_window,
        "spatial_crop": spatial_crop,
        "temporal_crop": temporal_crop,
    }
    if z_score_signals:
        output_dict.update(
            {
                "mean_convolved": mean_convolved,
                "std_convolved": std_convolved,
            }
        )

    return output_dict
