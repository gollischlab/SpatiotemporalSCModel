from typing import Union, Optional

import numpy as np
from tqdm import tqdm

try:
    import cupy as cp
    from cupyx.scipy.signal import convolve2d
    from cupyx.scipy.ndimage import gaussian_filter
    _HAVE_CUPY = True
except ImportError:
    import numpy as cp
    from scipy.signal import convolve2d
    from scipy.ndimage import gaussian_filter
    _HAVE_CUPY = False

from sc_model.utils.project_variables import MAX_FLOAT_SIZE
from sc_model.dataio import WhitenoiseLoader, NaturalMovieLoader


def convolve_stimulus_with_kernels(
    stimulus: Union[WhitenoiseLoader, NaturalMovieLoader, np.ndarray],
    spatial_filter: np.ndarray,
    temporal_filter: np.ndarray,
    total_trials: int,
    stimulus_smoothing: Optional[float] = None,
):
    """
    Convolve a given stimulus with spatial and temporal kernels.

    This function performs a convolution operation on the input stimulus using the specified
    spatial and temporal kernels. The convolution is performed in chunks to optimize memory usage
    and speed. The function also applies a Gaussian filter to the stimulus if specified. It uses
    CuPy for GPU-accelerated computation if available, otherwise falls back to NumPy.

    :param stimulus: The input stimulus to be convolved. It can be either a 3D array with dimensions
        (trials, time, space) or the WhitenoiseLoader or NaturalMovieLoader object.
    :type stimulus: np.ndarray, WhitenoiseLoader, NaturalMovieLoader
    :param spatial_filter: The spatial filter to convolve the stimulus with. It should be a 2D array.
    :type spatial_filter: np.ndarray
    :param temporal_filter: The temporal filter to convolve the stimulus with. It should be a 1D array.
    :type temporal_filter: np.ndarray
    :param total_trials: The number of trials in the stimulus to process. If specified, only the first
        `total_trials` trials will be processed. If None, all trials will be processed.
    :type total_trials: int
    :param stimulus_smoothing: The standard deviation for Gaussian kernel. The Gaussian kernel is used
        for smoothing the stimulus. If None, no smoothing is applied. Default is None.
    :type stimulus_smoothing: float, optional

    :return: The convolved response after applying the spatial and temporal filters to the stimulus.
        It is a 2D array with dimensions (trials, time).
    :rtype: np.ndarray
    """
    # Initialize an empty array to store the convolved response
    convolved_response = np.zeros(
        (total_trials, stimulus.shape[1] - temporal_filter.size + 1)
    )

    # Convert the temporal and spatial kernels into CuPy arrays for GPU-accelerated computation
    temporal_filter = cp.asarray(temporal_filter)
    spatial_filter = cp.asarray(spatial_filter)
    k = temporal_filter.size

    # maximum size of the data chunk that can be processed at once based on the available GPU memory
    # value is arbitrary and is defined in `sc_model.utils.project_variables.py` and can be decreased there
    # in case the user runs into memory issues on the GPU or increased to speed up the model
    # current value works well for a GPU with 8GB of memory and still leaves room for other applications in VRAM
    # note: this value differs from the one used in `convolve_stimulus_with_kernels_for_sc` because the ln model
    # uses a different convolution method and requires less memory
    max_float_size = 2 * MAX_FLOAT_SIZE

    # Loop over each trial in the stimulus
    for tr, trial in tqdm(enumerate(stimulus[:total_trials]), total=total_trials, desc="Trial", leave=False):

        # Calculate the size of each data chunk based on the available GPU memory
        frame_nbytes = trial[0].nbytes / (1024 ** 2)  # in megabytes
        chunk_size = int(np.floor(max_float_size / frame_nbytes))
        chunks_needed = int(np.ceil(trial.shape[0] / chunk_size))

        chunked_convolution = []
        for ch in tqdm(range(chunks_needed), desc="Chunk", leave=False):
            chunk_start = (ch * chunk_size) - k + 1
            if chunk_start < 0:
                chunk_start = 0
            chunk_end = (ch + 1) * chunk_size

            # Convert the chunk into a CuPy array for GPU-accelerated computation
            chunk = cp.asarray(trial[chunk_start:chunk_end])

            # If stimulus_smoothing is not None, apply a Gaussian filter to the chunk
            if stimulus_smoothing is not None:
                chunk = gaussian_filter(
                    chunk,
                    sigma=(0, stimulus_smoothing, stimulus_smoothing),
                    output=chunk,
                )

            # Calculate the spatial convolution by multiplying the chunk with
            # the spatial kernel and summing over the spatial dimensions
            spat_conv = (chunk * spatial_filter).sum(axis=(1, 2))

            # Calculate the temporal convolution by convolving the result of
            # the spatial convolution with the temporal kernel
            chunked_convolution.append(_get(cp.convolve(spat_conv, temporal_filter, mode="valid")))

        # Concatenate the result of the temporal convolution for each chunk
        # and store it in the convolved_response array
        convolved_response[tr] = np.concatenate(chunked_convolution)

    # Return the convolved_response array which contains the convolved response
    # for each trial in the stimulus
    return convolved_response


def convolve_stimulus_with_kernels_for_sc(
    stimulus: Union[WhitenoiseLoader, NaturalMovieLoader, np.ndarray],
    spatial_filter: np.ndarray,
    temporal_filter: np.ndarray,
    total_trials: int,
    stimulus_smoothing: Optional[float] = None,
):
    """
    Convolve a given stimulus with spatial and temporal kernels for the SC model.

    This function performs a convolution operation on the input stimulus using the specified
    spatial and temporal kernels. The convolution is performed in chunks to optimize memory usage
    and speed. The function also applies a Gaussian filter to the stimulus if specified. It uses
    CuPy for GPU-accelerated computation if available, otherwise falls back to NumPy.

    :param stimulus: The input stimulus to be convolved. It can be either a 3D array with dimensions
        (trials, time, space) or the WhitenoiseLoader or NaturalMovieLoader object.
    :type stimulus: np.ndarray, WhitenoiseLoader, NaturalMovieLoader
    :param spatial_filter: The spatial filter to convolve the stimulus with. It should be a 2D array.
    :type spatial_filter: np.ndarray
    :param temporal_filter: The temporal filter to convolve the stimulus with. It should be a 1D array.
    :type temporal_filter: np.ndarray
    :param total_trials: The number of trials in the stimulus to process. If specified, only the first
        `total_trials` trials will be processed. If None, all trials will be processed.
    :type total_trials: int
    :param stimulus_smoothing: The standard deviation for Gaussian kernel. The Gaussian kernel is used
        for smoothing the stimulus. If None, no smoothing is applied. Default is None.
    :type stimulus_smoothing: float, optional

    :return: A tuple containing two arrays:
        - The mean luminosity signal (I_mean) for each trial.
        - The local spatial contrast signal (LSC) for each trial.
        Each array has dimensions (trials, time).
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    # Initialize empty arrays to store the convolved responses
    convolved_response_i_mean = np.zeros(
        (total_trials, stimulus.shape[1] - temporal_filter.size + 1)
    )
    convolved_response_lsc = np.zeros_like(convolved_response_i_mean)

    # Get the spatial crop size
    spatial_crop = spatial_filter.shape[0] // 2

    # Get the non-zero pixels in the spatial kernel
    non_zero_pixels = spatial_filter[spatial_filter != 0.0]
    # Calculate the sum of non-zero pixels
    non_zero_pixels_sum = non_zero_pixels.sum()

    # Convert the spatial and temporal kernels into CuPy arrays for GPU-accelerated computation
    spatial_filter = cp.asarray(spatial_filter.flatten())
    temporal_filter = cp.asarray(temporal_filter)
    k = temporal_filter.size

    # maximum size of the data chunk that can be processed at once based on the available GPU memory
    # value is arbitrary and is defined in `sc_model.utils.project_variables.py` and can be decreased there
    # in case the user runs into memory issues on the GPU or increased to speed up the model
    # current value works well for a GPU with 8GB of memory and still leaves room for other applications in VRAM
    max_float_size = MAX_FLOAT_SIZE

    # Loop over each trial in the stimulus
    for tr, trial in tqdm(enumerate(stimulus[:total_trials]), total=total_trials, desc="Trial", leave=False):
        # Calculate the size of each data chunk based on the available GPU memory
        frame_nbytes = trial[0].nbytes / (1024 ** 2)  # in megabytes
        chunk_size = int(np.floor(max_float_size / frame_nbytes))
        chunks_needed = int(np.ceil(trial.shape[0] / chunk_size))

        chunked_imean = []
        chunked_lsc = []
        # Loop over each chunk in the trial
        for ch in tqdm(range(chunks_needed), desc="Chunk", leave=False):
            # Calculate the start and end indices of the chunk
            chunk_start = (ch * chunk_size) - k + 1
            # Ensure that the start index is not negative
            if chunk_start < 0:
                chunk_start = 0
            # Calculate the end index of the chunk
            chunk_end = (ch + 1) * chunk_size

            chunk = trial[chunk_start:chunk_end]

            # reshape chunk to (Time, X*Y)
            chunk = cp.asarray(chunk.reshape(
                (chunk.shape[0], chunk.shape[1] * chunk.shape[2])
            ))
            # calculate temporal convolution
            temp_conv = convolve2d(
                chunk, cp.expand_dims(temporal_filter, axis=-1), mode="valid"
            )
            # apply Gaussian filter to the temporal convolution
            if stimulus_smoothing is not None:
                filt_temp_conv = gaussian_filter(
                    temp_conv.reshape((
                        temp_conv.shape[0], spatial_crop * 2, spatial_crop * 2
                    )),
                    sigma=(0, stimulus_smoothing, stimulus_smoothing),
                    truncate=3.0,
                ).reshape(temp_conv.shape)
            else:
                # if no smoothing is applied, use the original temporal convolution
                filt_temp_conv = temp_conv.copy()

            # calculate the I_mean and LSC
            # according to Liu and Gollisch, Natural Image Coding
            imean = (spatial_filter * temp_conv).sum(axis=-1) / non_zero_pixels_sum
            # inner_sum = spatial_kernel * temp_conv
            # spat_conv = inner_sum.mean(axis=-1)
            # spat_conv = inner_sum / spatial_kernel.size  # I_mean
            lcl_sptl_cntrst = cp.sqrt(
                (
                        spatial_filter * (filt_temp_conv - cp.expand_dims(imean, 1)) ** 2
                ).sum(axis=-1) / non_zero_pixels_sum
            )  # LSC

            chunked_imean.append(_get(imean))
            chunked_lsc.append(_get(lcl_sptl_cntrst))
        convolved_response_i_mean[tr] = np.concatenate(chunked_imean)
        convolved_response_lsc[tr] = np.concatenate(chunked_lsc)

    return convolved_response_i_mean, convolved_response_lsc


def _get(arr: cp.ndarray) -> np.ndarray:
    if _HAVE_CUPY:
        return arr.get()
    else:
        return arr
