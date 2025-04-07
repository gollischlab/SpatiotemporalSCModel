from typing import Union, Optional

import numpy as np
from scipy.stats import median_abs_deviation


def spat_temp_kern_using_mis(
    sta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the spatial and temporal filters using the most important stixel method.

    The most important stixel is the one with the highest variance across time. The spatial filter
    is the frame of the STA at the time of the most important stixel, and the temporal filter is the
    time course of the most important stixel, reversed.

    :param sta: The spatiotemporal array to compute the spatial and temporal filters from.
    :type sta: np.ndarray

    :return: A 2-tuple containing the spatial and temporal filters.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    # Get the position of the most important stixel
    mis_pos = get_mis_pos(sta)
    t_, y_, x_ = mis_pos
    mis_val = sta[mis_pos]

    # Compute the spatial and temporal filters
    spatial_filter = np.sign(mis_val) * sta[t_, :, :]  # sign of the most important stixel * spatial filter
    temporal_filter = sta[:, y_, x_].flatten()[::-1]  # reverse the temporal filter

    return spatial_filter, temporal_filter


def spat_temp_kern_using_sigpix(
    sta: np.ndarray,
    sigpix_threshold: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the spatial and temporal filters using the significant pixels method.

    The significant pixels are the ones with the highest absolute values across time. The
    temporal filter is the average time course of the significant pixels, and the spatial filter is
    the projection of the average temporal filter onto the spatiotemporal array.

    :param sta: The spatiotemporal array to compute the spatial and temporal filters from.
    :type sta: np.ndarray
    :param sigpix_threshold: The threshold for selecting significant pixels. Default is 6.0.
    :type sigpix_threshold: float, optional

    :return: A 2-tuple containing the spatial and temporal filters.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    # Get the maximum absolute value of each pixel across time
    pixel_maxima = np.abs(sta.reshape(sta.shape[0], sta.shape[1] * sta.shape[2])).max(axis=0)
    # Compute the median absolute deviation of the pixel maxima
    pixmax_mad = median_abs_deviation(pixel_maxima)
    # Compute the selection threshold
    selection_threshold = np.median(pixel_maxima) + sigpix_threshold * 1.4826 * pixmax_mad
    # Select the top candidates based on the selection threshold
    top_candidates = np.where(pixel_maxima > selection_threshold)[0]
    # Extract the selected pixels from the spatiotemporal array
    selected_pixels = sta.reshape((sta.shape[0], sta.shape[1] * sta.shape[2]))[:, top_candidates]
    # Compute the temporal filter as the average time course of the selected pixels
    temp_kern = np.mean(selected_pixels, axis=1)
    # Compute the spatial filter by projecting the average temporal filter onto the sta
    spat_kern = np.dot(np.transpose(sta, (1, 2, 0)), temp_kern)
    # Flip the temporal filter before returning
    return spat_kern, temp_kern[::-1]


def get_spat_temp_filt(
    sta: np.ndarray,
    spatial_crop_size: int,
    temporal_crop_size: Optional[int] = None,
    force_crop_size: bool = False,
    method: str = "sigpix",
    sigpix_threshold: float = 6.0,
    rf_center: Union[tuple, None] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the spatial and temporal filters from the spatiotemporal array.

    This function computes the spatial and temporal filters from the spatiotemporal array using
    either the most important stixel method or the significant pixels method. The spatial filter is
    cropped around the receptive field center, and the temporal filter is cropped to the specified
    size. Each filter is normalized to have a unit norm.

    :param sta: The spatiotemporal array to compute the spatial and temporal filters from.
    :type sta: np.ndarray
    :param spatial_crop_size: The size of the spatial crop around the receptive field center.
    :type spatial_crop_size: int
    :param temporal_crop_size: The size of the temporal crop. If None, the full temporal filter is used.
    :type temporal_crop_size: int, optional
    :param force_crop_size: If True, the spatial crop size is forced. Default is False.
    :type force_crop_size: bool
    :param method: The method to use for computing the spatial and temporal filters.
        Can be 'mis' (most important stixel) or 'sigpix' (significant pixels). Default is 'sigpix'.
    :type method: str, optional
    :param sigpix_threshold: The threshold for selecting significant pixels. Default is 6.0.
    :type sigpix_threshold: float, optional
    :param rf_center: User-defined (y, x) coordinates for the receptive field center.
        If None, the center is automatically determined. Default is None.
    :type rf_center: tuple, optional

    :return: A 2-tuple containing the cropped spatial and temporal filters.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    if method == "mis":
        full_spatial, full_temporal = spat_temp_kern_using_mis(sta=sta)
    elif method == "sigpix":
        full_spatial, full_temporal = spat_temp_kern_using_sigpix(sta=sta, sigpix_threshold=sigpix_threshold)
    else:
        raise ValueError(f"Undefined method `{method}` for computing sta!")

    if rf_center is None:
        _, y_, x_ = get_mis_pos(sta)
    else:
        y_, x_ = rf_center

    cropped_spatial_filter = crop_spatial_filter(
        spatial_filter=full_spatial,
        spatial_crop_size=spatial_crop_size,
        force_crop_size=force_crop_size,
        receptive_field_center=(y_, x_),
    )

    cropped_temporal_filter = full_temporal[:temporal_crop_size]

    cropped_spatial_filter /= np.linalg.norm(cropped_spatial_filter)
    cropped_temporal_filter /= np.linalg.norm(cropped_temporal_filter)

    return cropped_spatial_filter, cropped_temporal_filter


def crop_spatial_filter(
    spatial_filter: np.ndarray,
    spatial_crop_size: int,
    force_crop_size: bool,
    receptive_field_center: tuple,
) -> np.ndarray:
    """
    Crop the spatial filter to the specified size around the receptive field center.

    :param spatial_filter: The spatial filter to crop.
    :type spatial_filter: np.ndarray
    :param spatial_crop_size: The size of the spatial crop around the receptive field center.
    :type spatial_crop_size: int
    :param force_crop_size: If True, the spatial crop size is forced. Default is False.
    :type force_crop_size: bool
    :param receptive_field_center: The (y, x) coordinates of the receptive field center.
    :type receptive_field_center: tuple

    :return: The cropped spatial filter.
    :rtype: np.ndarray
    """
    y_, x_ = receptive_field_center
    if force_crop_size:
        cropped_filter = np.zeros((spatial_crop_size * 2, spatial_crop_size * 2))

        kern_xmin, kern_ymin = 0, 0
        kern_ymax, kern_xmax = cropped_filter.shape

        sta_height, sta_width = spatial_filter.shape

        sta_xmin = x_ - spatial_crop_size
        sta_xmax = x_ + spatial_crop_size
        sta_ymin = y_ - spatial_crop_size
        sta_ymax = y_ + spatial_crop_size

        if sta_xmin < 0:
            x_diff = 0 - sta_xmin
            kern_xmin = x_diff
            sta_xmin = 0
        if sta_xmax >= sta_width:
            x_diff = sta_xmax - sta_width
            kern_xmax = kern_xmax - x_diff
            sta_xmax = sta_width

        if sta_ymin < 0:
            y_diff = 0 - sta_ymin
            kern_ymin = y_diff
            sta_ymin = 0
        if sta_ymax >= sta_height:
            y_diff = sta_ymax - sta_height
            kern_ymax = kern_ymax - y_diff
            sta_ymax = sta_height

        cropped_filter[
            kern_ymin:kern_ymax, kern_xmin:kern_xmax
        ] = spatial_filter[
            sta_ymin:sta_ymax, sta_xmin:sta_xmax
        ]

    else:
        cropped_filter = spatial_filter[
            y_ - spatial_crop_size:y_ + spatial_crop_size,
            x_ - spatial_crop_size:x_ + spatial_crop_size,
        ]

    return cropped_filter


def get_mis_pos(array: np.ndarray) -> tuple:
    """
    Get the position of the most important stixel in the spatiotemporal array.
    The most important stixel is defined as the one with the highest variance across time.

    :param array: The spatiotemporal array to compute the most important stixel from.
    :type array: np.ndarray

    :return: A tuple containing the time, y, and x coordinates of the most important stixel.
    :rtype: tuple[int, int, int]
    """
    temporal_variances = np.var(array, axis=0)
    pix_pos = np.unravel_index(
        np.argmax(temporal_variances), (array.shape[1], array.shape[2])
    )
    time_pos = np.argmax(np.abs(array[:, pix_pos[0], pix_pos[1]]))
    return time_pos, pix_pos[0], pix_pos[1]
