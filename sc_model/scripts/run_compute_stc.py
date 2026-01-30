"""Compute Spike-Triggered Covariance (STC) eigenvectors for a single cell.

This script computes the STC eigenvectors from spike-triggered ensemble
analysis, extracting the top-N eigenvectors sorted by absolute eigenvalue
deviation from the raw stimulus covariance.

The spatial crop size is automatically determined by fitting a 2D Gaussian
to the spatial filter from the STA and computing the half-sidelength of a
square that encloses the n-sigma contour.

Usage:
    python run_compute_stc.py --dataset 20220426_SS_252MEA6010_le_n3 \
        --cell_id 0 --n_stc_filters 16

Results are saved to:
    RESULTS_PATH/dataset/white_noise/stc/per_cell_tempcrop_T_nfilters_N/cell_ID.pkl
"""

import argparse
import pickle

import numpy as np

from sc_model.dataio import get_data, RESULTS_PATH
from sc_model.utils import (
    kaardal_stc,
    get_top_eigenvectors,
    convolve_temporal_only,
    fit_gauss2d,
    compute_enclosing_square_half_sidelength,
)
from sc_model.utils.receptive_fields import get_spat_temp_filt, get_mis_pos


parser = argparse.ArgumentParser(
    prog="compute_stc",
    description="""
    Compute Spike-Triggered Covariance (STC) eigenvectors for a single cell.

    The STC analysis identifies stimulus features that modulate the cell's
    response variance, revealing potential subunit structure in the receptive
    field. The spatial crop size is automatically determined by fitting a 2D
    Gaussian to the spatial filter and computing the n-sigma contour.
    """,
)

# Dataset arguments
parser.add_argument(
    "--dataset", default="20220412_SN_252MEA6010_le_s4", type=str,
    help="Dataset name. Either 20220412_SN_252MEA6010_le_s4 or 20220426_SS_252MEA6010_le_n3"
)
parser.add_argument(
    "--cell_id", default=272, type=int,
    help="Cell ID to analyze."
)

# Filter parameters
parser.add_argument(
    "--initial_spat_crop", default=10, type=int,
    help="Initial spatial crop size for RF center fitting. Default: 10"
)
parser.add_argument(
    "--max_spat_crop", default=8, type=int,
    help="Maximum spatial crop size (caps the Gaussian-derived crop). Default: 8"
)
parser.add_argument(
    "--temporal_crop_size", default=30, type=int,
    help="Temporal filter size (time bins into the past). Default: 30"
)
parser.add_argument(
    "--n_stc_filters", default=16, type=int,
    help="Number of top eigenvectors to save. Default: 16"
)
parser.add_argument(
    "--n_sigma", default=3.0, type=float,
    help="Number of sigma for Gaussian contour (determines crop size). Default: 3.0"
)

# STA parameters
parser.add_argument(
    "--sigpix_threshold", default=6.0, type=float,
    help="Threshold for significant pixels in STA (in std units). Default: 6.0"
)

# Output control
parser.add_argument(
    "--overwrite", default=False, action="store_true",
    help="Overwrite existing results."
)
parser.add_argument(
    "--device", default="cuda", type=str,
    help="Device for computation ('cuda' or 'cpu'). Default: 'cuda'"
)


def compute_variable_crop_size(
    sta,
    initial_spat_crop,
    temporal_crop,
    sigpix_threshold,
    n_sigma,
    max_spat_crop,
):
    """
    Compute variable spatial crop size based on Gaussian fit to STA spatial filter.

    This function fits a 2D Gaussian to the spatial filter extracted from the STA
    and computes the half-sidelength of a square that encloses the n-sigma contour.

    :param sta: Spike-triggered average. Shape (n_time, height, width).
    :type sta: np.ndarray
    :param initial_spat_crop: Initial spatial crop size for RF center finding.
    :type initial_spat_crop: int
    :param temporal_crop: Temporal crop size.
    :type temporal_crop: int
    :param sigpix_threshold: Threshold for significant pixels.
    :type sigpix_threshold: float
    :param n_sigma: Number of sigma for the Gaussian contour.
    :type n_sigma: float
    :param max_spat_crop: Maximum allowed spatial crop size.
    :type max_spat_crop: int

    :return: Tuple of (spatial_crop, rf_center, spat_filt, temp_filt, gauss_fit_params):
        - spatial_crop: computed spatial crop size (capped at max_spat_crop)
        - rf_center: (y, x) RF center coordinates
        - spat_filt: spatial filter from STA
        - temp_filt: temporal filter from STA
        - gauss_fit_params: parameters of the fitted Gaussian
    :rtype: tuple[int, tuple, np.ndarray, np.ndarray, np.ndarray]
    """
    # Find RF center from STA
    mis_pos = get_mis_pos(sta)
    rf_center = (mis_pos[1], mis_pos[2])

    # Extract spatial and temporal filters from STA using initial crop
    spat_filt, temp_filt = get_spat_temp_filt(
        sta=sta,
        spatial_crop_size=initial_spat_crop,
        temporal_crop_size=temporal_crop,
        rf_center=rf_center,
        sigpix_threshold=sigpix_threshold,
    )

    # Fit 2D Gaussian to the spatial filter
    try:
        gauss_fit_params, _, _ = fit_gauss2d(
            spatial_kernel=spat_filt,
            sigma_window=None,
            initial_guesses=(
                np.abs(spat_filt).max(),
                initial_spat_crop,  # center x (relative to cropped filter)
                initial_spat_crop,  # center y (relative to cropped filter)
                2.0,  # sigma_x
                2.0,  # sigma_y
                np.pi / 4,  # theta
            ),
            bounds=(
                [-np.inf, 0, 0, 0, 0, 0],
                [np.inf, 2 * initial_spat_crop, 2 * initial_spat_crop,
                 initial_spat_crop, initial_spat_crop, np.pi],
            ),
        )

        # Compute the enclosing square half-sidelength
        half_sidelength = compute_enclosing_square_half_sidelength(
            gauss_fit_params, n_sigma=n_sigma
        )
        spatial_crop = int(np.ceil(half_sidelength))
        spatial_crop = min(spatial_crop, max_spat_crop)

        print(f"Gaussian fit successful:")
        print(f"  sigma_x={gauss_fit_params[3]:.2f}, sigma_y={gauss_fit_params[4]:.2f}")
        print(f"  n_sigma={n_sigma} contour half-sidelength: {half_sidelength:.2f}")
        print(f"  Computed spatial crop: {spatial_crop} (max: {max_spat_crop})")

    except Exception as e:
        print(f"Warning: Gaussian fit failed ({e}), using max_spat_crop={max_spat_crop}")
        spatial_crop = max_spat_crop
        gauss_fit_params = None

    return spatial_crop, rf_center, spat_filt, temp_filt, gauss_fit_params


def prepare_stimulus_and_response(data, cell_id, temporal_crop, spatial_crop, rf_center, temp_filt):
    """
    Prepare temporally-convolved stimulus and spike counts for STC.

    :param data: Data dictionary from get_data().
    :type data: dict
    :param cell_id: Cell ID.
    :type cell_id: int
    :param temporal_crop: Temporal filter size.
    :type temporal_crop: int
    :param spatial_crop: Spatial filter size (computed from Gaussian fit).
    :type spatial_crop: int
    :param rf_center: RF center coordinates (y, x).
    :type rf_center: tuple
    :param temp_filt: Temporal filter for convolution.
    :type temp_filt: np.ndarray

    :return: Tuple of (stim_convolved, spike_counts):
        - stim_convolved: shape (n_times, n_features), temporally-convolved stimulus
        - spike_counts: shape (n_times,), spike counts per time bin
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    # Configure stimulus loader with the computed spatial crop
    data["stimuli"].rf_center = rf_center
    data["stimuli"].temporal_filter_size = temporal_crop
    data["stimuli"].spatial_filter_size = spatial_crop
    data["stimuli"].load_trials_to_cache()

    # Get training stimulus and responses
    train_stim = data["stimuli"].get_training_set()  # (n_trials, n_frames, h, w)
    train_resp = data["train_responses"][cell_id]  # (n_trials, n_frames)

    # Convolve stimulus with temporal filter (preserving spatial structure)
    n_trials, n_frames, h, w = train_stim.shape
    stim_convolved_list = []
    spike_counts_list = []

    for trial_idx in range(n_trials):
        trial_stim = train_stim[trial_idx]  # (n_frames, h, w)
        trial_resp = train_resp[trial_idx]  # (n_frames,)

        # Convolve with temporal filter using the temporal-only function
        convolved = convolve_temporal_only(
            stimulus=trial_stim,
            temporal_filter=temp_filt,
        )  # (n_valid_frames, h, w)

        # Flatten spatial dimensions
        n_valid = convolved.shape[0]
        convolved_flat = convolved.reshape(n_valid, -1)  # (n_valid, h*w)

        # Align responses (account for temporal filter size)
        resp_aligned = trial_resp[temporal_crop - 1:][:n_valid]

        stim_convolved_list.append(convolved_flat)
        spike_counts_list.append(resp_aligned)

    # Concatenate all trials
    stim_convolved = np.concatenate(stim_convolved_list, axis=0)
    spike_counts = np.concatenate(spike_counts_list, axis=0)

    return stim_convolved, spike_counts


if __name__ == "__main__":
    args = parser.parse_args()

    dataset = args.dataset
    cell_id = args.cell_id
    initial_spat_crop = args.initial_spat_crop
    max_spat_crop = args.max_spat_crop
    temporal_crop = args.temporal_crop_size
    n_stc_filters = args.n_stc_filters
    n_sigma = args.n_sigma
    sigpix_threshold = args.sigpix_threshold
    overwrite = args.overwrite
    device = args.device

    print(f"Arguments: {args}")
    print(f"Computing STC for cell {cell_id}")

    # Define output path (using variable crop naming)
    output_path = (
        RESULTS_PATH / dataset / "white_noise" / "stc" /
        f"per_cell_tempcrop_{temporal_crop}_nfilters_{n_stc_filters}" /
        f"cell_{cell_id}.pkl"
    )

    if output_path.exists() and not overwrite:
        print(f"Output file already exists at {output_path}, skipping...")
    else:
        print(f"Saving results to: {output_path}")

        # Load data
        data = get_data(
            dataset=dataset,
            stimulus="white_noise",
            stimulus_seed=None,
            stimuli=True,
            stas=True,
            responses=True,
        )

        # Get STA and normalize
        sta = data["stas"][cell_id]
        frame_height = data["stimuli"].frame_height / data["stimuli"].downsample
        resample_ratio = round(frame_height / sta.shape[1])
        if resample_ratio > 1:
            sta = np.kron(sta, np.ones((1, resample_ratio, resample_ratio), dtype=int))
        sta = (sta - sta.mean()) / sta.std()

        # Compute variable crop size from Gaussian fit
        print("\nComputing variable spatial crop size...")
        spatial_crop, rf_center, spat_filt, temp_filt, gauss_fit_params = compute_variable_crop_size(
            sta=sta,
            initial_spat_crop=initial_spat_crop,
            temporal_crop=temporal_crop,
            sigpix_threshold=sigpix_threshold,
            n_sigma=n_sigma,
            max_spat_crop=max_spat_crop,
        )

        # Prepare stimulus and response using the computed crop size
        print("\nPreparing stimulus and response data...")
        stim_convolved, spike_counts = prepare_stimulus_and_response(
            data=data,
            cell_id=cell_id,
            temporal_crop=temporal_crop,
            spatial_crop=spatial_crop,
            rf_center=rf_center,
            temp_filt=temp_filt,
        )

        print(f"Stimulus shape: {stim_convolved.shape}")
        print(f"Total spikes: {spike_counts.sum():.0f}")
        print(f"RF center: {rf_center}")
        print(f"Final spatial crop: {spatial_crop}")

        # Compute STC
        print(f"\nComputing STC on device: {device}...")
        eigenvalues, eigenvectors, st_avg, raw_cov = kaardal_stc(
            stimulus=stim_convolved,
            spike_counts=spike_counts,
            device=device,
        )

        # Extract top eigenvectors
        top_eigenvectors, top_eigenvalues = get_top_eigenvectors(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            n_top=n_stc_filters,
        )

        print(f"Top {n_stc_filters} eigenvalues: {top_eigenvalues}")

        # Prepare output dictionary
        output = {
            "eigenvectors": top_eigenvectors,  # (n_stc_filters, n_features)
            "eigenvalues": top_eigenvalues,  # (n_stc_filters,)
            "st_avg": st_avg,  # (n_features,)
            "rf_center": rf_center,
            "spatial_crop_size": spatial_crop,  # Variable, computed from Gaussian fit
            "temporal_crop_size": temporal_crop,
            "n_stc_filters": n_stc_filters,
            "cell_id": cell_id,
            "temporal_filter": temp_filt,
            "all_eigenvalues": eigenvalues,
            # Additional metadata for reproducibility
            "n_sigma": n_sigma,
            "max_spat_crop": max_spat_crop,
            "initial_spat_crop": initial_spat_crop,
            "gauss_fit_params": gauss_fit_params,
        }

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(output, f)

        print(f"\nResults saved to: {output_path}")
        print("Done!")
