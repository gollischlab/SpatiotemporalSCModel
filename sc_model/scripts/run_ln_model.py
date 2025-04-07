import argparse

import numpy as np
import pandas as pd

from sc_model.models.ln_model import ln_model_single_cell
from sc_model.utils.nonlinearities import vectorized_softplus, vectorized_softplus_derivative
from sc_model.utils.receptive_fields import get_mis_pos
from sc_model.dataio import get_data, save_df_to_pickle, RESULTS_PATH


parser = argparse.ArgumentParser(
    prog="ln_model",
    description="""
    Train and evaluate a single-cell linear-nonlinear model on the specified dataset and stimulus.
    
    The model consists of a linear filter, representing an estimate of the cell's receptive field, followed by a 
    non-linear activation function. The parameters of the model are optimized using maximum likelihood estimation (MLE).
    The model is trained on the specified dataset and stimulus, and the results are saved to a different pickle file
    for each combination of input parameters.
    """,
)
parser.add_argument(
    "--dataset", default="20220426_SS_252MEA6010_le_n3", type=str,
    help="Dataset name. Either 20220412_SN_252MEA6010_le_s4 or 20220426_SS_252MEA6010_le_n3"
)
parser.add_argument(
    "--stimulus", default="naturalistic_movies", type=str,
    help="Stimulus name. Either white_noise or naturalistic_movies"
)
parser.add_argument(
    "--stimulus_seed", default=0, type=int,
    help="Stimulus seed. Either 1 or 2 for naturalistic movie. Must be 0 for white noise."
)
parser.add_argument(
    "--cell_id", default=0, type=int,
    help="""
    Cell ID. The ID of the cell to analyze. 
    For 20220412_SN_252MEA6010_le_s4: 0..369. For 20220426_SS_252MEA6010_le_n3: 0..109.
    """
)
parser.add_argument(
    "--spatial_crop_size", default=20, type=int,
    help="""
    Spatial crop size. Specifies the size of the spatial filter to use in the model.
    Value specifies the number of pixels to each side of the receptive field center.
    For example, a value of 20 means a spatial filter size of 40x40 pixels.
    """
)
parser.add_argument(
    "--temporal_crop_size", default=30, type=int,
    help="""
    Temporal crop size. Value specifies the number of time points in the past to include in the temporal filter.
    """
)
parser.add_argument(
    "--stimulus_smoothing", default=0.0, type=float,
    help="Stimulus smoothing. Specifies the amount of Gaussian smoothing to apply to the stimulus (in units of pixels)."
)
parser.add_argument(
    "--sigpix_threshold", default=6.0, type=float,
    help="""
    Significant pixel threshold. Specifies the threshold for selecting significant pixels in the spatial filter.
    Pixels with a value greater than this threshold are used to estimate the temporal and spatial filter. Default value 
    of 6.0 should work well for the provided datasets.
    """
)
parser.add_argument(
    "--overwrite", default=False, dest="overwrite", action="store_true",
    help="""
    Overwrite existing results. If specified, existing results will be overwritten.
    If not specified, existing results will be skipped.
    """
)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset
    stimulus = args.stimulus
    stimulus_seed = args.stimulus_seed
    cell_id = args.cell_id
    temporal_crop = args.temporal_crop_size
    spatial_crop = args.spatial_crop_size
    stim_smooth = args.stimulus_smoothing
    overwrite = args.overwrite
    sigpix_threshold = args.sigpix_threshold

    if stim_smooth == 0.0:
        stim_smooth = None

    if stimulus_seed == 0:
        stimulus_seed = None

    print(f"Arguments: {args}")

    output_path = (
        RESULTS_PATH / dataset / stimulus / f"seed_{stimulus_seed}" / "ln_model" / f"per_cell_{dataset}_"
        f"tempcrop_{temporal_crop}_spatcrop_{spatial_crop}_stimsmooth_{stim_smooth}_sigpix_{sigpix_threshold}" /
        f"cell_{cell_id}.pkl"
    )

    if output_path.exists() and not overwrite:
        print(f"Output file already exists at {output_path}, skipping...")
    else:
        print(f"Saving results to: {output_path}")

        data = get_data(
            dataset=dataset,
            stimulus=stimulus,
            stimulus_seed=stimulus_seed,
            stimuli=True,
            stas=True,
            responses=True,
        )

        bounds = [
            (0., None),
            (None, None),
            (None, None),
        ]

        data["stimuli"].temporal_filter_size = temporal_crop
        data["stimuli"].spatial_filter_size = spatial_crop

        sta = data["stas"][cell_id]
        frame_height = data["stimuli"].frame_height / data["stimuli"].downsample
        resample_ratio = round(frame_height / sta.shape[1])
        if resample_ratio > 1:
            sta = np.kron(
                sta,
                np.ones((1, resample_ratio, resample_ratio), dtype=int)
            )
        # rescale STA in units of std. around mean
        sta = (sta - sta.mean()) / sta.std()
        mis_pos = get_mis_pos(sta)
        rf_center = (mis_pos[1], mis_pos[2])
        data["stimuli"].rf_center = rf_center
        data["stimuli"].load_trials_to_cache()

        train_resp = data["train_responses"][cell_id]
        test_resp = data["test_responses"][cell_id]

        frozen_stimuli = data["stimuli"].get_test_set()
        ground_truth = test_resp[:, temporal_crop - 1:].mean(axis=0)

        output = ln_model_single_cell(
            train_stimuli=data["stimuli"],
            test_stimuli=frozen_stimuli,
            train_responses=train_resp,
            ground_truth=ground_truth,
            cell_sta=sta,
            temporal_crop=temporal_crop,
            spatial_crop=spatial_crop,
            fit_func=vectorized_softplus,
            fit_func_der=vectorized_softplus_derivative,
            starter_params=[train_resp.max(), -2.0, 1.0],
            sigma_window=3.0,
            fit_bounds=bounds,
            mle_method="L-BFGS-B",
            check_gradient=False,
            stimulus_smoothing=stim_smooth,
            sigpix_threshold=sigpix_threshold,
        )

        fit_parameters = output.pop("fit_parameters")
        params = {
            "a": fit_parameters[0],
            "b": fit_parameters[1],
            "w": fit_parameters[2],
        }
        output.update(params)

        output["cell_id"] = cell_id

        for k, v in output.items():
            if not isinstance(v, list):
                output[k] = [v]

        output_df = pd.DataFrame(output)

        save_df_to_pickle(
            dataframe=output_df,
            destination=output_path,
            mkdir=True,
            overwrite=overwrite,
        )
