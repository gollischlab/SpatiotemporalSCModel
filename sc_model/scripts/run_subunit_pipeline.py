"""Train subunit model using precomputed STC eigenvectors.

This script implements the full subunit model training pipeline:
1. Load precomputed STC eigenvectors from run_compute_stc.py
2. Train Logical OR model with Group Lasso regularization to discover subunits
3. Prune subunits based on L2 norm threshold
4. Train subunit model with two-stage training (L1 regularization + pruning + retraining)
5. Evaluate on test set and save results

The spatial crop size is automatically loaded from the STC results (computed
by fitting a 2D Gaussian to the spatial filter).

Usage:
    python run_subunit_pipeline.py --dataset 20220426_SS_252MEA6010_le_n3 \
        --stimulus white_noise --cell_id 0 --n_subunits 16

Requires: Precomputed STC results from run_compute_stc.py

Results saved to:
    RESULTS_PATH/dataset/stimulus/subunit_model/seed_X/per_cell_tempcrop_T_nsubunits_N_orlambda_L_sublambda_L2/cell_ID.pkl
"""

import argparse
import pickle

import numpy as np
import torch

from sc_model.dataio import get_data, RESULTS_PATH
from sc_model.models import LogicalORModel, SubunitModel
from sc_model.utils import (
    train_logical_or_model,
    train_subunit_model,
    prune_functional_subunits,
    pearson_correlation,
    convolve_temporal_only,
)


parser = argparse.ArgumentParser(
    prog="subunit_pipeline",
    description="""
    Train subunit model using precomputed STC eigenvectors.

    The pipeline consists of three phases:
    1. Train Logical OR model with Group Lasso to discover functional subunits
    2. Prune subunits based on L2 norm threshold
    3. Train subunit model with two-stage training

    The spatial crop size is loaded from the STC results (computed via Gaussian fit).
    Requires precomputed STC results from run_compute_stc.py.
    """,
)

# Dataset arguments
parser.add_argument(
    "--dataset", default="20220412_SN_252MEA6010_le_s4", type=str,
    help="Dataset name. Either 20220412_SN_252MEA6010_le_s4 or 20220426_SS_252MEA6010_le_n3"
)
parser.add_argument(
    "--stimulus", default="white_noise", type=str,
    help="Stimulus type. Either 'white_noise' or 'naturalistic_movies'"
)
parser.add_argument(
    "--stimulus_seed", default=0, type=int,
    help="Stimulus seed. 0 for white noise, 1 or 2 for naturalistic movies."
)
parser.add_argument(
    "--cell_id", default=272, type=int,
    help="Cell ID to analyze."
)

# Model architecture
parser.add_argument(
    "--n_subunits", default=16, type=int,
    help="Number of subunits to fit in Logical OR model. Default: 16"
)
parser.add_argument(
    "--n_stc_filters", default=16, type=int,
    help="Number of STC filters to use (must match STC computation). Default: 16"
)
parser.add_argument(
    "--temporal_crop_size", default=30, type=int,
    help="Temporal filter size (must match STC computation). Default: 30"
)

# Logical OR training parameters
parser.add_argument(
    "--or_max_epochs", default=500, type=int,
    help="Max epochs for Logical OR training. Default: 500"
)
parser.add_argument(
    "--or_patience", default=50, type=int,
    help="Early stopping patience for Logical OR. Default: 50"
)
parser.add_argument(
    "--or_learning_rate", default=5e-3, type=float,
    help="Learning rate for Logical OR. Default: 5e-3"
)
parser.add_argument(
    "--or_reg_lambda", default=1e-4, type=float,
    help="Group Lasso regularization strength for Logical OR. Default: 1e-4"
)
parser.add_argument(
    "--or_warmup_epochs", default=50, type=int,
    help="Warmup epochs before regularization for Logical OR. Default: 50"
)
parser.add_argument(
    "--or_prune_threshold", default=0.1, type=float,
    help="Pruning threshold for Logical OR subunits (fraction of max norm). Default: 0.1"
)

# Subunit model training parameters
parser.add_argument(
    "--sub_max_epochs", default=500, type=int,
    help="Max epochs per stage for subunit model training. Default: 500"
)
parser.add_argument(
    "--sub_patience", default=50, type=int,
    help="Early stopping patience for subunit model. Default: 50"
)
parser.add_argument(
    "--sub_learning_rate", default=1e-3, type=float,
    help="Learning rate for subunit model. Default: 1e-3"
)
parser.add_argument(
    "--sub_l1_lambda", default=1e-4, type=float,
    help="L1 regularization on subunit model weights. Default: 1e-4"
)
parser.add_argument(
    "--sub_prune_threshold", default=0.1, type=float,
    help="Pruning threshold for subunit model weights (fraction of max). Default: 0.1"
)

# Common training parameters
parser.add_argument(
    "--batch_size", default=512, type=int,
    help="Batch size for training. Default: 512"
)
parser.add_argument(
    "--validation_fraction", default=0.2, type=float,
    help="Fraction of training data to use for validation. Default: 0.2"
)
parser.add_argument(
    "--random_seed", default=12345, type=int,
    help="Random seed for reproducibility. Default: 12345"
)
parser.add_argument(
    "--sigpix_threshold", default=6.0, type=float,
    help="Threshold for significant pixels in STA. Default: 6.0"
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


def prepare_data_splits(
    data,
    cell_id,
    temporal_crop,
    spatial_crop,
    rf_center,
    temp_filt,
    validation_fraction,
    random_seed,
):
    """
    Prepare train/val/test data splits with temporally-convolved stimulus.

    :param data: Data dictionary from get_data().
    :type data: dict
    :param cell_id: Cell ID.
    :type cell_id: int
    :param temporal_crop: Temporal filter size.
    :type temporal_crop: int
    :param spatial_crop: Spatial filter size (from STC results).
    :type spatial_crop: int
    :param rf_center: RF center coordinates (from STC results).
    :type rf_center: tuple
    :param temp_filt: Temporal filter (from STC results).
    :type temp_filt: np.ndarray
    :param validation_fraction: Fraction of training trials for validation.
    :type validation_fraction: float
    :param random_seed: Random seed.
    :type random_seed: int

    :return: Tuple of (train_stim, train_resp, val_stim, val_resp, test_stim,
        test_resp) containing all data splits.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    rng = np.random.default_rng(random_seed)

    # Configure stimulus loader with parameters from STC results
    data["stimuli"].rf_center = rf_center
    data["stimuli"].temporal_filter_size = temporal_crop
    data["stimuli"].spatial_filter_size = spatial_crop
    data["stimuli"].load_trials_to_cache()

    # Get training data
    train_stim_raw = data["stimuli"].get_training_set()  # (n_trials, n_frames, h, w)
    train_resp_raw = data["train_responses"][cell_id]  # (n_trials, n_frames)

    # Split into train/validation
    n_trials = train_stim_raw.shape[0]
    n_val_trials = max(1, int(n_trials * validation_fraction))
    all_indices = np.arange(n_trials)
    rng.shuffle(all_indices)
    val_indices = all_indices[:n_val_trials]
    train_indices = all_indices[n_val_trials:]

    print(f"Train trials: {len(train_indices)}, Validation trials: {len(val_indices)}")

    # Process training data
    def process_trials(trial_indices, stim_raw, resp_raw):
        stim_list = []
        resp_list = []
        for idx in trial_indices:
            trial_stim = stim_raw[idx]
            trial_resp = resp_raw[idx]

            # Use temporal-only convolution
            convolved = convolve_temporal_only(
                stimulus=trial_stim,
                temporal_filter=temp_filt,
            )
            n_valid = convolved.shape[0]
            convolved_flat = convolved.reshape(n_valid, -1)
            resp_aligned = trial_resp[temporal_crop - 1:][:n_valid]

            stim_list.append(convolved_flat)
            resp_list.append(resp_aligned)

        return np.concatenate(stim_list, axis=0), np.concatenate(resp_list, axis=0)

    train_stim, train_resp = process_trials(train_indices, train_stim_raw, train_resp_raw)
    val_stim, val_resp = process_trials(val_indices, train_stim_raw, train_resp_raw)

    # Process test data
    test_stim_raw = data["stimuli"].get_test_set()  # (1, n_frames, h, w)
    test_resp_raw = data["test_responses"][cell_id]  # (n_repeats, n_frames)

    # Convolve test stimulus
    test_convolved = convolve_temporal_only(
        stimulus=test_stim_raw[0],
        temporal_filter=temp_filt,
    )
    n_valid_test = test_convolved.shape[0]
    test_stim = test_convolved.reshape(n_valid_test, -1)

    # Average test responses across repeats
    test_resp = test_resp_raw[:, temporal_crop - 1:][:, :n_valid_test].mean(axis=0)

    print(f"Train samples: {train_stim.shape[0]}, Val samples: {val_stim.shape[0]}, Test samples: {test_stim.shape[0]}")

    return train_stim, train_resp, val_stim, val_resp, test_stim, test_resp


def load_stc_results(stc_path):
    """
    Load precomputed STC results.

    :param stc_path: Path to STC pickle file.
    :type stc_path: Path

    :return: STC results dictionary.
    :rtype: dict
    """
    with open(stc_path, "rb") as f:
        stc_results = pickle.load(f)
    return stc_results


if __name__ == "__main__":
    args = parser.parse_args()

    # Extract arguments
    dataset = args.dataset
    stimulus = args.stimulus
    stimulus_seed = args.stimulus_seed
    cell_id = args.cell_id
    n_subunits = args.n_subunits
    n_stc_filters = args.n_stc_filters
    temporal_crop = args.temporal_crop_size
    batch_size = args.batch_size
    validation_fraction = args.validation_fraction
    random_seed = args.random_seed
    sigpix_threshold = args.sigpix_threshold
    overwrite = args.overwrite
    device = args.device

    # Handle seed conventions
    if stimulus_seed == 0:
        stimulus_seed = None

    print(f"Arguments: {args}")

    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Define STC path (variable crop - no spatcrop in folder name)
    stc_path = (
        RESULTS_PATH / dataset / "white_noise" / "stc" /
        f"per_cell_tempcrop_{temporal_crop}_nfilters_{n_stc_filters}" /
        f"cell_{cell_id}.pkl"
    )

    # Check if STC results exist
    if not stc_path.exists():
        raise FileNotFoundError(
            f"STC results not found at {stc_path}. "
            f"Please run run_compute_stc.py first."
        )

    # Load STC results to get spatial_crop_size for output path
    print(f"Loading STC results from: {stc_path}")
    stc_results = load_stc_results(stc_path)
    stc_filters = stc_results["eigenvectors"]  # (n_filters, n_features)
    spatial_crop = stc_results["spatial_crop_size"]
    rf_center = stc_results["rf_center"]
    temp_filt = stc_results["temporal_filter"]

    print(f"Loaded {stc_filters.shape[0]} STC filters with {stc_filters.shape[1]} features")
    print(f"Spatial crop size from STC: {spatial_crop}")
    print(f"RF center from STC: {rf_center}")

    # Define output path
    output_path = (
        RESULTS_PATH / dataset / stimulus / "subunit_model" / f"seed_{stimulus_seed}" /
        f"per_cell_tempcrop_{temporal_crop}_nsubunits_{n_subunits}_"
        f"orlambda_{args.or_reg_lambda:.0e}_sublambda_{args.sub_l1_lambda:.0e}" /
        f"cell_{cell_id}.pkl"
    )

    if output_path.exists() and not overwrite:
        print(f"Output file already exists at {output_path}, skipping...")
    else:
        print(f"Saving results to: {output_path}")

        # Load data and prepare splits
        print("Loading data...")
        data = get_data(
            dataset=dataset,
            stimulus=stimulus,
            stimulus_seed=stimulus_seed,
            stimuli=True,
            stas=True,
            responses=True,
        )

        train_stim, train_resp, val_stim, val_resp, test_stim, test_resp = prepare_data_splits(
            data=data,
            cell_id=cell_id,
            temporal_crop=temporal_crop,
            spatial_crop=spatial_crop,
            rf_center=rf_center,
            temp_filt=temp_filt,
            validation_fraction=validation_fraction,
            random_seed=random_seed,
        )

        # Binarize responses for Logical OR model (spike vs no spike)
        train_resp_binary = (train_resp > 0).astype(np.float32)
        val_resp_binary = (val_resp > 0).astype(np.float32)

        # =====================================================================
        # Phase 1: Train Logical OR model
        # =====================================================================
        print(f"\n{'='*60}")
        print("PHASE 1: Training Logical OR model")
        print(f"{'='*60}")

        or_model = LogicalORModel(
            stc_filters=stc_filters,
            n_subunits=n_subunits,
            device=device,
        )

        or_results = train_logical_or_model(
            model=or_model,
            train_stim=train_stim,
            train_resp=train_resp_binary,
            val_stim=val_stim,
            val_resp=val_resp_binary,
            batch_size=batch_size,
            max_epochs=args.or_max_epochs,
            patience=args.or_patience,
            learning_rate=args.or_learning_rate,
            reg_lambda=args.or_reg_lambda,
            warmup_epochs=args.or_warmup_epochs,
            device=device,
        )

        or_model = or_results["model"]
        or_best_epoch = or_results["best_epoch"]

        # =====================================================================
        # Phase 2: Prune Logical OR subunits
        # =====================================================================
        print(f"\n{'='*60}")
        print("PHASE 2: Pruning Logical OR subunits")
        print(f"{'='*60}")

        functional_subunits = or_model.get_functional_subunits()
        A_matrices = or_model.get_A_matrices()
        thresholds = or_model.get_thresholds()

        pruned_subunits, pruned_A, pruned_thresholds, or_keep_mask, subunit_norms = prune_functional_subunits(
            functional_subunits=functional_subunits,
            A_matrices=A_matrices,
            thresholds=thresholds,
            prune_threshold=args.or_prune_threshold,
            normalize=True,
        )

        n_after_or_prune = pruned_subunits.shape[0]
        print(f"Subunits after OR pruning: {n_after_or_prune}")

        # =====================================================================
        # Phase 3: Train subunit model
        # =====================================================================
        print(f"\n{'='*60}")
        print("PHASE 3: Training subunit model")
        print(f"{'='*60}")

        subunit_model = SubunitModel(
            functional_subunits=pruned_subunits,
            a_init=1.0,
            b_init=0.5,
            device=device,
        )

        subunit_results = train_subunit_model(
            model=subunit_model,
            train_stim=train_stim,
            train_resp=train_resp,
            val_stim=val_stim,
            val_resp=val_resp,
            batch_size=batch_size,
            max_epochs=args.sub_max_epochs,
            patience=args.sub_patience,
            learning_rate=args.sub_learning_rate,
            l1_lambda=args.sub_l1_lambda,
            prune_threshold=args.sub_prune_threshold,
            device=device,
        )

        final_model = subunit_results["model"]
        subunit_keep_mask = subunit_results["keep_mask"]
        n_final_subunits = final_model.n_subunits

        # =====================================================================
        # Evaluation on test set
        # =====================================================================
        print(f"\n{'='*60}")
        print("EVALUATION: Computing test correlation")
        print(f"{'='*60}")

        final_model.eval()
        with torch.no_grad():
            test_stim_t = torch.tensor(test_stim, dtype=torch.float32, device=device)
            test_pred = final_model(test_stim_t).cpu().numpy()
            test_resp_t = torch.tensor(test_resp, dtype=torch.float32)
            test_pred_t = torch.tensor(test_pred, dtype=torch.float32)
            test_corr = pearson_correlation(test_pred_t, test_resp_t).item()

        print(f"Test correlation: {test_corr:.4f}")
        print(f"Final number of subunits: {n_final_subunits}")

        # =====================================================================
        # Save results
        # =====================================================================
        output = {
            # Test performance
            "test_corr": test_corr,
            "test_pred": test_pred,
            "test_resp": test_resp,

            # Final model parameters
            "final_subunits": final_model.get_subunits(),
            "subunit_weights": final_model.get_weights(),
            "subunit_a": final_model.a.item(),
            "subunit_b": final_model.b.item(),

            # Training info
            "or_best_epoch": or_best_epoch,
            "subunit_best_epoch": subunit_results["best_epoch"],
            "subunit_stage1_best_epoch": subunit_results["stage1_best_epoch"],
            "subunit_stage2_best_epoch": subunit_results["stage2_best_epoch"],

            # Subunit counts
            "n_original_subunits": n_subunits,
            "n_after_or_prune": n_after_or_prune,
            "n_final_subunits": n_final_subunits,

            # Pruning masks
            "or_keep_mask": or_keep_mask,
            "subunit_keep_mask": subunit_keep_mask,

            # Training histories
            "or_history": or_results["history"],
            "subunit_history": subunit_results["history"],

            # Metadata (scaled to match stimulus resolution)
            "cell_id": cell_id,
            "spatial_crop_size": spatial_crop,  # Scaled if resample_ratio > 1
            "temporal_crop_size": temporal_crop,
            "rf_center": rf_center,  # Scaled if resample_ratio > 1
            "temporal_filter": temp_filt,

            # Original STC metadata (before scaling)
            "stc_spatial_crop_size": stc_results["spatial_crop_size"],
            "stc_rf_center": stc_results["rf_center"],

            # STC info (stc_filters are scaled if resample_ratio > 1)
            "stc_filters": stc_filters,
            "stc_eigenvalues": stc_results["eigenvalues"],

            # Hyperparameters
            "hyperparameters": {
                "n_subunits": n_subunits,
                "n_stc_filters": n_stc_filters,
                "or_reg_lambda": args.or_reg_lambda,
                "or_warmup_epochs": args.or_warmup_epochs,
                "or_prune_threshold": args.or_prune_threshold,
                "sub_l1_lambda": args.sub_l1_lambda,
                "sub_prune_threshold": args.sub_prune_threshold,
                "batch_size": batch_size,
                "random_seed": random_seed,
            },
        }

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(output, f)

        print(f"\nResults saved to: {output_path}")
        print("Done!")
