import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sc_model.utils import CODE_DIR, DATA_REPO

from sc_model.dataio.natural_movie_loader import NaturalMovieLoader
from sc_model.dataio.whitenoise_loader import WhitenoiseLoader


assert CODE_DIR.exists(), f"{CODE_DIR} does not exist, please provide the correct path in sc_model.utils.project_paths"
assert DATA_REPO.exists(), f"{DATA_REPO} does not exist, please provide the correct path in sc_model.utils.project_paths"

DATA_PATH = DATA_REPO / "data"
RESULTS_PATH = DATA_REPO / "results"
MOVIE_IMAGES_PATH = DATA_PATH / "naturalistic_movies" / "stimulus" / "raw_images"

MOVIE_IMAGE_HEIGHT = 800
MOVIE_IMAGE_WIDTH = 1000

FRAME_HEIGHT = 600
FRAME_WIDTH = 800

WN_FRAME_HEIGHT = 150
WN_FRAME_WIDTH = 200


def get_data(
    dataset: str,
    stimulus: str,
    stimulus_seed: Optional[int],
    stas: bool = True,
    stimuli: bool = True,
    responses: bool = True,
):
    """
    Get the data for the specified animal, dataset, and stimulus.

    :param dataset: The name of the dataset to load.
    :type dataset: str
    :param stimulus: The name of the stimulus to load.
    :type stimulus: str
    :param stimulus_seed: The seed for the stimulus. Only used for naturalistic movies.
    :type stimulus_seed: Optional[int]
    :param stas: Whether to load the STAs or not.
    :type stas: bool
    :param stimuli: Whether to load the stimuli or not.
    :type stimuli: bool
    :param responses: Whether to load the responses or not.
    :type responses: bool

    :raises NotImplementedError: If the stimulus is "reversing_grating".
    :raises ValueError: If the combination of stimulus and seed is not valid.

    :return: A dictionary containing the data.
    :rtype: dict
    """
    # let user know that reversing grating data cannot be loaded using this package
    if stimulus == "reversing_grating":
        raise NotImplementedError("Reversing grating data cannot be loaded using this package.")

    # make sure that the combination of stimulus and seed is valid
    if (
            (stimulus == "naturalistic_movies" and stimulus_seed not in [1, 2]) or
            (stimulus == "white_noise" and stimulus_seed is not None)
    ):
        raise ValueError(
            f"The combination of stimulus {stimulus} and seed {stimulus_seed} is not valid."
        )

    # Load the responses for the experiment
    train_responses, test_responses = load_responses(
        stimulus=stimulus,
        dataset_name=dataset,
        stimulus_seed=stimulus_seed,
        binned=True,
    )

    # Get the number of frames for train and test responses
    train_frames = train_responses.shape[-1]
    test_frames = test_responses.shape[-1]

    # define train trials
    train_trials = np.arange(train_responses.shape[1]).tolist()

    # Initialize the output dictionary
    output_dict = {}

    # If STAs are requested, load them
    if stas:
        if stimulus == "naturalistic_movies":
            # when using white noise stas on movies, the stas are flipped
            sta_array = np.flip(load_stas(dataset=dataset), axis=2)
        elif stimulus == "white_noise":
            sta_array = load_stas(dataset)
        else:
            raise ValueError(f"Requested stimulus {stimulus} not recognized!")

        output_dict["stas"] = sta_array

    # If responses are requested, add them to output dictionary
    if responses:
        output_dict["train_responses"] = train_responses
        output_dict["test_responses"] = test_responses

    # If stimuli are requested, load them
    if stimuli:
        if stimulus == "naturalistic_movies":
            fixation_dict = pd.read_csv(
                filepath_or_buffer=DATA_PATH / stimulus / "stimulus" / f"fixations_seed_{stimulus_seed}.txt",
                sep=" ",
                names=["frame_id", "t_start", "t_end", "center_x", "center_y", "flip"],
                dtype={"frame_id": int, "t_start": float, "t_end": float, "center_x": int, "center_y": int, "flip": bool}
            ).to_dict(orient="list")
            stimulus_mean = load_mean(stimulus_seed)
            stimuli_loader = NaturalMovieLoader(
                images_path=MOVIE_IMAGES_PATH,
                image_height=MOVIE_IMAGE_HEIGHT,
                image_width=MOVIE_IMAGE_WIDTH,
                frame_height=FRAME_HEIGHT,
                frame_width=FRAME_WIDTH,
                fixation_dict=fixation_dict,
                stimulus_mean=stimulus_mean,
                temporal_filter_size=30,  # dummy value
                spatial_filter_size=30,  # dummy value
                train_frames=train_frames,
                test_frames=test_frames,
                total_trials=train_responses.shape[1],
                train_trials=train_trials,
                validation_trials=[],
                downsample=1,  # default value, can be changed if need be
            )
            output_dict["stimuli"] = stimuli_loader

        elif stimulus == "white_noise":
            stimuli_loader = WhitenoiseLoader(
                stimulus_folder=DATA_PATH / stimulus / dataset,
                frame_height=WN_FRAME_HEIGHT,
                frame_width=WN_FRAME_WIDTH,
                temporal_filter_size=30,  # dummy value
                spatial_filter_size=30,  # dummy value
                train_frames=train_frames,
                test_frames=test_frames,
                total_trials=train_responses.shape[1],
                train_trials=train_trials,
                validation_trials=[],
            )
            output_dict["stimuli"] = stimuli_loader
        else:
            raise ValueError(f"Requested stimulus {stimulus} not recognized!")

    return output_dict


def save_df_to_pickle(
    dataframe: pd.DataFrame,
    destination: Path,
    mkdir: bool,
    overwrite: bool = False,
):
    """
    Save a pandas DataFrame to a pickle file. If the file already exists, check if overwrite is enabled.
    If the file does not exist, create the directory if mkdir is enabled.

    :param dataframe: The DataFrame to save.
    :type dataframe: pd.DataFrame
    :param destination: The path to save the DataFrame to.
    :type destination: Path
    :param mkdir: Whether to create the directory if it does not exist.
    :type mkdir: bool
    :param overwrite: Whether to overwrite the file if it already exists.
    :type overwrite: bool

    :raises FileNotFoundError: If the directory does not exist and mkdir is False.
    :raises UserWarning: If the file already exists and overwrite is False.

    :return: None
    """
    if not destination.parent.exists():
        if mkdir:
            destination.parent.mkdir(parents=True)
        else:
            raise FileNotFoundError(
                "The path to the file doesn't exist. "
                "Enable `mkdir` if you'd like to create the path."
            )

    if okay_to_write_file(destination, overwrite):
        dataframe.to_pickle(destination.as_posix())
    else:
        raise UserWarning(
            "The file was not written to disk because another file already exists at the location and "
            "`overwrite` was disabled. Enable overwrite if you'd like to save the file anyway."
        )


def okay_to_write_file(savepath: Path, overwrite: bool) -> bool:
    """
    Check if the file can be written to disk. If the file already exists, check if overwrite is enabled.
    If the file does not exist, return True.

    :param savepath:
    :type savepath: Path
    :param overwrite:
    :type overwrite: bool

    :return:
    :rtype: bool
    """
    if savepath.exists():
        if overwrite:
            return True
        else:
            return False
    else:
        return True


def load_mean(
    stimulus_seed: int,
) -> np.ndarray:
    """
    Load the mean image for the specified stimulus seed.
    :param stimulus_seed: The seed for the stimulus.
    :type stimulus_seed: int

    :return : The mean image as a numpy array.
    :rtype: np.ndarray
    """
    mean_path = DATA_PATH / "naturalistic_movies" / "stimulus" / f"stimulus_mean_seed_{stimulus_seed}.npy"
    mean = np.load(mean_path)
    return mean


def load_stas(
    dataset: str,
) -> np.ndarray:
    """
    Load the STAs for the specified dataset.

    :param dataset: The name of the dataset to load.
    :type dataset: str

    The STAs are stored in a numpy array with shape (num_cells, temporal_crop, height, width).

    :return : The STAs as a numpy array.
    :rtype: np.ndarray
    """
    load_path = DATA_PATH / "white_noise" / dataset / "stas.npy"
    return np.load(load_path.as_posix(), mmap_mode="r")


def load_responses(
    stimulus: str,
    dataset_name: str,
    stimulus_seed: Optional[int] = None,
    binned: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the responses for the specified stimulus and dataset.

    :param stimulus: The name of the stimulus to load.
    :type stimulus: str
    :param dataset_name: The name of the dataset to load.
    :type dataset_name: str
    :param stimulus_seed: The seed for the stimulus. Only used for naturalistic movies.
    :type stimulus_seed: Optional[int]
    :param binned: Whether to load binned responses or not.
    :type binned: bool

    :return: A tuple containing the train and test responses as numpy arrays.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    # Load the responses for the specified stimulus and dataset
    if stimulus == "naturalistic_movies":
        responses_folder = DATA_PATH / stimulus / "responses" / dataset_name / f"seed_{stimulus_seed}"
    elif stimulus == "white_noise":
        responses_folder = DATA_PATH / stimulus / dataset_name / "responses"
    elif stimulus == "reversing_grating":
        raise NotImplementedError("Reversing grating data cannot be loaded using this package.")
    else:
        raise ValueError(f"Stimulus {stimulus} not recognized!")

    if binned:
        train_responses = responses_folder / "train_responses_binned.npy"
        test_responses = responses_folder / "test_responses_binned.npy"
    else:
        train_responses = responses_folder / "train_responses_seconds.npy"
        test_responses = responses_folder / "test_responses_seconds.npy"

    return np.load(train_responses), np.load(test_responses)


def get_reversing_grating_data(
    dataset: str,
) -> dict[str, np.ndarray]:
    """
    Load the reversing grating data for the specified dataset.

    :param dataset: The name of the dataset to load.
    :type dataset: str

    :return: A dictionary containing the reversing grating data.
    :rtype: dict[str, np.ndarray]
    """
    dataset_path = DATA_PATH / "reversing_grating" / dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset} not found in {dataset_path.parent}")
    else:
        responses_array = np.load(dataset_path / "responses.npy", allow_pickle=True)
        stimulus_array = np.load(dataset_path / "stimulus_array.npy")
        frametimes_array = np.load(dataset_path / "frametimes_array.npy")

        return {
            "responses": responses_array,
            "stimulus": stimulus_array,
            "frametimes": frametimes_array
        }


def get_cell_classification(
    dataset: str,
) -> dict[str, list]:
    """
    Load the cell classification data for the specified dataset.

    :param dataset: The name of the dataset to load.
    :type dataset: str

    :return: A dictionary containing the cell classification data.
    :rtype: dict[str, list]
    """
    dataset_path = DATA_PATH / "cell_classification" / dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset} not found in {dataset_path.parent}")
    else:
        with open(dataset_path / "cell_classification.json", "r") as f:
            cell_classification = json.load(f)
        return cell_classification
