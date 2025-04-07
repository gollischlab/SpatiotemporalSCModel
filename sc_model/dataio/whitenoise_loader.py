from pathlib import Path
from typing import Union, Optional, List

import numpy as np
try:
    import cupy as cp
    _HAVE_CUPY = True
    from cupy import asnumpy as asarray
except ImportError:
    _HAVE_CUPY = False
    import numpy as cp
    from numpy import asarray


class WhitenoiseLoader(object):
    """
    Class for loading and processing white noise stimuli.

    This class handles the loading of white noise stimuli from a specified folder,
    including both repeating and non-repeating stimuli. It provides methods for
    accessing individual trials, frames, and cropping the stimuli to a specified
    receptive field center. The class also supports caching of trials for faster
    access and can load the stimuli to a GPU device if available.

    Attributes:
        stimulus_folder (Path): Path to the folder containing the stimulus data.
        frame_height (int): Height of the stimulus frames.
        frame_width (int): Width of the stimulus frames.
        downsample (int): Downsampling factor for the stimuli.
        rf_center_x (int): X-coordinate of the receptive field center.
        rf_center_y (int): Y-coordinate of the receptive field center.
        temp_filt_size (int): Size of the temporal kernel.
        spat_filt_size (int): Size of the spatial kernel.
        train_frames (int): Number of training frames per trial.
        test_frames (int): Number of test frames per trial.
        total_trials (int): Total number of trials available in the dataset.
        validation_trials (List[int]): List of validation trial IDs.
        train_trials (List[int]): List of training trial IDs.
        all_trials (np.ndarray): Array containing all trials loaded from disk.
        test_set (np.ndarray): Array containing the test set loaded from disk.
        _trial_cache (Optional[np.ndarray, cp.ndarray]): Cached trials for faster access.
        current_trial (int): Index of the current trial being accessed.

    Methods:
        get_trial(trial_id: int) -> np.ndarray:
            Returns the specified trial as a cropped array.
        get_frame(frame_id: int) -> np.ndarray:
            Returns the specified frame from the test set or training trials.
        crop_frame(frame: np.ndarray) -> np.ndarray:
            Crops the given frame to the specified receptive field center.
        load_trials_to_cache() -> None:
            Loads all trials into cache for faster access.
        load_all_trials() -> None:
            Loads all trials from disk into memory.
        load_cache_to_device() -> None:
            Transfers the cached trials to a GPU device if available.
        load_test_set() -> None:
            Loads the test set from disk into memory.
        get_validation_set() -> cp.ndarray or np.ndarray:
            Returns the validation set as a cropped array.
        get_test_set() -> cp.ndarray or np.ndarray:
            Returns the test set as a cropped array.
    """
    def __init__(
            self,
            stimulus_folder: Union[Path, str],
            frame_height: int,
            frame_width: int,
            temporal_filter_size: int,
            spatial_filter_size: int,
            train_frames: int,
            test_frames: int,
            total_trials: int,
            train_trials: List,
            validation_trials: Optional[List] = None,
    ):
        self.stimulus_folder: Path = Path(stimulus_folder)
        self.non_repeating_stimulus_folder: Path = self.stimulus_folder / "non_repeating_stimuli"

        self.frame_height: int = frame_height
        self.frame_width: int = frame_width

        self.downsample: int = 1

        self.rf_center_x: int = self.frame_width // 2
        self.rf_center_y: int = self.frame_height // 2

        self.temp_filt_size: int = temporal_filter_size
        self.spat_filt_size: int = spatial_filter_size

        self.train_frames: int = train_frames
        self.test_frames: int = test_frames
        self.total_frames_per_trial: int = self.train_frames + self.test_frames
        self.total_trials: int = total_trials

        self.validation_trials: List[int] = [] if validation_trials is None else validation_trials
        self.train_trials: List[int] = train_trials

        self.all_trials: np.ndarray = np.zeros(
            (
                max(self.train_trials + self.validation_trials) + 1,
                self.train_frames,
                self.frame_height,
                self.frame_width
            ),
            dtype=np.int8,
        )
        self.test_set: np.ndarray = np.zeros(
            (1, self.test_frames, self.frame_height, self.frame_width),
            dtype=np.int8,
        )

        self._trial_cache: Optional[np.ndarray, cp.ndarray] = None

        self.load_all_trials()
        self.load_test_set()

        self.current_trial: int = min(self.train_trials)

    def get_trial(self, trial_id: int):
        """
        Returns the specified trial as a cropped array.

        :param trial_id:
            The ID of the trial to be retrieved.
        :return:
            The cropped trial as a numpy array.
        """
        if trial_id not in self.train_trials + self.validation_trials:
            raise ValueError(
                f"Requested `trial_id` ({trial_id}) not found in training or validation trials!"
            )
        if self._trial_cache is not None:
            return self._trial_cache[trial_id]
        else:
            self.crop_trial(self.all_trials[trial_id])

    def get_frame(self, frame_id):
        """
        Returns the specified frame from the test set or training trials.

        :param frame_id:
            The ID of the frame to be retrieved.
        :return:
            The cropped frame as a numpy array.
        """
        if frame_id < self.test_frames:
            return self.test_set[0, frame_id]
        else:
            frame_id -= self.test_frames
            trial_id = frame_id // self.train_frames
            frame_pos = frame_id % self.train_frames
            return self.all_trials[trial_id, frame_pos]

    def crop_frame(self, frame: np.ndarray):
        """
        Crops the given frame to the specified receptive field center.

        This method is used to extract a square region around the receptive field center
        from the given frame. The size of the cropped region is determined by the spatial filter size.

        :param frame:
            The frame to be cropped.
        :return:
            The cropped frame as a numpy array.
        """
        return frame[
            self.rf_center_y - self.spatial_filter_size:self.rf_center_y + self.spatial_filter_size,
            self.rf_center_x - self.spatial_filter_size:self.rf_center_x + self.spatial_filter_size,
        ]

    def load_trials_to_cache(self):
        """
        Loads all trials into cache for faster access.

        This method pre-computes the cropped trials and stores them in a cache for faster access
        during training and testing. The trials are cropped to the specified receptive field center
        and the size of the cropped region is determined by the spatial filter size.

        :return:
            None
        """
        self._trial_cache = np.zeros(
            (
                self.all_trials.shape[0],
                self.train_frames,
                2 * self.spat_filt_size,
                2 * self.spat_filt_size,
            ),
            dtype=np.float32,
        )

        for t, trial in enumerate(self.all_trials):
            self._trial_cache[t] = self.crop_trial(trial)

    def load_all_trials(self):
        """
        Loads all trials from disk into memory.
        This method reads the non-repeating stimuli from the specified folder and loads them into memory.

        :raises FileNotFoundError:
            If the specified trial file does not exist.

        :return:
            None
        """
        for trial_id in range(max(self.train_trials + self.validation_trials) + 1):
            data_file = self.stimulus_folder / "non_repeating_stimuli" / f"trial_{trial_id:02d}" / "frames.npy"
            if data_file.exists():
                self.all_trials[trial_id] = np.load(data_file)
            else:
                raise FileNotFoundError(f"{data_file} does not exist.")

    def load_cache_to_device(self):
        """
        Transfers the cached trials to a GPU device if available.

        This method uses CuPy to transfer the cached trials to the GPU for faster processing.
        It raises a warning if CuPy is not installed or if no trials are loaded to the cache.

        :raises RuntimeWarning:
            If CuPy is not installed or if no trials are loaded to the cache.

        :return:
            None
        """
        if not _HAVE_CUPY:
            raise RuntimeWarning("Cupy not installed. Cannot load cache to device!")
        elif not self.all_trials.size > 0:
            raise RuntimeWarning("No trials loaded to cache!")
        else:
            self._trial_cache = cp.asarray(self._trial_cache)

    def load_test_set(self):
        """
        Loads the test set from disk into memory.

        This method reads the repeating stimuli from the specified folder and loads them into memory.

        :raises FileNotFoundError:
            If the specified test set file does not exist.

        :return:
            None
        """
        test_set_path = self.stimulus_folder / "repeating_stimuli.npy"
        if test_set_path.exists():
            self.test_set = np.load(test_set_path)[None, ...]
        else:
            raise FileNotFoundError(f"{test_set_path} does not exist.")

    def get_validation_set(self):
        """
        Returns the validation set as a cropped array.
        This method crops the validation trials to the specified receptive field center
        and returns them as a numpy array. The size of the cropped region is determined
        by the spatial filter size.

        :raises RuntimeError:
            If the dataloader was not loaded with validation trials.

        :return:
            The cropped validation set as a numpy array.
        """
        if not self.validation_trials:
            raise RuntimeError("Dataloader was not loaded with validation trials. No validation set to be returned!")

        output_array = cp.ones(
            (
                len(self.validation_trials),
                self.train_frames,
                2 * self.spatial_filter_size,
                2 * self.spatial_filter_size,
            ),
            dtype=cp.float32,
        ) * 0.5

        for t, trial_id in enumerate(self.validation_trials):
            output_array[t] = self.get_trial(trial_id)

        return output_array

    def get_test_set(self):
        """
        Returns the test set as a cropped array.

        This method crops the test set to the specified receptive field center
        and returns it as a numpy array. The size of the cropped region is determined
        by the spatial filter size.

        :return:
            The cropped test set as a numpy array.
        """
        return self.crop_trial(self.test_set)

    @property
    def rf_center(self):
        """
        Returns the receptive field center as a tuple of (center_y, center_x).

        The center coordinates are stored as integers and represent the
        position of the receptive field center in the stimulus frame.

        :return:
            A tuple containing the y and x coordinates of the receptive field center.
        """
        return self.rf_center_y, self.rf_center_x

    @rf_center.setter
    def rf_center(self, value):
        """
        Sets the receptive field center to the specified (center_y, center_x) coordinates.
        The coordinates must be provided as a 2-tuple of integers. If the provided
        coordinates are not integers or if the length of the tuple is not 2,
        a ValueError is raised.

        :param value:
            A 2-tuple containing the y and x coordinates of the receptive field center.

        :raises ValueError:
            If the provided coordinates are not integers or if the length of the tuple is not 2.

        :return:
            None
        """
        if isinstance(value, tuple):
            if len(value) == 2:
                if all([isinstance(val, (int, np.integer)) for val in value]):
                    self.rf_center_y = value[0]
                    self.rf_center_x = value[1]
                    self._trial_cache = None
                else:
                    raise ValueError("center_x and center_y values must be integers!")
            else:
                raise ValueError("rf_center must be a 2-tuple `(center_y, center_x)`!")
        else:
            raise ValueError("rf_center must be a 2-tuple `(center_y, center_x)`!")

    @property
    def temporal_filter_size(self):
        """
        Returns the size of the temporal filter.

        The temporal filter size is an integer that determines the number of frames
        used in the temporal filter. It is used to crop the temporal dimension of
        the STA to the specified size.

        :return:
            The size of the temporal filter as an integer.
        """
        return self.temp_filt_size

    @temporal_filter_size.setter
    def temporal_filter_size(self, value):
        """
        Sets the size of the temporal filter to the specified value.
        The value must be an integer. If the provided value is not an integer,
        a ValueError is raised.

        :param value:
            The size of the temporal filter as an integer.

        :raises ValueError:
            If the provided value is not an integer.

        :return:
            None
        """
        if isinstance(value, int):
            self.temp_filt_size = value
        else:
            raise ValueError("Temporal kernel size must be an integer!")

    @property
    def spatial_filter_size(self):
        """
        Returns the size of the spatial filter.

        The spatial filter size is an integer that determines the size of the
        square spatial filter used in the model. It is used to crop the spatial
        dimension of the STA to the specified size and is defined as the number of
        pixels to each side of the receptive field center. For example, value of
        20 means a spatial filter size of 40x40 pixels.

        :return:
            The size of the spatial filter as an integer.
        """
        return self.spat_filt_size

    @spatial_filter_size.setter
    def spatial_filter_size(self, value):
        """
        Sets the size of the spatial filter to the specified value.

        The value must be an integer. If the provided value is not an integer,
        a ValueError is raised.

        :param value:
            The size of the spatial filter as an integer.
        :raises ValueError:
            If the provided value is not an integer.
        :return:
            None
        """
        if isinstance(value, (int, np.integer)):
            self.spat_filt_size = value
            self._trial_cache = None
        else:
            raise ValueError("Spatial kernel size must be an integer!")

    @property
    def shape(self):
        """
        Returns the shape of the training trials.

        :return:
            A tuple containing the shape of the training trials.
            The shape is defined as (number of trials, number of frames, height, width).
        """
        return len(self.train_trials), self.train_frames, 2 * self.spat_filt_size, 2 * self.spat_filt_size,

    def crop_trial(self, trial: np.ndarray):
        """
        Crops the given trial to the specified receptive field center.

        :param trial:
            The trial to be cropped.
        :return:
            The cropped trial as a numpy array.
        """
        return trial[
            ...,
            self.rf_center_y - self.spatial_filter_size:self.rf_center_y + self.spatial_filter_size,
            self.rf_center_x - self.spatial_filter_size:self.rf_center_x + self.spatial_filter_size,
        ]

    def __iter__(self):
        self.current_trial = 0
        return self

    def __next__(self):
        if self.current_trial >= len(self.train_trials):
            raise StopIteration
        else:
            trial_frames = self.get_trial(self.train_trials[self.current_trial])
            self.current_trial += 1
            return trial_frames

    def __getitem__(self, item):
        if isinstance(item, int):
            # if a single integer index n requested, yield frames from the nth training trial
            if item >= len(self.train_trials):
                raise ValueError(
                    f"Requested trial id ({item}) higher than total number of training trials ({len(self.train_trials)})"
                )
            trial_id = self.train_trials[item]

            yield self.get_trial(trial_id)

        elif isinstance(item, slice):
            # if a slice is requested, slice over the list of training trials
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop <= len(self.train_trials) else len(self.train_trials)
            step = item.step if item.step is not None else 1

            if start >= stop:
                raise RuntimeError("Given slice selects none of the training trials!")

            for trial_id in np.array(self.train_trials)[start:stop:step]:
                yield self.get_trial(trial_id)

        elif isinstance(item, (list, np.ndarray)):
            # if a list or np.ndarray requested, return training trials corresponding to those indices
            if isinstance(item, list):
                assert all([isinstance(val, int) for val in item]), "Requested list of indices must contain integers!"
            elif isinstance(item, np.ndarray):
                assert np.issubdtype(item, np.integer), "Requested array of indices must be of integer dtype!"
                assert item.ndim == 1, "Requested array of indices must be 1-dimensional!"

            for trial_id in np.array(self.train_trials)[item]:
                yield self.get_trial(trial_id)

        else:
            raise ValueError(
                f"Invalid item {item} requested! Only `int`, `list`, `np.ndarray` or `slice` objects supported."
            )
