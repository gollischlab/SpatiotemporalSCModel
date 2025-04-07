import glob
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
try:
    import cupy as cp
    _HAVE_CUPY = True
except ImportError:
    import numpy as cp
    _HAVE_CUPY = False
from numba import njit, prange
from tqdm import tqdm


def _get(arr: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
    if isinstance(arr, cp.ndarray):
        return arr.get()
    else:
        return arr


@njit
def _normalize(arr: np.ndarray, mean: np.ndarray):
    # essentially compute Weber contrast pixel-by-pixel
    return (arr - mean) / mean


@njit
def _crop_frame(frame: np.ndarray, center_x: int, center_y: int, spatial_kernel_size: int):
    return frame[
           center_y - spatial_kernel_size:center_y + spatial_kernel_size,
           center_x - spatial_kernel_size:center_x + spatial_kernel_size,
    ]


@njit
def _get_trial_start_end(trial_id: int, test_frames: int, train_frames: int):
    start_frame = test_frames + trial_id * train_frames
    end_frame = start_frame + train_frames

    return start_frame, end_frame


@njit(parallel=True)
def _load_to_cache(
    storage_array: np.ndarray,
    num_train_frames: int,
    num_test_frames: int,
    images: np.ndarray,
    fixation_frames: np.ndarray,
    fixation_x: np.ndarray,
    fixation_y: np.ndarray,
    frame_width: int,
    frame_height: int,
    flips: np.ndarray,
    stimulus_mean: np.ndarray,
    downsample: int,
    normalize_frame: bool = True,
    center_x: int = None,
    center_y: int = None,
    spatial_crop_size: int = None,
):
    for trial_id in prange(storage_array.shape[0]):
        start_frame, end_frame = _get_trial_start_end(
            trial_id=trial_id,
            test_frames=num_test_frames,
            train_frames=num_train_frames,
        )

        for j, frame_id in enumerate(range(start_frame, end_frame)):
            frame = _get_frame(
                frame_id=frame_id,
                images=images,
                fixation_frames=fixation_frames,
                fixation_x=fixation_x,
                fixation_y=fixation_y,
                frame_width=frame_width,
                frame_height=frame_height,
                flips=flips,
                stimulus_mean=stimulus_mean,
                downsample=downsample,
                normalize_frame=normalize_frame,
            )
            frame = _crop_frame(
                frame=frame,
                center_x=center_x,
                center_y=center_y,
                spatial_kernel_size=spatial_crop_size,
            )
            storage_array[trial_id, j] = frame

    return storage_array


@njit
def _get_frame(
    frame_id: int,
    images: np.ndarray,
    fixation_frames: np.ndarray,
    fixation_x: np.ndarray,
    fixation_y: np.ndarray,
    frame_width: int,
    frame_height: int,
    flips: np.ndarray,
    stimulus_mean: np.ndarray,
    downsample: int,
    normalize_frame: bool = True,
):
    output_frame = np.ones((frame_height, frame_width), dtype=np.float32) * 127.

    frame_ymin, frame_xmin = 0, 0
    frame_ymax, frame_xmax = output_frame.shape

    image = images[fixation_frames[frame_id]]
    image_height, image_width = image.shape
    center_x = fixation_x[frame_id]
    center_y = fixation_y[frame_id]

    movie_xmin = center_x - frame_width // 2
    movie_xmax = center_x + frame_width // 2
    movie_ymin = center_y - frame_height // 2
    movie_ymax = center_y + frame_height // 2

    if movie_xmin < 0:
        x_diff = 0 - movie_xmin
        frame_xmin = x_diff
        movie_xmin = 0
    if movie_xmax >= image_width:
        x_diff = movie_xmax - image_width
        frame_xmax = frame_xmax - x_diff
        movie_xmax = image_width

    if movie_ymin < 0:
        y_diff = 0 - movie_ymin
        frame_ymin = y_diff
        movie_ymin = 0
    if movie_ymax >= image_height:
        y_diff = movie_ymax - image_height
        frame_ymax = frame_ymax - y_diff
        movie_ymax = image_height

    output_frame[
        frame_ymin:frame_ymax, frame_xmin:frame_xmax
    ] = image[
            movie_ymin:movie_ymax, movie_xmin:movie_xmax
        ]

    if flips[frame_id]:
        output_frame = np.flipud(output_frame)

    if normalize_frame:
        output_frame = _normalize(output_frame, stimulus_mean)

    return output_frame[::downsample, ::downsample]


class NaturalMovieLoader:
    """
    Class for loading natural movie stimuli from disk and providing access to
    individual frames and trials. The class supports loading data into
    GPU memory using CuPy for faster processing. The class also provides
    functionality for cropping frames around a specified receptive field
    center and normalizing frames based on a specified mean stimulus.

    Attributes:
        images_path (Union[Path, str]): Path to the directory containing the raw image files.
        image_height (int): Height of the images.
        image_width (int): Width of the images.
        frame_height (int): Height of the frames.
        frame_width (int): Width of the frames.
        fixation_dict (dict): Dictionary containing fixation information.
        stimulus_mean (Optional[np.ndarray]): Mean stimulus for normalization.
        temporal_kernel_size (int): Size of the temporal kernel.
        spatial_kernel_size (int): Size of the spatial kernel.
        train_trials (List): List of training trials.
        train_frames (int): Number of frames in each training trial.
        test_frames (int): Number of frames in each test trial.
        total_trials (int): Total number of trials.
        downsample (int): Downsampling factor for frames.
        validation_trials (Optional[List]): List of validation trials.
        images (np.ndarray): Array to hold the loaded images.
        stimulus_mean (np.ndarray): Mean stimulus for normalization.
        output_trial (np.ndarray): Array to hold the output trial.
        current_trial (int): Current trial index.
        _trial_cache (Optional[np.ndarray]): Cache for storing trials.

    Methods:
        load_data_from_disk(): Load image data from disk into memory.
        load_trials_to_cache(): Load trials into cache for faster access.
        load_cache_to_device(): Load the trial cache to GPU memory.
        get_frame(frame_id: int, normalize_frame: bool = True) -> np.ndarray: Get a specific frame by its ID.
        get_trial_start_end(trial_id: int): Get the start and end frame IDs for a specific trial.
        get_trial(trial_id: int): Get the frames for a specific trial.
        get_validation_set(): Get the validation set of trials.
        get_test_set(): Get the test set of trials.
        rf_center: tuple: Get or set the receptive field center.
        temporal_kernel_size: int: Get or set the size of the temporal kernel.
        spatial_kernel_size: int: Get or set the size of the spatial kernel.
        shape: tuple: Get the shape of the training data.
        loop_frames(start_trial: int = None, start_frame: int = None, num_frames: int = None): Loop through frames.
        crop_frame(frame: np.ndarray): Crop a frame around the receptive field center.
    """
    def __init__(
            self,
            images_path: Union[Path, str],
            image_height: int,
            image_width: int,
            frame_height: int,
            frame_width: int,
            fixation_dict: dict,
            stimulus_mean: Optional[np.ndarray],
            temporal_filter_size: int,
            spatial_filter_size: int,
            train_trials: List,
            train_frames: int = 25500,
            test_frames: int = 5100,
            total_trials: int = 10,
            downsample: int = 1,
            validation_trials: Optional[List] = None,
    ):
        self.image_filenames: list = sorted(glob.glob((images_path / "*_img_*.raw").as_posix()))
        self.frame_ids: list = [int(Path(fname).name[:5]) for fname in self.image_filenames]
        self.image_ids: list = [Path(fname).name[10:15] for fname in self.image_filenames]

        self.downsample: int = downsample

        self.image_height: int = image_height
        self.image_width: int = image_width

        self.frame_height: int = frame_height
        self.frame_width: int = frame_width

        self.rf_center_x: int = self.frame_width // 2
        self.rf_center_y: int = self.frame_height // 2

        self.temp_filt_size: int = temporal_filter_size
        self.spat_filt_size: int = spatial_filter_size

        self.fixation_frames: np.ndarray = np.array(fixation_dict["frame_id"])
        self.fixation_x: np.ndarray = np.array(fixation_dict["center_x"])
        self.fixation_y: np.ndarray = np.array(fixation_dict["center_y"])
        # flips from the fixation file are inverted to account for the unconventional int to bool representation
        self.flips: np.ndarray = ~np.array(fixation_dict["flip"], dtype=bool)
        self.total_fixations: int = self.fixation_frames.size

        self.train_frames: int = train_frames
        self.test_frames: int = test_frames
        self.total_frames_per_trial: int = self.train_frames + self.test_frames
        self.total_trials: int = total_trials

        self.validation_trials: list = [] if validation_trials is None else validation_trials
        self.train_trials: list = train_trials

        self.images: np.ndarray = np.zeros(
            shape=(
                len(self.image_filenames),
                self.image_height,
                self.image_width,
            ),
            dtype=np.uint8,
        )
        self.stimulus_mean = stimulus_mean if stimulus_mean is not None \
            else np.zeros((self.frame_height, self.frame_width))

        self.load_data_from_disk()

        self.output_trial: np.ndarray = np.zeros(
            (self.train_frames, 2 * self.spatial_filter_size, 2 * self.spatial_filter_size),
        )

        self.current_trial: int = 0

        self._trial_cache = None

    def load_data_from_disk(self):
        """
        Load the image data from disk into memory.

        The images are loaded sequentially into a numpy array and reshaped to the
        specified height and width. It is assumed that they are stored on disk as
        binary image files in Fortran order (column-major order) in the 8-bit
        unsigned integer format.

        :return:
            None
        """
        for f, filename in tqdm(
            enumerate(self.image_filenames),
            total=len(self.image_filenames),
            desc="Loading images",
        ):
            image: np.ndarray = np.fromfile(
                filename,
                dtype=np.uint8
            ).reshape(
                (self.image_height, self.image_width),
                order="F",
            )

            self.images[f] = image

    def load_trials_to_cache(self):
        """
        Load the trials into cache for faster access.

        The trials are loaded into a numpy array and cropped around the specified
        receptive field center. The frames are normalized based on the specified
        mean stimulus. The trials are stored in a cache for faster access during
        training and testing. The cache is a 4D numpy array with dimensions
        (num_trials, num_frames, height, width). The trials are loaded in parallel
        using Numba's njit and prange functions for faster processing.

        :return:
            None
        """
        total_trials_to_cache = max(self.train_trials + self.validation_trials) + 1
        self._trial_cache = np.ones(
            (
                total_trials_to_cache,
                self.train_frames,
                2 * self.spatial_filter_size,
                2 * self.spatial_filter_size,
            ),
            dtype=np.float32,
        ) * 0.5

        self._trial_cache = _load_to_cache(
            storage_array=self._trial_cache,
            num_train_frames=self.train_frames,
            num_test_frames=self.test_frames,
            images=self.images,
            fixation_frames=self.fixation_frames,
            fixation_x=self.fixation_x,
            fixation_y=self.fixation_y,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            flips=self.flips,
            stimulus_mean=self.stimulus_mean,
            downsample=self.downsample,
            normalize_frame=True,
            center_x=self.rf_center_x,
            center_y=self.rf_center_y,
            spatial_crop_size=self.spatial_filter_size,
        )

    def load_cache_to_device(self):
        """
        Load the trial cache to GPU memory using CuPy.

        The trial cache is loaded into GPU memory for faster processing. This
        requires CuPy to be installed. If CuPy is not installed or if the trial cache
        is empty, a warning is raised.

        :raises RuntimeWarning:
            If CuPy is not installed or if the trial cache is empty.

        :return:
            None
        """
        if not _HAVE_CUPY:
            raise RuntimeWarning("Cupy not installed. Cannot load cache to device!")
        elif not self._trial_cache.size > 0:
            raise RuntimeWarning("No trials loaded to cache!")
        else:
            self._trial_cache = cp.asarray(self._trial_cache)

    def get_frame(self, frame_id: int, normalize_frame: bool = True) -> np.ndarray:
        """
        Get a specific frame by its ID.

        The frame is loaded from the image data based on its frame ID, which is the
        index of the fixation frame in the fixation dictionary, and cropped around
        the receptive field center. It is normalized based on the specified mean stimulus.
        The cropped frame is returned as a numpy array.

        :param frame_id:
            The ID of the frame to be retrieved.
        :param normalize_frame:
            If True, the frame is normalized based on the mean stimulus.

        :return:
            The cropped and normalized frame as a numpy array.
        """
        out = _get_frame(
            frame_id=frame_id,
            images=self.images,
            fixation_frames=self.fixation_frames,
            fixation_x=self.fixation_x,
            fixation_y=self.fixation_y,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            flips=self.flips,
            stimulus_mean=self.stimulus_mean,
            downsample=self.downsample,
            normalize_frame=normalize_frame,
        )

        return out

    def get_trial_start_end(self, trial_id: int):
        """
        Get the start and end frame IDs for a specific trial.

        The start and end frame IDs are calculated based on the trial ID, the number
        of training frames, and the number of test frames. The start frame ID is
        the index of the first frame in the trial, and the end frame ID is the index
        of the last frame in the trial. The start and end frame IDs are returned as
        a tuple.

        :param trial_id:
            The ID of the trial for which to get the start and end frame IDs.
        :return:
            A tuple containing the start and end frame IDs for the specified trial.
        """
        start_frame, end_frame = _get_trial_start_end(
            trial_id=trial_id,
            test_frames=self.test_frames,
            train_frames=self.train_frames,
        )

        return start_frame, end_frame

    def get_trial(self, trial_id: int):
        """
        Get the frames for a specific trial.

        The frames are loaded from the trial cache if available, or from the image
        data if the trial cache is not available. The frames are cropped around the
        receptive field center and returned as a numpy array.

        :param trial_id:
            The ID of the trial for which to get the frames.
        :return:
            The frames for the specified trial as a numpy array.
        """
        if trial_id not in self.train_trials + self.validation_trials:
            raise ValueError(
                f"Requested `trial_id` ({trial_id}) not in training or validation trials!"
            )
        if self._trial_cache is None:
            start_frame, end_frame = self.get_trial_start_end(trial_id)

            for i, frame_id in enumerate(range(start_frame, end_frame)):
                frame = self.get_frame(frame_id)
                self.output_trial[i] = self.crop_frame(frame)
            return self.output_trial
        else:
            return self._trial_cache[trial_id]

    def get_validation_set(self):
        """
        Get the validation set of trials.

        The validation set is a subset of the training trials that are used for
        validation during training. The validation set is loaded from the trial
        cache if available, or from the image data if the trial cache is not
        available. The frames are cropped around the receptive field center and
        returned as a numpy array.

        :return:
            The frames for the validation set as a numpy array.
        """
        if not self.validation_trials:
            raise RuntimeError("Dataloader was not loaded with validation trials. No validation set to be returned!")

        output_array = np.ones(
            (
                len(self.validation_trials),
                self.train_frames,
                2 * self.spatial_filter_size,
                2 * self.spatial_filter_size,
            ),
            dtype=np.float32,
        ) * 0.5
        for t, trial_id in enumerate(self.validation_trials):
            output_array[t] = _get(self.get_trial(trial_id))

        return output_array

    def get_test_set(self):
        """
        Get the test set of trials.

        The test set is a subset of the training trials that are used for
        testing the model. The frames are cropped around the receptive field
        center and returned as a numpy array.

        :return:
            The frames for the test set as a numpy array.
        """
        output_array = np.ones(
            (
                1,
                self.test_frames,
                2 * self.spatial_filter_size,
                2 * self.spatial_filter_size,
            ),
            dtype=np.float32,
        ) * 0.5
        for frame_id in range(self.test_frames):
            frame = self.get_frame(frame_id)
            output_array[0, frame_id] = self.crop_frame(frame)

        return output_array

    @property
    def rf_center(self):
        """
        Get or set the receptive field center.

        The receptive field center is a tuple containing the (y, x) coordinates
        of the center of the receptive field. The coordinates are in pixel units
        and are used to crop the frames around the receptive field center.

        :return:
            A tuple containing the (y, x) coordinates of the receptive field center.
        """
        return self.rf_center_y, self.rf_center_x

    @rf_center.setter
    def rf_center(self, value):
        """
        Set the receptive field center.

        The receptive field center is a tuple containing the (y, x) coordinates
        of the center of the receptive field. The coordinates are in pixel units
        and are used to crop the frames around the receptive field center. The
        coordinates must be integers. If the coordinates are not integers, a
        ValueError is raised. If the coordinates are valid, the receptive field
        center is set to the specified coordinates and the trial cache is cleared.

        :param value:
            A tuple containing the (y, x) coordinates of the receptive field center.

        :raises ValueError:
            If the coordinates are not integers or if the tuple is not of length 2.

        :return:
            None
        """
        if isinstance(value, tuple):
            if len(value) == 2:
                if all([isinstance(val, (int, np.integer)) for val in value]):
                    self.rf_center_y = value[0]
                    self.rf_center_x = value[1]

                    del self._trial_cache

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
        Get or set the size of the temporal filter.

        The size specifies the number of time points in the past to include in the
        temporal filter when extracting it from the spike-triggered average.

        :return:
            The size of the temporal filter.
        """
        return self.temp_filt_size

    @temporal_filter_size.setter
    def temporal_filter_size(self, value):
        """
        Set the size of the temporal filter.

        The size specifies the number of time points in the past to include in the
        temporal filter when extracting it from the spike-triggered average. The
        size must be an integer. If the size is not an integer, a ValueError is
        raised. If the size is valid, the temporal filter size is set to the
        specified size and the trial cache is cleared.

        :param value:
            The size of the temporal filter.

        :raises ValueError:
            If the size is not an integer.

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
        Get or set the size of the spatial filter.

        The size specifies the number of pixels to each side of the receptive
        field center when extracting the spatial filter from the spike-triggered
        average.

        :return:
            The size of the spatial filter.
        """
        return self.spat_filt_size

    @spatial_filter_size.setter
    def spatial_filter_size(self, value):
        """
        Set the size of the spatial filter.

        The size specifies the number of pixels to each side of the receptive
        field center when extracting the spatial filter from the
        spike-triggered average. The size must be an integer. If the size is not
        an integer, a ValueError is raised. If the size is valid, the spatial
        filter size is set to the specified size and the trial cache is cleared.

        :param value:
            The size of the spatial filter.

        :raises ValueError:
            If the size is not an integer.

        :return:
            None
        """
        if isinstance(value, (int, np.integer)):
            self.spat_filt_size = value

            del self.output_trial
            del self._trial_cache

            self.output_trial = np.zeros(
                (self.train_frames, 2 * self.spatial_filter_size, 2 * self.spatial_filter_size),
                dtype=np.float32
            )
            self._trial_cache = None
        else:
            raise ValueError("Spatial kernel size must be an integer!")

    @property
    def shape(self) -> tuple:
        """
        Get the shape of the training data.

        :return:
            A tuple containing the number of training trials, the number of frames,
            the height of the frames, and the width of the frames.
        """
        return (
            len(self.train_trials),
            self.train_frames,
            self.frame_height // self.downsample,
            self.frame_width // self.downsample
        )

    def loop_frames(self, start_trial: int = None, start_frame: int = None, num_frames: int = None):
        """
        Loop through frames in the training trials.

        The frames are cropped around the receptive field center and returned
        as a generator. The generator yields frames one by one, allowing for
        efficient processing of large datasets. The start trial, start frame,
        and number of frames can be specified to control the range of frames
        to be processed. If the start trial or start frame is not specified,
        the first trial and first frame are used. If the number of frames is
        not specified, all frames from the start frame to the end of the trial
        are processed. The frames are cropped around the receptive field center
        and returned as a numpy array.

        :param start_trial:
            The ID of the trial to start from.
        :param start_frame:
            The frame ID to start from.
        :param num_frames:
            The number of frames to process.

        :return:
            A generator yielding the cropped frames one by one.
        """
        if start_trial is None:
            start_trial = 0

        if start_frame is None:
            start_frame = 0

        if num_frames is None:
            num_frames = self.train_frames - start_frame

        if num_frames < 1:
            raise ValueError("Number of requested frames must atleast be 1!")

        first_frame_in_trial, _ = self.get_trial_start_end(trial_id=start_trial)

        start_point = first_frame_in_trial + start_frame

        for i, frame_id in enumerate(range(start_point, start_point + num_frames)):
            frame = self.get_frame(frame_id)
            frame = self.crop_frame(frame)
            yield frame

    def crop_frame(self, frame: np.ndarray):
        """
        Crop a frame around the receptive field center.

        :param frame:
            The frame to be cropped.
        :return:
            The cropped frame as a numpy array.
        """
        return _crop_frame(
            frame=frame,
            center_x=self.rf_center_x,
            center_y=self.rf_center_y,
            spatial_kernel_size=self.spatial_filter_size,
        )

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
            trial_id = np.array(self.train_trials)[item]

            start_frame, end_frame = self.get_trial_start_end(trial_id)

            if self._trial_cache is None:
                for i, frame_id in enumerate(range(start_frame, end_frame)):
                    frame = self.get_frame(frame_id)
                    yield self.crop_frame(frame)
            else:
                for frame in self._trial_cache[trial_id]:
                    yield frame

        elif isinstance(item, slice):
            # if a slice is requested, slice over the list of training trials
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop <= len(self.train_trials) else len(self.train_trials)
            step = item.step if item.step is not None else 1

            if start >= stop:
                raise RuntimeError("Given slice selects none of the training trials!")

            for t, trial_id in enumerate(np.array(self.train_trials, dtype=int)[start:stop:step]):
                yield self.get_trial(trial_id)

        elif isinstance(item, (list, np.ndarray)):
            # if a list or np.ndarray requested, return training trials corresponding to those indices
            if isinstance(item, list):
                assert all([isinstance(val, int) for val in item]), "Requested list of indices must contain integers!"
                assert max(item) < self.total_trials, IndexError(f"Index `{max(item)}` is larger than number of trials!")

                num_trials = len(item)
                assert num_trials < self.total_trials, "Requested list has more indices than total trials!"

            elif isinstance(item, np.ndarray):
                assert np.issubdtype(item, np.integer), "Requested array of indices must be of integer dtype!"
                assert item.ndim == 1, "Requested array of indices must be 1-dimensional!"
                assert item.max() < self.total_trials, IndexError(f"Index `{item.max()}` is larger than number of trials!")

                num_trials = item.size
                assert num_trials < self.total_trials, "Requested list has more indices than total trials!"

            else:
                raise RuntimeError("Strange condition encountered... Please debug!")

            for t, trial_id in enumerate(np.array(self.train_trials, dtype=int)[item]):
                yield self.get_trial(trial_id)

        else:
            raise ValueError(
                f"Invalid item {item} requested! Only `int`, `list`, `np.ndarray` or `slice` objects supported."
            )
