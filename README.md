# SpatiotemporalSCModel

Code repository to fit linear-nonlinear and spatial contrast models to responses of retinal ganglion cells under spatiotemporal stimulation. It accompanies the paper:

**Sridhar S, Vystrčilová M, Khani MH, Karamanlis D, Schreyer HM, Ramakrishna V, Krüppel S, Zapp SJ, Mietsch M, Ecker AS, Gollisch T: Modeling spatial contrast sensitivity in responses of primate retinal ganglion cells to natural movies.**

The code is designed to work with the data that was published along with the paper, which is available at [https://doi.org/10.12751/g-node.3dfiti](https://doi.org/10.12751/g-node.3dfiti). However, any dataset stored with the same structure can be loaded with this code.

The codebase is written in Python and is intended to be used as an installed Python package. It is designed to work efficiently on an NVIDIA GPU, but it can also be run on a CPU. The code is extensively documented. However, if you still have questions or problems, feel free to open an issue on GitHub, or contact us directly via email.

# Installation

## Setup the virtual environment

It is recommended to use a virtual environment to avoid conflicts with other Python packages. We provide instructions for setting up a virtual environment using `conda`, however, you can use a virtual environment manager of your choice.
 
The following command will set up the conda environment named `sc_model` and activate it:

```bash
conda create -n sc_model python=3.12
conda activate sc_model
```

## (Optional) CUDA and Cupy Installation

If you want to run the code on a GPU, you need to install CUDA on your system. You can find the installation instructions for your operating system on the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).

You then need to install the `cupy` package, which is a GPU-accelerated library for numerical computations. The installation instructions for `cupy` can be found on the [CuPy installation page](https://docs.cupy.dev/en/stable/install.html). 

**Note:** Make sure to install the version of `cupy` that matches your CUDA version. Also, if using a virtual environment manager, make sure to install `cupy` in the same virtual environment where you will be running the code.

## Install the required packages

To install the code in this repository along with the required packages, clone the repository and run the following command in the root directory:

```bash
pip install -e .
```

This will install the package named `sc_model` in editable mode, allowing you to make changes to the code without needing to reinstall the package.

# Usage

The installed package `sc_model` provides code to load data from the accompanying data repository and fit two models to it -- the linear-nonlinear (LN) model and the spatial contrast (SC) model. The model fitting scripts are designed to be run from the command line, but the relevant functions can also be imported and run in a Python shell (not shown below).

**Note:** for all following examples, we assume that the data repository is cloned into the same parent directory as this code repository i.e. the data loading functions look for the data repository in the parent directory. If you have cloned the data repository elsewhere, you must change the `DATA_REPO` path accordingly in the file `sc_model/utils/project_variables.py`.

## Fit the models

Assuming the package `sc_model` is installed as instructed above, navigate to the `sc_model/scripts/` folder and run the following command to fit the LN model to the data:

```bash
python run_ln_model.py \
--dataset 20220426_SS_252MEA6010_le_n3 \
--stimulus naturalistic_movies \
--stimulus_seed 1 \
--cell_id 100 \
--spatial_crop_size 80 \
--temporal_crop_size 30 \
--stimulus_smoothing 0.0 \
--sigpix_threshold 6.0
```

This command will fit the LN model to the responses of the cell `100` from the dataset `20220426_SS_252MEA6010_le_n3` to the naturalistic movies stimulus. The model will be fitted using a spatial filter size of `80` pixels, a temporal crop size of `30` frames, no stimulus smoothing, and a significance threshold of `6.0` for the spatial receptive field (RF) estimation.

Similarly, to fit the SC model, run the following command:

```bash
python run_sc_model.py \
--dataset 20220412_SN_252MEA6010_le_s4 \
--stimulus white_noise \
--stimulus_seed 0 \
--cell_id 100 \
--spatial_crop_size 20 \
--temporal_crop_size 30 \
--stimulus_smoothing 2.0 \
--sigpix_threshold 6.0
```

This command will fit the SC model to the responses of cell `100` from the dataset `20220412_SN_252MEA6010_le_s4` to spatiotemporal white-noise. The model will be fitted using a spatial crop size of `20` pixels, a temporal crop size of `30` frames, a stimulus smoothing of `2.0` pixels, and a significance threshold of `6.0` for the spatial RF estimation. 

Both scripts load the specified data (training and test stimuli, the cell's STA and responses, etc.), train the respective model on the training set and evaluate on the test set. A full description of what each parameter does can be found in the help message of each script. To see the help message, run the following command:

```bash
python run_ln_model.py --help
```
or 

```bash
python run_sc_model.py --help
```
The help message will show you all the available parameters and their default values. 

**Note on GPU usage:** if you have CUDA and cupy correctly set up, the models will fit the data using GPU routines. Since the data used for training can be quite large for most GPUs, the stimulus frames in each trial are broken up into smaller chunks. In case you still run into memory issues, you can reduce the size of these chunks by reducing the `MAX_FLOAT_SIZE` parameter in the file `sc_model/utils/project_variables.py`. 

## View the results

The results of the model fitting are saved in the `results` folder in the root directory of the repository. The results are saved in a series of subfolders corresponding to the model, dataset, stimulus and model parameters chosen. If, when running the model scripts, the `results` foldeer does not exist, it will be created automatically.

The results for each cell is stored in a `pickle` file. To load the results, you can run the following code in a Python shell:

```python
import pickle
with open("path/to/pickle/file.pkl", "rb") as f:
    res = pickle.load(f)
```

This will load a `res` dictionary that contains the fitted parameters, the correlation coefficient of the model on the test set, the intermediate signals computed during model fitting (convolution output for the LN model and the Imean and LSC values for the SC model) and other model parameters. For a complete description of the keys in the `res` dictionary, refer to the documentation of the functions in `sc_model/scripts/run_ln_model.py` and `sc_model/scripts/run_sc_model.py`.

## Load reversing grating data

This repository also contains convenience functions to load the stimulus, frametimes and response files for the reversing grating stimulus for each dataset. The data can be loaded using the following command:

```python
from sc_model.dataio import get_reversing_grating_data
rg_data = get_reversing_grating_data("20220412_SN_252MEA6010_le_s4")
```

This will return a dictionary containing the keys `responses`, `stimulus` and `frametimes`. Refer to the data manual provided with the data repository for a complete description of the data format.

## Load cell classification data

The results in the manuscript often contain comparisons across different cell types. These cell types are classified based on the characteristics of the cell's spatial receptive field and temporal dynamics. The classification is provided along with this repository in the form of `json` files for each dataset. These files can be conveniently viewed in any text editor, and can also be loaded into Python using the following command:

```python
from sc_model.dataio import get_cell_classification
cl_data = get_cell_classification("20220412_SN_252MEA6010_le_s4")
```

The dictionary `cl_data` contains a key for each cell class found in the requested dataset, and the corresponding value is a list containing the cell IDs of the cells belonging to that class.