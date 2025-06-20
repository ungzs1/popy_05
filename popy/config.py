"""
Global variables and project configuration settings are defined here.

Access these variables by importing this module and using the dot notation.
For example, to access the PROJECT_PATH variable, use the following code:

import config
path = config.PROJECT_PATH
"""

import os
import xarray as xr
import numpy as np


### PATHS ###
pathes = {'OFFICE_windows': 'C:\\ZSOMBI\\OneDrive\\PoPy',
          #'OFFICE_mac': '/Users/zsombi/Library/CloudStorage/OneDrive-Personal/PoPy/',
          'OFFICE_mac': '/Users/zsombi/ZSOMBI/SBRI/PoPy',
          #'HOME': '/Users/zsombi/OneDrive/PoPy'}
          'HOME': '/Users/zsombi/Library/CloudStorage/GoogleDrive-uuungvarszi@gmail.com/Other computers/My Mac/ZSOMBI/SBRI/PoPy'}

# check which path exists on the local computer
for key, path in pathes.items():
    if os.path.exists(path):
        PROJECT_PATH_LOCAL = path
        #print(f'Using {key} path')
        break

# check if the drive is accessible
# the path of the drive
DRIVE_PATH = '\\sbri-share.adn.inserm.fr\\PROCYK\\PHD_Clément2\\LUV_Project\\Analysis\\files_already_sort'

# check if the drive is accessible
if os.path.exists(DRIVE_PATH):
    PROJECT_PATH_DRIVE = DRIVE_PATH
    #print('Drive is accessible')
else:
    PROJECT_PATH_DRIVE = None
    #print('Inserm drive is not accessible')

### PROCESSING NEUIRAL DATA ###

# sampling rate of spikes and behav during preprocessing
PREPROCESSING_SAMPLING_RATE = 1000

# for time normalization, we use the following epoch lengths
EPOCH_LENS = np.array([1., 1., 0.5, 0.5, 0.5, 4.])


### PLOTTING ###
COLORS = {
    'LPFC': 'tab:blue',
    'MCC': 'grey',
    'dLPFC': 'tab:blue',
    'vLPFC': 'tab:purple',
    'ka': 'tab:purple',
    'po': 'tab:green',
    'ka_simulation': 'tab:brown',
    'po_simulation': 'tab:red',
    'yu_DCZ': 'tab:blue',
    'yu_sham': 'tab:orange',
    1: '#238823', 0: '#D2222D',
    1.0: '#238823', 0.0: '#D2222D',
    }


# Optimal parameters for the models
MODEL_PARAMS = {'ka': {'alpha': 0.40963354578309075, 'V0': 0.11827469139505799, 'beta': 10.638021927670694},
                'po': {'alpha': 0.3037981353357479, 'V0': 0.18398040544466904, 'beta': 6.973374756267803},
                'yu_sham': {'alpha': 0.4513693025955279, 'V0': 0.16954116683423484, 'beta': 7.678119299510347},
                'yu_DCZ': {'alpha': 0.4942628307389808, 'V0': 0.1283662338249597, 'beta': 6.46241106842488}}
MODEL_PARAMS['ka_simulation'] = MODEL_PARAMS['ka']
MODEL_PARAMS['po_simulation'] = MODEL_PARAMS['po']
MODEL_PARAMS['yu_sham_simulation'] = MODEL_PARAMS['yu_sham']
MODEL_PARAMS['yu_DCZ_simulation'] = MODEL_PARAMS['yu_DCZ']

MODEL_PARAMS_RL = {'ka': {'alpha': 0.12667346795497197, 'beta': 8.706042488026716},
                   'po': {'alpha': 0.14989319361701048, 'beta': 4.613720171756842},
                   'yu_sham': {'alpha': 0.20112907842232014, 'beta': 5.335652953142322},
                   'yu_DCZ': {'alpha': 0.20897838964272342, 'beta': 4.78188888988298}}

# Cage plotting things
coord_chars = [chr(i) for i in range(97, 97+19)]

def_coords_ka = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.,
                    1., 0., 0.],
                [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
                    1., 1., 0.],
                [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                    1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1.,
                    0., 0., 0.],
                [0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1.,
                    0., 0., 0.],
                [0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.]]).T
def_coords_po = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 2.,
                    2., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 2., 1., 0., 0., 1.,
                    0., 1., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 2., 3., 1., 1., 0., 0.,
                    1., 0., 0.],
                [0., 0., 0., 0., 5., 2., 0., 0., 0., 0., 1., 1., 2., 1., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 1., 2., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.,
                    0., 0., 0.],
                [0., 0., 0., 1., 2., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1.,
                    0., 0., 0.],
                [0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 2., 1.,
                    0., 0., 0.],
                [0., 0., 0., 1., 2., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 3.,
                    0., 0., 0.],
                [0., 0., 0., 1., 2., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 2.,
                    1., 0., 0.],
                [0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                    2., 0., 0.],
                [0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                    2., 0., 0.],
                [0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    1., 2., 0.],
                [0., 0., 0., 1., 2., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                    0., 1., 0.],
                [0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.]])

GRID_DEFAULT = {'ka': xr.DataArray(def_coords_ka.T, 
                        coords={"x": coord_chars,
                            "y": coord_chars,
                            'loc_x': ('x', np.flip(np.arange(-9, 10, 1))),
                            'loc_y': ('y', np.arange(-9, 10, 1))},
                        dims=("x", "y"))
                ,
                "po": xr.DataArray(def_coords_po.T, 
                        coords={"x": coord_chars,
                            "y": coord_chars,
                            'loc_x': ('x', np.flip(np.arange(-9, 10, 1))),
                            'loc_y': ('y', np.arange(-9, 10, 1))},
                        dims=("x", "y"))
                }
