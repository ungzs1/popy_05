"""
Global variables and project configuration settings are defined here.

Access these variables by importing this module and using the dot notation.
For example, to access the PROJECT_PATH variable, use the following code:

import config
path = config.PROJECT_PATH
"""

import os
import xarray as xr

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


### PREPROCESSING UNTILS ###
# sampling rate of spikes and behav during preprocessing
PREPROCESSING_SAMPLING_RATE = 1000

### PLOTTING ###
import numpy as np

COLORS = {
    'LPFC': 'tab:blue',
    'MCC': 'grey',
    'ka': 'tab:purple',
    'po': 'tab:green',
    'simulation_ka': 'tab:orange',
    'simulation_po': 'tab:red',
    }

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
