import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors

from popy.neural_data_tools import *
from popy.behavior_data_tools import *
from popy.io_tools import load_metadata
import popy.config as cfg

import os


def _init_location_grid():
    """
    Helper function for plot_on_cortical_grid().

    Initialize a 19x19 grid with nans, and sets the coordinates of the grid (i.e. the letter and number of each cell),
    to matvh the coordinates of the cortical grid of Clement. From this point, the values can be filled in the grid
    using the letter or number coordinates.
    """
    
    # lat-med axis
    coords_num_x = np.arange(-9, 10)  
    coords_char_x = ['s','r','q','p','o','n','m','l','k','j','i','h','g','f','e','d','c','b','a']

    # ant-post axis
    coords_num_y = np.arange(-9, 10) 
    coords_chars_y = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s']

    # create empty grid
    matrix_nans = np.ones((len(coords_num_x), len(coords_num_y)))
    matrix_nans[:] = np.nan
    data_xr = xr.DataArray(matrix_nans, 
                           coords={"x": coords_num_x,
                                   "y": coords_num_y,
                                   'loc_x': ('x', coords_char_x),
                                   'loc_y': ('y', coords_chars_y)},
                           dims=("x", "y"),
                           attrs={'description': 'Cortical grid of Clement (with his wierd coding)',
                                  'x': 'lat-med axis coordinates, from lateral (most negative) to medial (most positive)',
                                  'y': 'ant-post axis coordinates, from anterior (most negative) to posterior (most positive)',
                                  'loc_x': 'lat-med axis letter codes, from lateral (last letter "s") to medial (first letter "a")',
                                  'loc_y': 'ant-post axis coordinates, from anterior (first letter "a") to posterior (last letter "s")'})
    
    return data_xr


def _get_grid_location(monkey, session, area):
    """
    Helper function for plot_on_cortical_grid().

    An easy way to get the grid location of a monkey and session. Returns the (x, y) coordinates of the grid (i.e. (med-lat, ant-post) coord).

    The positions are two letters, the first one is the medial-lateral axis, the second one is the anterior-posterior axis.

    e.g. ['a', 'a'] is the most medial and most anterior position, which translates to coordinater (9, -9) thanks to Clement...
    """
    
    # load metadata
    metadata = load_metadata()

    # get grid location
    position = metadata.loc[(metadata.monkey == monkey) & (metadata.session == session)][f'position_{area}'].values[0]

    if pd.isna(position):
        print(f"No position information for recording: monkey {monkey}, session {session}, area {area}")
        return np.nan

    decode_first_pos = {'a': 9, 'b': 8, 'c': 7, 'd': 6, 'e': 5, 'f': 4, 'g': 3, 'h': 2, 'i': 1, 'j': 0, 'k': -1, 'l': -2, 'm': -3, 'n': -4, 'o': -5, 'p': -6, 'q': -7, 'r': -8, 's': -9}
    decode_second_pos = {'a': -9, 'b': -8, 'c': -7, 'd': -6, 'e': -5, 'f': -4, 'g': -3, 'h': -2, 'i': -1, 'j': 0, 'k': 1, 'l': 2, 'm': 3, 'n': 4, 'o': 5, 'p': 6, 'q': 7, 'r': 8, 's': 9}

    return (decode_first_pos[position[0]], decode_second_pos[position[1]])


def _plot_matrix(grid: xr.DataArray,
                monkey: str,
                ax=None, fig=None, title=None, label=None, vmin=None, vmax=None, print_values=False, cmap=plt.cm.get_cmap('Greys')):
    """
    Helper function for plot_on_cortical_grid().

    A plotting method to plot a heatmap grid on the cortical grid of Clement. 
    It loads an image of the sulci of the monkey, and plots the matrix on top of it.

    Parameters
    ----------

    grid : xr.DataArray
        A 19x19 grid with coordinates of the cortical grid of Clement, created by the function init_location_grid.

    monkey : str
        The monkey name, to load the sulci image.

    etc...
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(70, 70))
        show = False

    # ploting helpers
    vmin = np.nanmin(grid) if vmin is None else vmin
    vmax = np.nanmax(grid) if vmax is None else vmax
    colors = cmap(np.linspace(0, .5, 100))  # Use the range from mid-red to deep red
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_greys', colors)

    # plot data
    im = ax.imshow(grid.T,cmap=custom_cmap, aspect='auto', origin='lower', 
                   extent=[grid.x.min()-.5, grid.x.max()+.5, grid.y.min()-.5, grid.y.max()+.5], 
                   vmin=vmin, vmax=vmax, alpha=1, zorder=0)
    cbar = plt.colorbar(im, ax=ax, fraction=.03, pad=0.04, label=label)

    if print_values:
        # plot values on the grid
        for x in grid.coords['x']:
            for y in grid.coords['y']:
                value = grid.sel(x=x, y=y).values
                if not np.isnan(value):
                    ax.text(x-.2, y+.2, str(int(value)), ha="center", va="center", color="black", fontsize=5, zorder=5, alpha=.5)
                
  
    # plot grey square where its no recording
    #ax.imshow(non_significants, cmap='Greys', alpha=.5, aspect='auto', origin='upper', extent=[-9, 9, -9, 9], vmin=-.1, vmax=.1)

    # plot sulci
    floc = os.path.join(cfg.PROJECT_PATH_LOCAL, 'data', 'recording_sites', f'sulci_{monkey}.png')  # load sulcus png
    img = plt.imread(floc)
    ax.imshow(img, origin='lower', extent=[-10, 10, -10, 10], zorder=10)  # plot sulci, bring to front

    # put text on top of the grid (LPFC MCC)
    pos_1 = [2, 8]
    pos_2 = [-4, -8.5]
    ax.text(pos_1[0], pos_1[1], 'MCC', ha="center", va="center", color="black", fontsize=10)
    ax.text(pos_2[0], pos_2[1], 'LPFC', ha="center", va="center", color="black", fontsize=10)

    # text on 'lateral, medial, anterior, posterior'
    ax.set_xlabel('Pos. rel. cage (mm)')
    ax.set_ylabel('Pos. rel. cage (mm)')
    ax.text(-8, -11.5, 'lat.', ha="center", va="center", color="black")
    ax.text(8, -11.5, 'med.', ha="center", va="center", color="black")
    ax.text(-12, 8, 'post.', ha="center", va="center", color="black", rotation=90)
    ax.text(-12, -8, 'ant.', ha="center", va="center", color="black", rotation=90)

    # show minor ticks at every 1, major at every 2
    ax.set_xticks(np.arange(-8, 9, 1), minor=True)
    ax.set_yticks(np.arange(-8, 9, 1), minor=True)
    ax.set_xticks(np.arange(-8, 9, 2), minor=False)
    ax.set_yticks(np.arange(-8, 9, 2), minor=False)
    #ax.grid(True, which='both', axis='both', linestyle='-', linewidth=1, color='grey', alpha=0.2)

    ax.set(xlim=(-9.5, 9.5), ylim=(-9.5, 9.5), aspect='equal')    
    sns.despine()

    # set title
    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_on_cortical_grid(
        df: pd.DataFrame,
        column_to_show=None,
        vmin=None, 
        vmax=None, 
        title=None, 
        bar_title=None,
        ax=None,
        print_values=False
        ):
    """
    The full routine to plot the data on the cortical grid of Clement. It will plot the data for each monkey, and each area, on the grid.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data. The dataframe should contain the columns 'monkey', 'session', 'area', and the column to show.

    column_to_show : str
        The column to show on the grid. This is the column that will be plotted on the grid.

    vmin : float
        The minimum value of the colorbar. If None, it will be the minimum value of the data.

    vmax : float
        The maximum value of the colorbar. If None, it will be the maximum value of the data.

    title : str
        The title of the plot.

    bar_title : str
        The title of the colorbar.
    """

    # get the data for the current monkey
    if ax is None:
        cm_to_in = 1/2.54
        fig, ax = plt.subplots(1, 1, figsize=(8*cm_to_in, 12*cm_to_in))

    # init grid 
    grid = _init_location_grid()
    
    # get the data for the current monkey, set it to the grid value
    locs_already_used = set()
    for (monkey, session, area), sub_df in df.groupby(['monkey', 'session', 'area']):
        # get the location of the current session, area for this monkey
        grid_loc_temp = _get_grid_location(monkey, session, area)
        if np.isnan(grid_loc_temp).any():
            continue

        # add to the grid (if the location is not already used, esle use the better value)
        value_to_show = sub_df[column_to_show].values[0]
        if grid_loc_temp not in locs_already_used:
            grid.loc[dict(x=grid_loc_temp[0], y=grid_loc_temp[1])] = value_to_show
            locs_already_used.add(grid_loc_temp)
        else:  # use the higher value
            grid.loc[dict(x=grid_loc_temp[0], y=grid_loc_temp[1])] = np.sum((value_to_show, grid.loc[dict(x=grid_loc_temp[0], y=grid_loc_temp[1])]))

    # plot the grid on the subplot
    _plot_matrix(grid, monkey, ax=ax, title=f"Monkey {monkey.upper()}", label=bar_title, vmin=vmin, vmax=vmax, print_values=print_values)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    if ax is None:
        return fig, ax
    else:
        return ax


def _plot_matrix_value_fb_custom(grid: xr.DataArray,
                monkey: str,
                not_signif_value: int,
                ax=None, fig=None, title=None, save=False, show=True, label=None, vmin=None, vmax=None, cmap=None):
    """
    Helper function for plot_on_cortical_grid().

    A plotting method to plot a heatmap grid on the cortical grid of Clement. 
    It loads an image of the sulci of the monkey, and plots the matrix on top of it.

    Parameters
    ----------

    grid : xr.DataArray
        A 19x19 grid with coordinates of the cortical grid of Clement, created by the function init_location_grid.

    monkey : str
        The monkey name, to load the sulci image.

    etc...
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(70, 70))
        show = False
    
    # load sulcus png
    floc = os.path.join(cfg.PROJECT_PATH_LOCAL, 'data', 'recording_sites', f'sulci_{monkey}.png')
    img = plt.imread(floc)

    # Create a custom colormap (reds, but lower values are not)
    ''' if cmap is None:
        reds = plt.cm.get_cmap('Reds')
        colors = reds(np.linspace(0.2, 1, 100))  # Use the range from mid-red to deep red
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_reds', colors)
    else:
        custom_cmap = cmap'''

    # plot sulcus, bring to front
    ax.imshow(img, origin='lower', extent=[-10, 10, -10, 10], zorder=10)

    # mask the non-significant values
    #matrix = grid.data.T  # transpose to match the grid, we want to x coordinate to be the horizontal axis
    #matrix_masked = np.where(matrix == not_signif_value, np.nan, matrix)  # set non-significant values to nan

    # plot heatmap of n_units
    matrix_n_units = grid.n_units.T
    vmin = np.nanmin(matrix_n_units) if vmin is None else vmin
    vmax = np.nanmax(matrix_n_units) if vmax is None else vmax
    # cmap is greys, but only use half of it
    greys = plt.cm.get_cmap('Greys')
    colors = greys(np.linspace(0, .5, 100))  # Use the range from mid-red to deep red
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_greys', colors)
    im = ax.imshow(matrix_n_units, cmap=custom_cmap, aspect='auto', origin='lower', extent=[grid.x.min()-.5, grid.x.max()+.5, grid.y.min()-.5, grid.y.max()+.5], vmin=vmin, vmax=vmax, alpha=1, zorder=0)
    cbar = plt.colorbar(im, ax=ax, fraction=.03, pad=0.04, label=label)

    # plot balls for feedback
    color = 'tab:green'
    matrix_fb = grid.n_signif_feedback
    max_value = matrix_fb.max().values
    multiplier = 7.5/max_value
    for x in matrix_fb.coords['x']:
        for y in matrix_fb.coords['y']:
            n_signif = matrix_fb.sel(x=x, y=y).values
            ax.plot(x-.2, y, 'o', color=color, markersize=n_signif*multiplier, alpha=.7, markeredgewidth=.5, zorder=6)

    # create legend (plot the balls for feedback and value, for max_value, half and 1/4)
    ax.plot(-100, -100, 'o', color=color, markersize=max_value*multiplier, alpha=.7, label=f'Feedback, {round(max_value*100)}%', markeredgewidth=.5)
    ax.plot(-100, -100, 'o', color=color, markersize=max_value*multiplier/2, alpha=.7, label=f'Feedback, {round(max_value*50)}%', markeredgewidth=.5)
    ax.plot(-100, -100, 'o', color=color, markersize=max_value*multiplier/4, alpha=.7, label=f'Feedback, {round(max_value*25)}%', markeredgewidth=.5)


    color = 'tab:orange'
    matrix_fb = grid.n_signif_value
    max_value = matrix_fb.max().values
    multiplier = 8/max_value
    for x in matrix_fb.coords['x']:
        for y in matrix_fb.coords['y']:
            n_signif = matrix_fb.sel(x=x, y=y).values
            ax.plot(x+.2, y, 'o', color=color, markersize=n_signif*multiplier, alpha=.5, markeredgewidth=.5, zorder=6)

    
    # create legend (plot the balls for feedback and value, for max_value, half and 1/4)
    ax.plot(-100, -100, 'o', color=color, markersize=max_value*multiplier, alpha=.7, label=f'Value, {round(max_value*100)}%', markeredgewidth=.5)
    ax.plot(-100, -100, 'o', color=color, markersize=max_value*multiplier/2, alpha=.7, label=f'Value, {round(max_value*50)}%', markeredgewidth=.5)
    ax.plot(-100, -100, 'o', color=color, markersize=max_value*multiplier/4, alpha=.7, label=f'Value, {round(max_value*25)}%', markeredgewidth=.5)

  
    # change the values of 'not_signif_value' to 1, the nans are nans, and the rest is zero
    #nan_mask = np.isnan(matrix)
    #matrix_non_signifs = matrix.copy()

    # Create a mask for values equal to not_signif_value
    #not_signif_mask = (matrix == not_signif_value)

    # Set the values: not_signif_value to 1, NaNs stay NaNs, and the rest to 0
    #matrix_non_signifs[not_signif_mask] = 1
    #matrix_non_signifs[~nan_mask & ~not_signif_mask] = 0    
    
    #ax.imshow(matrix_non_signifs, cmap='Greys', alpha=.15, aspect='auto', origin='lower', extent=[grid.x.min()-.5, grid.x.max()+.5, grid.y.min()-.5, grid.y.max()+.5], vmin=0, vmax=1)
    
    

    # plot grey square where its not significant
    #ax.imshow(non_significants, cmap='Greys', alpha=.5, aspect='auto', origin='upper', extent=[-9, 9, -9, 9], vmin=-.1, vmax=.1)

    # put text on top of the grid (LPFC MCC)
    pos_1 = [2, 8]
    pos_2 = [-4, -8.5]
    ax.text(pos_1[0], pos_1[1], 'MCC', ha="center", va="center", color="black", fontsize=10)
    ax.text(pos_2[0], pos_2[1], 'LPFC', ha="center", va="center", color="black", fontsize=10)

    # text on 'lateral, medial, anterior, posterior'
    ax.set_xlabel('Pos. rel. cage (mm)')
    ax.set_ylabel('Pos. rel. cage (mm)')
    ax.text(-8, -11.5, 'lat.', ha="center", va="center", color="black")
    ax.text(8, -11.5, 'med.', ha="center", va="center", color="black")
    ax.text(-12, 8, 'post.', ha="center", va="center", color="black", rotation=90)
    ax.text(-12, -8, 'ant.', ha="center", va="center", color="black", rotation=90)

    # show minor ticks at every 1, major at every 2
    ax.set_xticks(np.arange(-8, 9, 1), minor=True)
    ax.set_yticks(np.arange(-8, 9, 1), minor=True)
    ax.set_xticks(np.arange(-8, 9, 2), minor=False)
    ax.set_yticks(np.arange(-8, 9, 2), minor=False)
    #ax.grid(True, which='both', axis='both', linestyle='-', linewidth=1, color='grey', alpha=0.2)

    # legend below plot
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -.5), frameon=True, ncol=2)

    # no spines
    '''ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(False)
    ax.spines['bottom'].set_linewidth(False)'''

    ax.set(xlim=(-9.5, 9.5), ylim=(-9.5, 9.5), aspect='equal')    

    # set title
    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_on_cortical_grid_value_fb_custom(
        df: pd.DataFrame,
        column_to_show=None,
        vmin=None, 
        vmax=None, 
        title=None, 
        bar_title=None
        ):
    """
    CUSTOM version of the plot_on_cortical_grid function to plot the number of units coding for feedback and value.

    The full routine to plot the data on the cortical grid of Clement. It will plot the data for each monkey, and each area, on the grid.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data. The dataframe should contain the columns 'monkey', 'session', 'area', and the column to show.

    column_to_show : str
        The column to show on the grid. This is the column that will be plotted on the grid.

    vmin : float
        The minimum value of the colorbar. If None, it will be the minimum value of the data.

    vmax : float
        The maximum value of the colorbar. If None, it will be the maximum value of the data.

    title : str
        The title of the plot.

    bar_title : str
        The title of the colorbar.
    """
    cm_to_in = 1/2.54
    fig, axs = plt.subplots(1, 2, figsize=(19*cm_to_in, 12*cm_to_in))

    # get the data for the current monkey
    for i, monkey in enumerate(["ka", "po"]):
        # init grid and subplot
        ax = axs[i]
        grid = _init_location_grid()
        # convert 'grid' DataArray to xarray Dataset with 3 DataArrays: n_units, feedback and value, each with the same coordinates
        grid_ds = xr.Dataset(
            {'n_signif_value': grid.copy(data=np.zeros(grid.data.shape)),
             'n_signif_feedback': grid.copy(data=np.zeros(grid.data.shape)),
             'n_units': grid.copy(data=np.zeros(grid.data.shape))},
            coords=grid.coords)
        
        # get the data for the current monkey, set it to the grid value
        locs_already_used = set()
        for (session, area), sub_df in df.loc[df.monkey==monkey].groupby(['session', 'area']):
                # get the location of the current session, area for this monkey
                grid_loc = _get_grid_location(monkey, session, area)
                if np.isnan(grid_loc).any():
                    continue

                value = sub_df["n_signif_value"].values[0]
                feedback = sub_df["n_signif_feedback"].values[0]
                n_units = sub_df["n_units"].values[0]
                
                # add to the grid (if the location is not already used, esle use the better value)
                if grid_loc not in locs_already_used:
                    grid_ds.n_signif_value.loc[dict(x=grid_loc[0], y=grid_loc[1])] = value
                    grid_ds.n_signif_feedback.loc[dict(x=grid_loc[0], y=grid_loc[1])] = feedback
                    grid_ds.n_units.loc[dict(x=grid_loc[0], y=grid_loc[1])] = n_units
                    locs_already_used.add(grid_loc)
                else:  # use the best value
                    if n_units > grid_ds.n_units.loc[dict(x=grid_loc[0], y=grid_loc[1])]:
                        grid_ds.n_signif_value.loc[dict(x=grid_loc[0], y=grid_loc[1])] = value
                        grid_ds.n_signif_feedback.loc[dict(x=grid_loc[0], y=grid_loc[1])] = feedback
                        grid_ds.n_units.loc[dict(x=grid_loc[0], y=grid_loc[1])] = n_units

        # plot the grid on the subplot
        _plot_matrix_value_fb_custom(grid_ds, monkey, ax=ax, title=f"Monkey {monkey.upper()}", label=bar_title, not_signif_value=-100)

        if title is not None:
            ax.set_title(title)

    plt.tight_layout()

    return fig, axs

