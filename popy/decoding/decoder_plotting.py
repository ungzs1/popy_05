import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import xarray as xr

from popy.plotting.plotting_tools import plot_keypoints


### Helper functions

def _plot_time_series(data, ax=None, color='black', linestyle='-', alpha=1, linewidth=None, label=None):
    """
    Plots a single time series data (can be firing rate, decoder performance, etc.).

    Parameters
    ----------
    data : xarray.Dataset
        The data to be plotted. Dimensions should be 1 by time.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The axis to plot the data. If None, a new figure will be created.
    color : str, optional
        The color of the plotted line. Default is 'black'.
    linestyle : str, optional
        The style of the plotted line. Default is '-'.
    alpha : float, optional
        The transparency of the plotted line. Default is 1.
    linewidth : float, optional
        The width of the plotted line. Default is None.
    label : str, optional
        The label for the plotted line. Default is None.

    Raises
    ------
    ValueError
        If the data is not an xarray.DataArray, or if it has more than one element in the first dimension,
        or if it does not have 'time' as the second dimension.

    Returns
    -------
    None
        This function does not return anything. It plots the data on the specified axis or a new figure.

    """
    if not isinstance(data, xr.DataArray):
        raise ValueError(f"The data should be an xarray.DataArray, not {type(data)}.")
    if len(data.data.shape) == 2 & data.data.shape[0] != 1:
        raise ValueError("The data should be either a 1D or a 2D array with only one element in the first dimension.")
    if data.dims[-1] != 'time':
        raise ValueError("The data should have 'time' as the last dimension.")

    data_vector = data.values.squeeze()
    time_vector = data.time

    # plot scores
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(time_vector, data_vector, color=color, linestyle=linestyle, label=label, alpha=alpha, linewidth=linewidth)


def _plot_multiple_time_series(data, ax=None, cmap='tab10', colors=None, vmin=None, vmax=None, cbar_label=None, ymin=None, ymax=None):
    """
    Plots multiple time series data (can be firing rate, decoder performance, etc.).

    Parameters
    ----------
    data : xarray.Dataset
        The data to be plotted. Dimensions should be n by time.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The axis to plot the data. If None, a new figure will be created.
    cmap : str, optional
        The colormap to be used.
    vmin : float, optional
        The minimum value of the colorbar.
    vmax : float, optional
        The maximum value of the colorbar.
    cbar_label : str, optional      
        The label of the colorbar.

    Raises
    ------
    ValueError
        If the data is not an xarray.DataArray or if the second dimension is not 'time'.

    Returns
    -------
    None
        This function does not return anything. It plots the data on the specified axis or creates a new figure if no axis is provided.

    """
    if not isinstance(data, xr.DataArray):
        raise ValueError("The data should be an xarray.DataArray.")
    if data.dims[1] != 'time':
        raise ValueError("The data should have 'time' as the second dimension.")

    data_vector = data.values
    time_vector = data.time.values

    # plot scores
    if ax is None:
        fig, ax = plt.subplots()
    
    # plotting
    if ymin is not None and ymax is not None:
        ymin_abs, ymax_abs = np.min([ymin, ymax]), np.max([ymin, ymax])
    else :
        ymin_abs, ymax_abs = 0, len(data.session)
    extent = [time_vector[0], time_vector[-1], ymin_abs, ymax_abs]
    cbar = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap, extent=extent,
                      vmin=vmin, vmax=vmax, interpolation='none')
    # nan values to white
    cbar.cmap.set_bad(color='white')
    plt.colorbar(cbar, ax=ax, label=cbar_label)


### Main functions


def show_single_session_decoder(results):
    """
    Plot the results of a decoder in time.
    
    Parameters
    ----------
    results : xarray.Dataset
        The results of the decoder.

    Returns
    -------
    None
    """

    time_vector = results.time
    n_trials = np.round(time_vector.max()/7.5)
    monkey = results.monkey.values[0]
    session = results.session.values[0]
    condition = results.attrs["glm_conditions"]

    cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(10*cm*n_trials, 10*cm))
    # default font size
    plt.rcParams.update({'font.size': 12})

    colors = {'MCC': 'black', 'LPFC': 'tab:blue'}

    areas = np.unique(results.area)

    for area in areas:
        # PLOT DATA
        data_curr = results.where(results['area'] == area, drop=True)

        _plot_time_series(data_curr.scores, ax=ax, color=colors[area], label=area, linewidth=1)

        # PLOT SIGNIFICANCE
        # if not exists signif column, pass
        if 'signif' not in results.data_vars:
            continue
        else:
            pass
            ax_=ax.twinx()
            ax_.set_ylim([0, 1.15])
            # hide ticks and labels and spines
            ax_.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            # x units
            ax_.set_yticks([])
            ax_.spines['right'].set_visible(False)
            ax_.spines['top'].set_visible(False)

            significance = data_curr.signif.squeeze()
            for i, sig in enumerate(significance[:-1]):
                if sig :
                    # plot bar over the plot for significant scores
                    ax_.plot([time_vector[i], time_vector[i+1]],
                            [1, 1] if area == 'MCC' else [1.1, 1.1],
                            color=colors[area],
                            linewidth=2)

    # plot keypoints
    plot_keypoints(ax)
    ax.grid(axis='x', linestyle='--', alpha=.5)

    # put legend to the right next to plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))


    # first row's values of 'monkey', 'session', 'target'
    title = f"{monkey} - {session} - {condition}"
    ax.set_title(title)

    # remve top and right spines
    sns.despine()

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('decodability ("%" or R2)', fontsize=12)

    plt.show()


def show_population_decoder_results(data, title=None, vmin=None, vmax=None, show_session_names=False, time_of_interest=None):
    """
    Display population decoder results.

    Parameters:
    data: xarray.Dataset
        The dataset containing the decoder results.

    Returns:
    None
    """

    fig, axs = plt.subplots(2, 1, figsize=(5, 10))
    if title is None:
        title = 'Population Decoder Results'
    plt.suptitle(title)

    time_vector = data.time.values

    # min and max non-na elements
    unique = np.unique(data.values)
    unique = unique[~np.isnan(unique)]
    unique = unique[unique >= 0]
    if vmin is None and vmax is None:
        vmax, vmin = np.max(unique), np.min(unique)
        vmax, vmin = vmax, vmin-(vmax-vmin)*0.1

    # count number of LPFC and MCC sessions
    n_LPFC = np.count_nonzero(data.area == 'LPFC')
    n_MCC = np.count_nonzero(data.area == 'MCC')

    for m, monkey in enumerate(['ka', 'po']):
        for a, area in enumerate(['LPFC', 'MCC']):
            ax = axs[m]
            
            ax.set_facecolor('lightgrey')  # set background color to grey

            # get data
            data_curr = data[data.area == area]
            data_curr = data_curr[data_curr.monkey == monkey]
            
            # sort based on mean first significant time
            #if not show_session_names:
            first_signifs = []
            for i, scores_temp in enumerate(data_curr):
                scores_temp = scores_temp[time_of_interest[0]:time_of_interest[1]] if time_of_interest is not None else scores_temp
                if np.isnan(scores_temp).all():
                    first_signifs.append(len(time_vector)+1)
                else:
                    first_signifs.append(np.where(~np.isnan(scores_temp))[0][0])

            sort_idx = np.argsort(first_signifs)
            data_curr = data_curr[sort_idx]
            #data_curr = data_curr.sortby(data_curr[:, 35:40].mean(axis=1), ascending=False)

            # imshow with no smoothing
            ymin = 0 if area == 'LPFC' else n_LPFC  # 0 for LPFC, number of LPFC sessions for MCC (to plot on top of LPFC)
            ymax = n_LPFC if area == 'LPFC' else n_LPFC+1 + n_MCC  # number of LPFC sessions for LPFC, total number of sessions for MCC
            _plot_multiple_time_series(data_curr, ax=ax, cmap='Blues' if area == 'LPFC' else 'Greys', ymin=ymin, ymax=ymax,
                                        cbar_label='R2 score', vmin=vmin, vmax=vmax)

            # plot keypoints
            plot_keypoints(ax, fontsize=6)

            # plot number of significant sessions
            ax.plot(time_vector, (data_curr==data_curr).sum(axis=0), color='tab:red', lw=1, label=f'# signif sessions', alpha=0.5, linestyle='--')
            #ax.legend(loc='upper left')
            title = f'{monkey} {area}'
            if data.attrs:
                title += f' - {data.attrs["glm_conditions"]}'
            ax.set_title(title)

            ax.grid(axis='x', linestyle='--', alpha=0.5)
            ax.set_ylabel('session')

            # yticks
            ax.set_yticks([])
            '''if show_session_names:
                yticks = np.arange(ymin, ymax, 1)
                ax.set_yticks(yticks+.5)
                ax.set_yticklabels([f'{area}_{session}' for session in data_curr.session.values])
                ax.grid(axis='y')'''
            
        # plot vertical line at n_LPFC
        ax.axhline(n_LPFC, color='black', linestyle='--', alpha=0.5)    

    plt.tight_layout()

    return fig, axs


def show_prop_signif_units(results, title=None):
    if title is None:
        title = 'Proportion of Significant Units'
    # rest of the code
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.suptitle(title)

    colors = {'LPFC': 'tab:blue', 'MCC': 'grey'}
    styles = {'ka': '-', 'po': '--'}
    handles, labels = [], []

    time_vector = results.time.values
    for m, monkey in enumerate(['ka', 'po']):
        for a, area in enumerate(['LPFC', 'MCC']):
            results_curr = results[results.area == area]
            results_curr = results_curr[results_curr.monkey == monkey]
            results_curr = (results_curr==results_curr).mean(axis=0)  # proportion of significant sessions (mean of non-nan values?)

            """ax.plot(time_vector, (results_curr==results_curr).mean(axis=0), color=colors[area], lw=1, label=f'{monkey} {area}', linestyle=styles[monkey])"""
            _plot_time_series(results_curr, ax=ax, color=colors[area], linestyle=styles[monkey], linewidth=1, label=f'{monkey} {area}')

    # legend below plot
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

    ax.set_title(f'condition: {results.attrs["glm_conditions"]}')
    ax.grid(axis='x', alpha=0.3)
    ax.set_ylabel('% significant sessions')
    # hide spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # order of legend
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    handles, labels = zip(*unique)
    # switch index 1 and 2
    order = np.arange(len(handles))
    order[1], order[2] = order[2], order[1]
    fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center', ncols=2, bbox_to_anchor=(0.5, 0.05))
    plot_keypoints(ax, fontsize=8)
    
    plt.tight_layout()

    return fig, ax