import pandas as pd
import numpy as np
import datetime
import os
import concurrent.futures
import traceback
import xarray as xr
import logging

from neural_value_helpers import *

from popy.io_tools import load_metadata
from popy.decoding.population_decoders import run_decoder
import popy.config as cfg
import matplotlib.pyplot as plt
import io
import xml.etree.ElementTree as ET
import math

### Load metadata

def get_all_sessions():
    session_metadata = load_metadata()
    session_metadata = session_metadata[session_metadata['block_len_valid'] == True]  # Only use sessions with valid block length

    monkeys = session_metadata['monkey'].values 
    sessions = session_metadata['session'].values

    return monkeys, sessions


### Configure logging

def end_log():
    # start time is the first log entry
    end_time = datetime.datetime.now()
    logging.info(f"Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

def init_io(PARAMS):
    os.makedirs(PARAMS['floc'], exist_ok=True)

    # configure logging
    logging.basicConfig(filename=os.path.join(PARAMS['floc'], 'log.txt'),
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filemode='w')  # 'w' mode will overwrite the log file

    start_time = datetime.datetime.now()
    logging.info("PARAMS:")
    for key, value in PARAMS.items():
        logging.info(f'{key}: {value}')
    logging.info(f"Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

### Save results

def save_results(dfs_all, floc, fname):
    dfs_all.to_pickle(os.path.join(floc, fname))




def fig_to_svg_string(fig):
    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    plt.close(fig)
    return buf.getvalue()

def extract_svg_body(svg_str):
    tree = ET.ElementTree(ET.fromstring(svg_str))
    root = tree.getroot()

    width = float(root.attrib.get('width').replace('pt', ''))
    height = float(root.attrib.get('height').replace('pt', ''))

    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]

    body = list(root)
    return width, height, body

def merge_svgs_custom_layout(figs):
    if len(figs) < 1:
        raise ValueError("At least one figure is required")

    # Process first figure (row 1)
    width1, height1, body1 = extract_svg_body(fig_to_svg_string(figs[0]))
    merged_elements = []

    # Wrap first fig in a group
    g1 = ET.Element("g", attrib={"transform": f"translate(0,0)"})
    g1.extend(body1)
    merged_elements.append(g1)

    # Remaining figures
    remaining_figs = figs[1:]
    n = len(remaining_figs)
    half = math.ceil(n / 2)
    left_figs = remaining_figs[:half]
    right_figs = remaining_figs[half:]

    # Determine x offset for left and right columns
    max_left_width = 0
    total_left_height = 0
    left_groups = []
    y_offset = height1  # start from row 2

    for fig in left_figs:
        w, h, body = extract_svg_body(fig_to_svg_string(fig))
        g = ET.Element("g", attrib={"transform": f"translate(0,{y_offset})"})
        g.extend(body)
        left_groups.append(g)
        y_offset += h
        max_left_width = max(max_left_width, w)
        total_left_height += h

    max_right_width = 0
    total_right_height = 0
    right_groups = []
    y_offset = height1

    for fig in right_figs:
        w, h, body = extract_svg_body(fig_to_svg_string(fig))
        g = ET.Element("g", attrib={"transform": f"translate({max_left_width},{y_offset})"})
        g.extend(body)
        right_groups.append(g)
        y_offset += h
        max_right_width = max(max_right_width, w)
        total_right_height += h

    # Merge all groups
    merged_elements.extend(left_groups)
    merged_elements.extend(right_groups)

    # Calculate total dimensions
    total_width = max_left_width + max_right_width
    total_height = height1 + max(total_left_height, total_right_height)

    # Build final SVG root
    svg_root = ET.Element(
        "svg",
        attrib={
            "xmlns": "http://www.w3.org/2000/svg",
            "width": f"{total_width}pt",
            "height": f"{total_height}pt",
            "viewBox": f"0 0 {total_width} {total_height}"
        }
    )
    svg_root.extend(merged_elements)
    return ET.ElementTree(svg_root)

### Set parameters

def convert_trajectory_format(data_projected):
    """ Convert the data_projected xarray to a format where each fb_sequence is averaged across trials."""
    data_projected_mean = []
    for fb_seq in np.unique(data_projected.fb_sequence.values):
        data_projected_temp = data_projected.sel(trial_id=data_projected.fb_sequence==fb_seq)
        if len(data_projected_temp) > 0:
            data_projected_temp = data_projected_temp.mean(dim='trial_id')
            data_projected_temp = data_projected_temp.assign_coords(fb_sequence=fb_seq)
            data_projected_mean.append(data_projected_temp)

    data_projected_mean = xr.concat(data_projected_mean, dim='fb_sequence')

    return data_projected_mean

### Run
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
def run_all_session_plotting(monkey, session, PARAMS):
    print('running: ', monkey, session)
    #monkey, session, subregion = 'ka', '210322', 'vLPFC'

    neural_dataset = load_data_custom(monkey, session, n_extra_trials=(-1, 0), sr=10)

    for subregion in np.unique(neural_dataset.subregion.data):
        all_figs = []
        neural_dataset_temp = neural_dataset.sel(unit=neural_dataset.subregion==subregion)
        #project neural data to the neural value space, get across-time decodability matrix
        decodability_matrix, data_projected = time_resolved_decoder_all_time(neural_dataset_temp, target='R_1', across_time=True)

        # Plot the results
        fig, axes = plot_across_time_decodability(projected_data=data_projected, decodability_matrix=decodability_matrix, vmin=0.1, vmax=.9)
        plt.suptitle('{} {} {}'.format(monkey, session, subregion), fontsize=10)
        all_figs.append(fig)

        # make plots
        for t_interest_value in ([-4, -2], [1.5, 3.5]):
            # get neural value for the current trial and the previous one
            data_projected = add_neural_value_coord(data_projected, t_interest=t_interest_value)

            fig, ax = plot_projected_data(data_projected, t_interest_value, n_extra_trials=(-1, 0), xlim=[-5, 5], paper_format=True)
            plt.suptitle(f'{monkey} {session} {subregion}, t_eval = {t_interest_value}s', fontsize=10)
            all_figs.append(fig)

            behav_new = data_projected.mean('time').coords.to_dataset().to_dataframe().reset_index()
            behav_new['monkey'] = monkey
            behav_new['session'] = session
            behav_new['subregion'] = subregion
            # put monkey, session, subregion in the front
            behav_new = behav_new[['monkey', 'session', 'subregion'] +
                                [c for c in behav_new.columns if c not in ['monkey', 'session', 'subregion']]]
            behav_new = behav_new.dropna()

            fig, ax = plot_Vt_per_sequence(behav_new, paper_format=True, ylim=None, show_datapoints=True, showfliers=False)
            ax.set_title(f'{monkey} {session} {subregion} - t_eval = {t_interest_value}s', fontsize=10)
            all_figs.append(fig)

            fig, ax = create_r_style_plot(None, behav_new, 'feedback', 'dV', paper_format=True)
            ax.set_title(f'{monkey} {session} {subregion} - t_eval = {t_interest_value}s', fontsize=10)
            all_figs.append(fig)

            fig, ax = green_red_plot(behav_new, paper_format=True)
            ax.set_title(f'{monkey} {session} {subregion} - t_eval = {t_interest_value}s', fontsize=10)
            all_figs.append(fig)

            n_perms = 1000
            cpds_all = process_single_session(behav_new, n_perms=n_perms)
            cpds_all = pd.DataFrame([cpds_all])

            fig, axs = plot_cpds_history_neural_value(cpds_all)
            axs.set_title(f'{monkey} {session} {subregion} - CPDs', fontsize=10)
            all_figs.append(fig)

        #save_figures_split_layout(all_figs, filename="combined_figures_row.png", figsize=(20, 5), title=f'{monkey} {session} {subregion}')
        floc = os.path.join(PATH, 'notebooks', 'population_decoding', 'figs', 'neural_value', 'single_sessions', f'{monkey}_{session}_{subregion}_combined_figures_row.svg')
        merged_tree = merge_svgs_custom_layout(all_figs)
        merged_tree.write(floc)
        plt.close('all')

    session_log = []
    behav_new = pd.DataFrame([])
    return behav_new, session_log

def run_across_time_decoding(monkey, session, PARAMS):
    print('running: ', monkey, session)
    #monkey, session, subregion = 'ka', '210322', 'vLPFC'

    neural_dataset = load_data_custom(monkey, session, n_extra_trials=(-1, 0), sr=10)

    across_time_matrices = []
    trajectories = []

    for subregion in np.unique(neural_dataset.subregion.data):
        neural_dataset_temp = neural_dataset.sel(unit=neural_dataset.subregion==subregion)
        
        #project neural data to the neural value space, get across-time decodability matrix
        across_time_matrix, data_projected = time_resolved_decoder_all_time(neural_dataset_temp, target='R_1', across_time=True)

        # convert to trajectory format
        trajectories_temp = convert_trajectory_format(data_projected)

        # add new dimension for data, and then store them
        trajectories_temp = trajectories_temp.expand_dims(session=[f'{monkey}_{session}_{subregion}'])
        across_time_matrix = across_time_matrix.expand_dims(session=[f'{monkey}_{session}_{subregion}'])

        across_time_matrices.append(across_time_matrix)
        trajectories.append(trajectories_temp)

    return across_time_matrices, trajectories

def run_neural_value_extraction(monkey, session, PARAMS):
    print('running: ', monkey, session)
    #monkey, session, subregion = 'ka', '210322', 'vLPFC'

    neural_dataset = load_data_custom(monkey, session, n_extra_trials=(-1, 0), sr=10)

    dfs = []
    for subregion in np.unique(neural_dataset.subregion.data):
        neural_dataset_temp = neural_dataset.sel(unit=neural_dataset.subregion==subregion)
        
        #project neural data to the neural value space, get across-time decodability matrix
        _, data_projected = time_resolved_decoder_all_time(neural_dataset_temp, target='R_1')

        for t_interest_value in [(-4, -2), (1.5, 3.5)]:
            # get neural value for the current trial and the previous one
            data_projected = add_neural_value_coord(data_projected, t_interest=t_interest_value)
            data_projected = data_projected.rename({'V_t': f'V_t_{t_interest_value[0]}_{t_interest_value[1]}'})
            data_projected = data_projected.rename({'V_t_p1': f'V_t_p1_{t_interest_value[0]}_{t_interest_value[1]}'})
            data_projected = data_projected.rename({'dV': f'dV_{t_interest_value[0]}_{t_interest_value[1]}'})

        behav_new = data_projected.mean('time').coords.to_dataset().to_dataframe().reset_index()
        behav_new['monkey'] = monkey
        behav_new['session'] = session
        #behav_new['area'] = neural_dataset.area
        behav_new['subregion'] = subregion
        dfs.append(behav_new)
            
    behav_new = pd.concat(dfs, ignore_index=True)
    behav_new = behav_new[['monkey', 'session', 'subregion'] + [c for c in behav_new.columns if c not in ['monkey', 'session', 'subregion']]]
        
    session_log = []
    return behav_new, session_log


### RUN THIS FOR NEURAL VALUE EXTRACTION
if False:
    PARAMS = {
        'floc': os.path.join(cfg.PROJECT_PATH_LOCAL, 'notebooks', 'population_decoding', 'results', 'behav_neural_value'),
        'msg': 'extracting plots',
    }

    if __name__ == '__main__':
        init_io(PARAMS)  # Initialize logging and create results folder

        monkeys, sessions = get_all_sessions()  # Get a pandas df containing all sessions' meta information

        n_cores = np.min([100, os.cpu_count()-1])  # get number of cores in the machine
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
            # submit jobs
            futures, future_proxy_mapping = [], {}
            for monkey, session in zip(monkeys, sessions):
                future = executor.submit(run_neural_value_extraction, monkey, session, PARAMS)  # Run decoder for each session
                futures.append(future)
                future_proxy_mapping[future] = (monkey, session)

            # wait for results, save them
            count_good = 0
            count_bad = 0

            dfs_all = []  
            for future in concurrent.futures.as_completed(futures):
                try:
                    res, session_log = future.result()
                    monkey_fut, session_fut = future_proxy_mapping[future]

                    # Append results to existing results and save after each session
                    if len(dfs_all) == 0:
                        dfs_all = res
                    else:
                        dfs_all = pd.concat([dfs_all, res], ignore_index=True)
                
                    # Save results after each session
                    save_results(dfs_all, PARAMS['floc'], fname='behav.pkl')  # Save results after each session

                    # Log progress
                    for line in session_log:
                        logging.info(line)
                    logging.info(f"Finished for monkey {monkey_fut} and session {session_fut}")

                    count_good += 1

                except Exception as e:  # Catch exceptions and log them
                    logging.error(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}")
                    print(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}\n")
                    traceback.print_exc()  # Print traceback (?)

                    count_bad += 1

                print(f'Progress: {count_good + count_bad}/{len(monkeys)} failed: {count_bad}')


        end_log()
        print(f'Finished all on {datetime.datetime.now()}')



### RUN THIS FOR ACROSS TIME DECODING
if False:
    PARAMS = {
        'floc': os.path.join(cfg.PROJECT_PATH_LOCAL, 'notebooks', 'population_decoding', 'results', 'neural_values_across_time'),
        'msg': 'generating 2 datasets: one for the across-time decodability matrix, one for the trajectories (for all sessions)',
    }


    if __name__ == '__main__':
        init_io(PARAMS)  # Initialize logging and create results folder

        monkeys, sessions = get_all_sessions()  # Get a pandas df containing all sessions' meta information

        n_cores = np.min([111, os.cpu_count()-1])  # get number of cores in the machine
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
            # submit jobs
            futures, future_proxy_mapping = [], {}
            for monkey, session in zip(monkeys, sessions):
                future = executor.submit(run_across_time_decoding, monkey, session, PARAMS)  # Run decoder for each session
                futures.append(future)
                future_proxy_mapping[future] = (monkey, session)

            # wait for results, save them
            count_good = 0
            count_bad = 0

            matrices_all = []
            projections_all = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    across_time_matrices, projections = future.result()
                    monkey_fut, session_fut = future_proxy_mapping[future]

                    # Append results to existing results and save after each session
                    if len(across_time_matrices) == 0:
                        logging.warning(f"No across-time matrices found for monkey {monkey_fut} and session {session_fut}. Skipping.")
                    else:
                        if len(matrices_all) == 0:
                            matrices_all = across_time_matrices
                        else:
                            matrices_all += across_time_matrices

                    if len(projections) == 0:
                        logging.warning(f"No projections found for monkey {monkey_fut} and session {session_fut}. Skipping.")
                    else:
                        if len(projections_all) == 0:
                            projections_all = projections
                        else:
                            projections_all += projections

                    # Save results after each session (to avoid losing data in case of an error)
                    # concatenate matrices and projections
                    matrices_all_xr = xr.concat(matrices_all, dim='session')
                    projections_all_xr = xr.concat(projections_all, dim='session')

                    matrices_all_xr.to_netcdf(os.path.join(PARAMS['floc'], 'across_time_decodability_matrix.nc'))  # Save across-time decodability matrix
                    projections_all_xr.to_netcdf(os.path.join(PARAMS['floc'], 'trajectories.nc'))  # Save trajectories

                    # Log progress
                    logging.info(f"Finished for monkey {monkey_fut} and session {session_fut}")

                    count_good += 1

                except Exception as e:  # Catch exceptions and log them
                    logging.error(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}")
                    print(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}\n")
                    traceback.print_exc()  # Print traceback (?)

                    count_bad += 1

                print(f'Progress: {count_good + count_bad}/{len(monkeys)} failed: {count_bad}')

        end_log()
        print(f'Finished all on {datetime.datetime.now()}')


### RUN THIS FOR PLOTTING

if True:
    PARAMS = {
        'floc': os.path.join(cfg.PROJECT_PATH_LOCAL, 'notebooks', 'population_decoding', 'results', 'plots'),
        'msg': 'extracting plots',
    }

    if __name__ == '__main__':
        init_io(PARAMS)  # Initialize logging and create results folder

        monkeys, sessions = get_all_sessions()  # Get a pandas df containing all sessions' meta information

        n_cores = np.min([100, os.cpu_count()-1])  # get number of cores in the machine
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
            # submit jobs
            futures, future_proxy_mapping = [], {}
            for monkey, session in zip(monkeys, sessions):
                future = executor.submit(run_all_session_plotting, monkey, session, PARAMS)  # Run decoder for each session
                futures.append(future)
                future_proxy_mapping[future] = (monkey, session)

            # wait for results, save them
            count_good = 0
            count_bad = 0

            dfs_all = []  
            for future in concurrent.futures.as_completed(futures):
                try:
                    res, session_log = future.result()
                    monkey_fut, session_fut = future_proxy_mapping[future]

                    # Log progress
                    for line in session_log:
                        logging.info(line)
                    logging.info(f"Finished for monkey {monkey_fut} and session {session_fut}")

                    count_good += 1

                except Exception as e:  # Catch exceptions and log them
                    logging.error(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}")
                    print(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}\n")
                    traceback.print_exc()  # Print traceback (?)

                    count_bad += 1

                print(f'Progress: {count_good + count_bad}/{len(monkeys)} failed: {count_bad}')


        end_log()
        print(f'Finished all on {datetime.datetime.now()}')

