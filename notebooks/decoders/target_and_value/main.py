import concurrent.futures

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from helpers import *

import warnings
warnings.filterwarnings('ignore')

savedir = '/Users/zsombi/Library/CloudStorage/OneDrive-Personal/PoPy/notebooks/decoders/target_and_value/results'

if __name__ == '__main__':
        
    # Load metadata
    metadata = load_metadata()
    metadata = metadata.loc[metadata['block_len_valid']]

    # initialize results container
    results = {
        'target_1': None,
        'target_2': None,
        'target_3': None,
    }

    # parallel with asynchronous execution on 10 CPUs
    # get number of cores in the machine
    n_cores = 1#os.cpu_count() -1
    print(f'Number of cores used: {n_cores}')
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        future_proxy_mapping = {}
        # submit jobs
        # loop throughs sessions
        for i_session, session_metadata in enumerate(metadata.iterrows()):
            monkey, session = session_metadata[1].monkey, session_metadata[1].session
            if monkey == 'po':
                continue
            if not session == '210322':
                continue
            future = executor.submit(process_session, monkey, session)
            futures.append(future)
            future_proxy_mapping[future] = (monkey, session)

        # wait for results, save them
        count = 0
        log = []
        for future in concurrent.futures.as_completed(futures):
            # save results
            monkey_fut, session_fut = future_proxy_mapping[future]
            try:
                xr_t1, xr_t2, xr_t3 = future.result()
            except Exception as e:
                print(f'Error in {monkey_fut} - {session_fut}')
                log.append(f'{monkey_fut} - {session_fut} - {e}')
                continue
        
            ## save data
            if results['target_1'] is None:
                results['target_1'] = xr_t1
                results['target_2'] = xr_t2
                results['target_3'] = xr_t3
            else:
                results['target_1'] = xr.concat([results['target_1'], xr_t1], dim='unit_id')
                results['target_2'] = xr.concat([results['target_2'], xr_t2], dim='unit_id')
                results['target_3'] = xr.concat([results['target_3'], xr_t3], dim='unit_id')
            
            count += 1
            print(f'Progress: {count}/{len(futures)}')

    # save results
    for target in results.keys():
        results[target].to_netcdf(f'{savedir}/{target}_run02.nc')
        print(f'{target} saved!')

    # print log
    for l in log:
        print('*** ERRORS ***')
        print(l)