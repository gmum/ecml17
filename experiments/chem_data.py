import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import os
from scripts.run_links import run
from config import config
results_dir = config['results_dir']

if __name__ == "__main__":

    np.seterr(all='ignore')

    dataset = '5ht6_KRFP_pca5'
    keys = ['cov_type', 'link_weight', 'k']
    metrics = ['links_accuracy']
    results = pd.DataFrame({'dataset': []})

    for k in [5, 10, 15]:
            for lw in [1,2,4,8]:
                # n_folds - default 5
                params = {'seed': 42,
                          'n_init': 10,
                          'n_jobs': 10,
                          'k': k,
                          'gauss_per_clust': 1,
                          'cov_type': 'fixed_spherical',
                          'verbose': False,
                          'link_weight': lw}

                key_params = [p for p in params if p in keys]
                res = {key: [params[key]] for key in key_params}
                res['dataset'] = [dataset]
                res_filename = "_".join(["{}:{}".format(key, v[0]) for key, v in res.iteritems()]) + ".csv"
                res_path = os.path.join(results_dir, res_filename)

                if not os.path.exists(res_path):
                    ret = run(dataset, metrics, params)
                    res.update({m: [ret[m]] for m in metrics})
                    part_results = pd.DataFrame.from_dict(res)
                    part_results.to_csv(res_path)
                else:
                    part_results = pd.read_csv(res_path)

                results = results.append(part_results)


    save_path = os.path.join(results_dir, 'chem_data.csv')
    results.to_csv(save_path, index=False)
