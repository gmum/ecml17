import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import os
from scripts.run_labels import run
from config import config

results_dir = config['results_dir']


if __name__ == "__main__":

    np.seterr(all='ignore')

    datasets = ['vertebral', 'ionosphere', 'ecoli', 'wisconsin', 'segmentation', 'glass']
    keys = ['bad_links']
    metrics = ['ARI']
    results = pd.DataFrame({'dataset': []})


    for ds in datasets:
        for bl in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            # k - default: n_class, n_folds - default 5
            params = {'n_links': 0.2,
                      'n_init': 10,
                      'n_jobs': 10,
                      'link_weight': 1,
                      'verbose': False,
                      'bad_links': bl}

            key_params = [p for p in params if p in keys]
            res = {k: [params[k]] for k in key_params}
            res['dataset'] = [ds]
            res_filename = "_".join(["{}:{}".format(k, v[0]) for k, v in res.iteritems()]) + ".csv"

            res_path = os.path.join(results_dir, res_filename)

            if not os.path.exists(res_path):
                ret = run(ds, metrics, params)
                res.update({m: [ret[m]] for m in metrics})
                part_results = pd.DataFrame.from_dict(res)
                part_results.to_csv(res_path)
            else:
                part_results = pd.read_csv(res_path)

            results = results.append(part_results)

    save_path = os.path.join(results_dir, 'bad_links.csv')
    results.to_csv(save_path, index=False)