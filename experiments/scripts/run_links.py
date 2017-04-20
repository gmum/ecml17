import sys
sys.path.append('..')

from config import config
data_dir = config['data_dir']
from config import logger
from utils import get_natural_constraints, get_balanced_constraints, get_beta_distributed_links, links_accuracy
from sklearn.model_selection import KFold

import numpy as np
from CEC import CEC
import os
import pandas as pd


def run(data_fname, metrics, params):

    X = np.loadtxt(os.path.join(data_dir, data_fname + '_data.csv'), delimiter=',')
    link_fname = data_fname.split('_')[0]
    links = np.loadtxt(os.path.join(data_dir, link_fname + '_links.csv'), delimiter=',').astype(int)
   
    ### Process experiment params
    # n_folds
    n_folds = params.get('n_folds', 5)
    # seed
    seed = params.get('seed', np.random.randint(np.iinfo(np.int32).max))

    # probability of bad links
    bad_links = params.get('bad_links', 0.)
    assert bad_links >= 0 and bad_links <= 1

    ### Process  CEC params

    # k
    k = params.get('k', 10)
    # n_jobs
    n_jobs = params.get('n_jobs', 10)
    # n_init
    n_init = params.get('n_init', 10)
    # link_weight
    link_weight = params.get('link_weight', 1)
    # cov_type
    cov_type = params.get('cov_type', 'fixed_spherical')
    # verbosity
    verbose = params.get('verbose', False)

    # Prepare results
    results = {m: [] for m in metrics}

    logger.info("Starting dataset {} with metrics: {} and params: {}".format(data_fname.upper(), ", ".join(metrics), params))
    
    # prepare folds
    folds = KFold(n_splits=n_folds, shuffle=False)

    # run CEC
    cec = CEC(k=k,
              n_init=n_init,
              n_jobs=n_jobs,
              link_weight=link_weight,
              seed=seed,
              verbose=verbose,
              cov_type=cov_type)

    for i, (train, test) in enumerate(folds.split(links)):
	
        p = cec.fit_predict(X, constraints=links[train])

        for metric in results.keys():
            results[metric].append(globals().get(metric)(links[test], p))

        logger.info("\t fold {}/{}".format(i+1, n_folds))

    # Average results
    for metric, scores in results.iteritems():
        results[metric] = np.mean(scores)

    return results


