import sys
sys.path.append('..')

from config import config
data_dir = config['data_dir']
from config import logger
from utils import get_natural_constraints, get_balanced_constraints, get_beta_distributed_links, ARI

import numpy as np
from CEC import CEC
import os
import pandas as pd


def run(data_fname, metrics, params):

    X = np.loadtxt(os.path.join(data_dir, data_fname + '_data.csv'), delimiter=',')
    y = np.loadtxt(os.path.join(data_dir, data_fname + '_label.csv'), delimiter=',')
    assert X.shape[0] == y.shape[0]

    ### Process experiment params
    # n_folds
    n_folds = params.get('n_folds', 5)
    # seed
    seed = params.get('seed', np.random.randint(np.iinfo(np.int32).max))

    ### Process links params
    # n_links - number of links in relation to number of data
    assert 'n_links' in params.keys()
    n_links = params['n_links']
    if n_links == 'auto':
        n_links = int(0.1 * X.shape[0])
    else:
        assert isinstance(n_links, float) or isinstance(n_links, int)
        assert n_links >= 0 and n_links < 1

        n_links = int(n_links * X.shape[0])

    # ratio of must/cannot
    ratio = params.get('ratio', 'natural')
    if ratio == 'natural':
        natural_constraints = True
    else:
        natural_constraints = False
        assert isinstance(ratio, float) or isinstance(ratio, int)
        assert ratio >= 0 and ratio <= 1

    # probability of bad links
    bad_links = params.get('bad_links', 0.)
    assert bad_links >= 0 and bad_links <= 1

    ### Process  CEC params

    # k
    k = params.get('k', len(np.unique(y)))
    # n_jobs
    n_jobs = params.get('n_jobs', 10)
    # n_init
    n_init = params.get('n_init', 10)
    # link_weight
    link_weight = params.get('link_weight', 0.5)
    # cov_type
    cov_type = params.get('cov_type', 'fixed_spherical')
    # verbosity
    verbose = params.get('verbose', False)


    # Prepare results
    results = {m: [] for m in metrics}

    logger.info("Starting dataset {} with metrics: {} and params: {}".format(data_fname.upper(), ", ".join(metrics), params))

    if n_links == 0 and cov_type == 'fixed_spherical':
        from sklearn.cluster import KMeans
        p = KMeans(n_clusters=k).fit(X).predict(X)
        for metric in metrics:
            results[metric] = globals().get(metric)(y, p)
    else:
        # run CEC
        cec = CEC(k=k,
                  n_init=n_init,
                  n_jobs=n_jobs,
                  link_weight=link_weight,
                  seed=seed,
                  verbose=verbose,
                  cov_type=cov_type,
                  cov_param=1)

        for i in xrange(n_folds):
            if natural_constraints:
                links = get_natural_constraints(y, n=n_links, seed=seed+i, bad_prob=bad_links)
            else:
                links = get_balanced_constraints(y, n_links, ratio, seed+i)

            p = cec.fit_predict(X, constraints=links)

            for metric in results.keys():
                results[metric].append(globals().get(metric)(y, p))

            logger.info("\t fold {}/{}".format(i+1, n_folds))

        # Average results
        for metric, scores in results.iteritems():
            results[metric] = np.mean(scores)

    return results


