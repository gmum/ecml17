from __future__ import division
from six import string_types

import logging
logging.basicConfig()
logger = logging.getLogger("CEC")
logger.setLevel(logging.INFO)

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import logsumexp

from joblib import Parallel, delayed

from solvers import HartiganSolver
from constrainted_hartigan import ConstrainedHartigan

COV_TYPES = ['standard', 'fixed_covariance', 'fixed_spherical', 'spherical', 'diagonal', 'eigenvalues']


class CEC(BaseEstimator):

    def __init__(self,
                 k,
                 method='Hartigan',
                 cov_type='fixed_spherical',
                 cov_param=None,
                 init="random",
                 tol=1e-5,
                 max_iter=100,
                 del_treshold=0.03,
                 n_init=10,
                 n_jobs=1,
                 link_weight='auto',
                 link_multiplier=1,
                 seed=None,
                 verbose=False):

        assert isinstance(method, string_types)
        assert method in ['Lloyd', 'Hartigan']
        self.method = method

        assert isinstance(k, int)
        assert k > 0
        self.init_k = k
        self.k = k

        assert isinstance(cov_type, string_types)
        assert cov_type in COV_TYPES
        self.cov_type = cov_type
        self.cov_param = cov_param

        assert isinstance(init, string_types) or isinstance(init, np.ndarray)
        if isinstance(init, string_types): assert init in ['random', 'k-means++', 'cec']
        self.init = init

        assert isinstance(tol, float) or isinstance(tol, int)
        self.tol = tol

        assert isinstance(max_iter, int)
        assert max_iter > 0
        self.max_iter = max_iter

        assert isinstance(del_treshold, float)
        assert del_treshold < 0.5
        self.del_treshold = del_treshold

        assert isinstance(n_init, int)
        assert n_init > 0
        self.n_init = n_init

        assert isinstance(n_jobs, int)
        assert n_jobs > 0
        self.n_jobs = n_jobs

        assert isinstance(link_weight, float) or isinstance(link_weight, int) or isinstance(link_weight, string_types)
        if isinstance(link_weight, int) or isinstance(link_weight, float):
            assert link_weight > 0
        self.link_weight = link_weight

        assert isinstance(link_multiplier, int)
        assert link_multiplier > 0
        self.link_multiplier = link_multiplier

        if seed is not None:
            assert isinstance(seed, int) or isinstance(seed, np.random.RandomState)
            if isinstance(seed, np.random.RandomState):
                self.seed = None
                self.rng = seed
            elif isinstance(seed, int):
                self.seed = seed
                self.rng = np.random.RandomState(seed)
        else:
            self.seed = np.random.randint(np.iinfo(np.int32).max)
            self.rng = np.random.RandomState(self.seed)

        assert isinstance(verbose, bool) or isinstance(verbose, int)
        if isinstance(verbose, int):
            assert verbose in [0,1]
            self.verbose = bool(verbose)
        else:
            self.verbose = verbose

    def fit_predict(self, X, constraints=None):

        self.fit(X, constraints=constraints)
        return self.labels

    def fit(self, X, constraints=None):

        if constraints is not None:
            if not isinstance(constraints, np.ndarray):
                try:
                    constraints = np.array(constraints)
                except:
                    raise TypeError('Could not cast constraints to numpy array, pass an array')

            if self.cov_type not in ['standard', 'spherical', 'fixed_spherical']:
                raise ValueError("cov_params with constriants are: standard, spherical")

            self.solver = ConstrainedHartigan(link_mult=self.link_multiplier,
                                              link_weight=self.link_weight,
                                              del_treshold=self.del_treshold,
                                              max_iter=self.max_iter,
                                              verbose=self.verbose,
                                              tol=self.tol,
                                              cov_type=self.cov_type)

            params = {'X': X, 'constraints': constraints, 'k': self.k}
        else:
            if self.method == 'Hartigan':
                self.solver = HartiganSolver(init=self.init,
                                             cov_type=self.cov_type,
                                             cov_param=self.cov_param,
                                             max_iter=self.max_iter,
                                             del_treshold=self.del_treshold,
                                             verbose=self.verbose,
                                             tol=self.tol,
                                             allow_lowering_k=False)
            else:
                raise NotImplementedError()

            params = {'X': X, 'k': self.k}

        if self.n_jobs == 1:
            best_energy = np.inf
            best_labels = []
            best_centers = []
            best_covs = []
            best_probs = []
            for run in xrange(self.n_init):
                if self.verbose:
                    logger.info("Run {}/{}".format(run+1, self.n_init))
                seed = self.seed + run
                energy, labels, probs, centers, covs = self.solver(seed=seed, **params)
                if energy < best_energy:
                    best_energy = energy
                    best_labels = labels
                    best_probs = probs
                    best_centers = centers
                    best_covs = covs

        else:
            if self.verbose:
                logger.info("Running {} parallel jobs for {} solver".format(self.n_jobs, self.method))
            seeds = self.rng.randint(np.iinfo(np.int32).max, size=self.n_init)

            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self.solver)(seed=seed, **params) for seed in seeds)

            energies, labels, probs, centers, covs = zip(*results)
            best_energy = np.min(energies)
            best_run = np.argmin(energies)
            best_labels = labels[best_run]
            best_probs = probs[best_run]
            best_centers = centers[best_run]
            best_covs = covs[best_run]

        self.energy = best_energy
        self.means = best_centers
        self.labels = best_labels
        self.covariances = best_covs
        self.weights = best_probs

        # self.k = self.weights.shape[0]


    def predict(self, X):

        assert(self.means is not None), "Model not trained"

        costs = np.array([-np.log(self.weights[i])
                          - np.log(multivariate_normal.pdf(X, mean=self.means[i], cov=self.covariances[i]))
                          for i in xrange(self.k)]).T

        return np.argmin(costs, axis=1)

    def predict_proba(self, X):

        assert (self.means is not None), "Model not trained"

        weighted_log_probs = np.log(self.weights) \
                             + np.array([multivariate_normal.logpdf(X, mean=self.means[i], cov=self.covariances[i])
                                  for i in xrange(self.k)]).T
        log_prob_norm = logsumexp(weighted_log_probs, axis=1)
        return np.exp(weighted_log_probs - log_prob_norm[:, np.newaxis])

























