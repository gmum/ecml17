from __future__ import division

from solvers import assign_to_closest

import numpy as np
from utils import must_consistent, cannot_consistent
from . import logger
from scipy.stats import multivariate_normal
from itertools import product



class ConstrainedHartigan(object):

    def __init__(self,
                 link_mult,
                 del_treshold,
                 max_iter,
                 tol,
                 verbose,
                 link_weight,
                 bad_prob=0.,
                 cov_type='standard',
                 init='random'):

        self.verbose = verbose
        self.max_iter = max_iter
        self.del_treshold = del_treshold
        self.link_weight = link_weight
        self.link_multiplier = link_mult
        self.tol = tol
        self.init = init
        self.cov_type = cov_type
        self.cov_param = 1
        self.bad_prob = bad_prob

        self.full_labels = None
        self.point_weights = None


    def kmeans_cost(self, clust_id):

        def _fixed_spherical(cov):
            """
            param: float
                fixed gaussian radius
            """
            assert isinstance(self.cov_param, float) or isinstance(self.cov_param, int)
            assert self.cov_param > 0

            return (cov.shape[0] / 2) * np.log(2 * np.pi) \
                   + (cov.shape[0] / 2) * np.log(self.cov_param) \
                   + (1 / (2 * self.cov_param)) * np.trace(cov)

        return self.weights[clust_id] * _fixed_spherical(self.covs[clust_id])

    def full_cov_cost(self, clust_id):

        def _xentropy(cov):
            return (cov.shape[0] / 2) * np.log(2 * np.pi * np.e) \
                   + 0.5 * np.log(np.linalg.det(cov))

        return self.weights[clust_id] * _xentropy(self.covs[clust_id])


    def speherical_cost(self, clust_id):

        def _spherical(cov):
            return (cov.shape[0] / 2) * np.log(2 * np.pi * np.e / cov.shape[0]) \
                   + (cov.shape[0] / 2) * np.log(np.trace(cov))

        return self.weights[clust_id] * _spherical(self.covs[clust_id])


    def calculate_new_params(self, x, n, add, weight, mean, cov):

        coef = 1 if add else -1

        # more then one point
        if x.ndim > 1:
            clust_n = 2 * self.link_weight
            clust_mean = np.mean(x, axis=0).reshape(1, -1)
            clust_cov = np.cov(x.T, bias=True)
        # one point
        else:
            clust_n = 1
            clust_mean = x
            clust_cov = 0

        count = weight * n

        p_1 = count / (count + (coef * clust_n))
        p_2 = clust_n / (count + (coef * clust_n))

        new_weight = weight + coef * (clust_n / n)
        new_mean = p_1 * mean + coef * (p_2 * clust_mean)

        new_cov = (p_1 * cov) + coef * (p_2 * clust_cov) + coef * (p_1 * p_2 * np.outer((mean - clust_mean), (mean - clust_mean)))

        return new_weight, new_mean, new_cov

    def calculate_cost(self, x, current, candidate):

        assert current != candidate

        current_cost = self.cec_cost(current) + self.cec_cost(candidate)

        # calculate evergy for candidte cluster
        weight_added, mean_added, cov_added = self.calculate_new_params(x,
                                                                        n=self.true_n,
                                                                        weight=self.weights[candidate],
                                                                        mean=self.means[candidate],
                                                                        cov=self.covs[candidate],
                                                                        add=True)

        self.weights[candidate] = weight_added
        self.means[candidate] = mean_added
        self.covs[candidate] = cov_added

        # calculate energy for current cluster
        if self.removed_clusters[current]:
            current_cost = np.inf
            cost_removed = 0
        else:
            weight_removed, mean_removed, cov_removed = self.calculate_new_params(x,
                                                                                  n=self.true_n,
                                                                                  weight=self.weights[current],
                                                                                  mean=self.means[current],
                                                                                  cov=self.covs[current],
                                                                                  add=False)

            self.weights[current] = weight_removed
            self.means[current] = mean_removed
            self.covs[current] = cov_removed

            cost_removed = self.cec_cost(current)


        cost_added = self.cec_cost(candidate)

        return current_cost, cost_added, cost_removed

    def __call__(self, X, constraints, k, seed):

        self.rng = np.random.RandomState(seed)
        self.k = k
        self.seed = seed

        if self.link_weight == 'auto':
            self.link_weight = 1. / self.link_multiplier

        if self.cov_type == 'standard':
            self.cec_cost = self.full_cov_cost
        elif self.cov_type == 'fixed_spherical':
            self.cec_cost = self.kmeans_cost
        elif self.cov_type == 'spherical':
            self.cec_cost = self.speherical_cost
        else:
            raise ValueError("Bad cost param")

        return self.solve_constrained_hartigan(X, constraints)

    def solve_constrained_hartigan(self, X, constraints):

        if self.verbose:
            logger.info(' Initializing...')

        if self.init == 'random':
            while True:
                init_means = X[self.rng.choice(X.shape[0], size=self.k, replace=False)]
                init_labels = assign_to_closest(X, init_means)
                init_weights = np.array([np.sum(init_labels == i) / X.shape[0] for i in xrange(self.k)])
                init_means = np.array([np.mean(X[np.where(init_labels == i)[0]], axis=0) for i in xrange(self.k)])
                init_covs = np.array([np.cov(X[np.where(init_labels == i)[0]].T, bias=True) for i in xrange(self.k)])

                if all(np.linalg.det(init_covs) > 0):
                    break
                else:
                    self.seed += 1
                    self.rng = np.random.RandomState(self.seed)
        else:
            raise NotImplementedError("Other inits not yet implemented")

        self.weights = init_weights
        self.means = init_means
        self.covs = init_covs
        self.labels = init_labels

        init_energy = np.sum(np.array([self.cec_cost(i) for i in range(self.k)]))

        # for tracking removed clusters
        self.removed_clusters = self.k * [False]
        self.alive_gausses = range(self.k)

        self.partition = [[i] for i in range(self.k)]
        self.c2y = {i: i for i in range(self.k)}

        if self.verbose:
            logger.info("Initialisation done, Energy {}".format(init_energy))

        # clusters points for calculating means, covs and wieghts with links
        clusters = [X[np.where(self.labels == i)[0]].tolist() for i in xrange(self.k)]

        # multiplier for links
        assert isinstance(self.link_multiplier, int)
        multiplier = self.link_multiplier

        multi_links = []
        link_labels = []
        link_weights = np.zeros(self.k)

        # replicate links and their initial labeling
        for link in constraints:

            id1, id2, label = link
            assert label in [0,1]
            x1, x2 = X[id1], X[id2]

            # get labels for current link
            must_label = self.init_link_labels(x1, x2, 1)
            cannot_label = self.init_link_labels(x1, x2, -1)

            # change label from 0 to -1
            label = label if label == 1 else -1

            if label == 1: # must link
                for m in xrange(multiplier):
                    set_label = 1
                    link_y = must_label

                    multi_links += [(x1, x2, set_label)]
                    link_labels += [link_y]

                    clusters[link_y[0]] += [x1]
                    clusters[link_y[1]] += [x2]
                    link_weights[link_y[0]] += self.link_weight
                    link_weights[link_y[1]] += self.link_weight

            elif label == -1: #cannot link
                for m in xrange(multiplier):
                    link_y = cannot_label
                    set_label = -1

                    multi_links += [(x1, x2, set_label)]
                    link_labels += [link_y]

                    clusters[link_y[0]] += [x1]
                    clusters[link_y[1]] += [x2]
                    link_weights[link_y[0]] += self.link_weight
                    link_weights[link_y[1]] += self.link_weight

        self.true_n = X.shape[0] + (2 * len(multi_links) * self.link_weight)

        # concat points to links and labels to link_labels
        XC = X.tolist() + multi_links
        clust_labels = self.labels.tolist() + link_labels

        assert len(clust_labels) == len(XC)

        # shuffle points and labels
        perm = self.rng.permutation(len(XC))
        reindex = perm.tolist()
        XC = np.array(XC)[perm].tolist()
        clust_labels = np.array(clust_labels, dtype=object)[perm].tolist()

        self.means = np.array([np.mean(np.vstack(x), axis=0) for x in clusters])
        self.covs = np.array([np.cov(np.vstack(x).T, bias=True) for x in clusters])

        for k in xrange(self.k):
            points_count = np.where(self.labels == k)[0].shape[0]
            links_count = link_weights[k]
            if links_count == 0:
                continue

            self.weights[k] = (points_count / self.true_n) + (links_count / self.true_n)

        ### Do EM loop
        it = 0
        removed_now = False
        energies = []

        if self.verbose:
            logger.info("Starting EM loop")

        while it <= self.max_iter:

            change = False
            updating_iter = False

            if removed_now:
                removed_now = False
                updating_iter = True

            for idx, (x, l) in enumerate(zip(XC, clust_labels)):

                # constraint
                if isinstance(l, tuple) or isinstance(l, list):
                    p = x[-1]
                    x = np.vstack(x[:-1])
                # single point
                elif isinstance(l, int):
                    p = 0
                    x = np.array(x)
                else:
                    raise TypeError("Unknown label type: {}".format(type(l)))

                # constraint
                if p in [-1,1]:

                    for cand_c1, cand_c2 in product(xrange(self.k), xrange(self.k)):

                        # at least one of the candidates was removed, skip
                        if self.removed_clusters[cand_c1] or self.removed_clusters[cand_c2]:
                            continue

                        # cannot-link and candidates are in the same class, skip
                        if p == -1 and self.c2y[cand_c1] == self.c2y[cand_c2]:
                            continue

                        # must-link and candidates are in different class, skip
                        elif p == 1 and self.c2y[cand_c1] != self.c2y[cand_c2]:
                            continue

                        curr_c1, curr_c2 = clust_labels[idx]

                        old_weights = self.weights.copy()
                        old_means = self.means.copy()
                        old_covs = self.covs.copy()

                        if curr_c1 != cand_c1:
                            current_cost_1, cost_added_1, cost_removed_1 = self.calculate_cost(x[0],
                                                                                               current=curr_c1,
                                                                                               candidate=cand_c1)

                        else:
                            cost_added_1 = 0
                            cost_removed_1 = 0
                            current_cost_1 = 0

                        if cand_c2 != curr_c2:
                            current_cost_2, cost_added_2, cost_removed_2 = self.calculate_cost(x[1],
                                                                                               current=curr_c2,
                                                                                               candidate=cand_c2)

                        else:
                            cost_added_2 = 0
                            cost_removed_2 = 0
                            current_cost_2 = 0

                        if (cost_added_1 + cost_added_2 + cost_removed_1 + cost_removed_2) < (current_cost_1 + current_cost_2):
                            # assign x to new cluster
                            clust_labels[idx] = [cand_c1, cand_c2]

                            change = True

                            if not updating_iter:
                                # check for clusters to remove
                                if not self.removed_clusters[curr_c1] and self.weights[curr_c1] < self.del_treshold:
                                    if self.verbose:
                                        logger.info("\t {} Removing small cluster {}, running updating iteration".format(p, curr_c1))
                                    self.removed_clusters[curr_c1] = True
                                    self.max_iter += 1

                                # check for clusters to remove
                                if not self.removed_clusters[curr_c2] and self.weights[curr_c2] < self.del_treshold:
                                    if self.verbose:
                                        logger.info("\t {} Removing small cluster {}, running updating iteration".format(p, curr_c2))
                                    self.removed_clusters[curr_c2] = True
                                    removed_now = True
                                    self.max_iter += 1

                        else:
                            self.weights = old_weights
                            self.means = old_means
                            self.covs = old_covs

                # single point
                elif p == 0:
                    for candidate_cl in xrange(self.k):

                        current_cl = clust_labels[idx]

                        # skip removed cluster or x's current cluster
                        if self.removed_clusters[candidate_cl] or candidate_cl == current_cl:
                            continue

                        old_weights = self.weights.copy()
                        old_means = self.means.copy()
                        old_covs = self.covs.copy()

                        current_cost, cost_added, cost_removed = self.calculate_cost(x,
                                                                                     current=current_cl,
                                                                                     candidate=candidate_cl)

                        # check if changing x's cluster would result in lower energy
                        if (cost_removed + cost_added) < current_cost:

                            change = True

                            # assign x to new cluster
                            clust_labels[idx] = candidate_cl

                            if not updating_iter:
                                # check for clusters to remove
                                if not self.removed_clusters[current_cl] and self.weights[current_cl] < self.del_treshold:
                                    if self.verbose:
                                        logger.info("\t {} Removing small cluster {}, running updating iteration".format(p, current_cl))
                                    self.removed_clusters[current_cl] = True
                                    removed_now = True
                                    self.max_iter += 1
                        else:
                            self.weights = old_weights
                            self.means = old_means
                            self.covs = old_covs

            if not removed_now:

                energy = np.sum(np.array([self.cec_cost(i) for i in range(self.k) if not self.removed_clusters[i]]))

                self.alive_gausses = [c for c in range(self.k) if not self.removed_clusters[c]]

                it += 1

                if it == self.max_iter:
                    if self.verbose:
                        logger.warning("\t Maximum number of iterations reached, final energy: {}".format(energy))
                    self.full_labels = np.array([clust_labels[reindex.index(i)] for i in range(len(X))])
                    ret_labels = np.array([self.c2y[clust_labels[reindex.index(i)]] for i in range(len(X))])

                    return energy, ret_labels, self.weights, self.means, self.covs

                if self.verbose:
                    logger.info("\t Iter: {} Enegry: {}".format(it, energy))

                energies.append(energy)

                if not change:
                    if self.verbose:
                        logger.info("\t No switch in clusters, done")
                    break

                if len(energies) > 3 and np.std(energies[-3:]) < self.tol:
                    if self.verbose:
                        logger.info("\t Energy change less than tolerance, done")
                    break

        self.full_labels = np.array([clust_labels[reindex.index(i)] for i in range(len(X))])
        ret_labels = np.array([self.c2y[clust_labels[reindex.index(i)]] for i in range(len(X))])

        points_clusters = [X[np.where(self.full_labels == l)[0]] for l in self.alive_gausses]

        points_weights = np.array([np.vstack(x).shape[0] / len(X) for x in points_clusters])
        points_means = np.array([np.mean(np.vstack(x), axis=0) for x in points_clusters])
        points_covs = np.array([np.cov(np.vstack(x).T, bias=True) for x in points_clusters])

        return energy, ret_labels, points_weights, points_means, points_covs


    def init_link_labels(self, x0, x1, p):

        pdfs = [multivariate_normal(mean=m, cov=c, allow_singular=True) for m, c in zip(self.means, self.covs)]

        neglog_costs = np.zeros((2, self.means.shape[0]))

        for i in xrange(len(self.means)):
            neglog_costs[0, i] = -np.log(pdfs[i].pdf(x0))
            neglog_costs[1, i] = -np.log(pdfs[i].pdf(x1))

        neglog_link_costs = np.clip(neglog_costs, 0, 9999999999)

        def _cost(k, l):
            return (-np.log(self.weights[k]) * neglog_link_costs[0, k]) + (-np.log(self.weights[l]) * neglog_link_costs[1, l])

        must = must_consistent(self.partition)
        cannot = cannot_consistent(self.partition)

        if p == 1:  # must link
            return must[np.argmin([_cost(k, l) for k, l in must])]
        elif p == -1:  # cannot link
            return cannot[np.argmin([_cost(k, l) for k, l in cannot])]
        else:
            raise ValueError("At this point link is expected to be -1 or 1 only!")