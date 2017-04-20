from __future__ import division

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster.k_means_ import _k_init
from sklearn.utils.extmath import row_norms
from . import logger


def assign_to_closest(X, centers, metric='euclidean'):
    return np.argmin(pairwise_distances(X, centers, metric=metric), axis=1)


class HartiganSolver(object):

    def __init__(self,
                 init,
                 cov_type,
                 cov_param,
                 tol,
                 max_iter,
                 del_treshold,
                 verbose,
                 allow_lowering_k=True):

        self.init = init
        self.equal = False
        self.verbose = verbose
        self.max_iter = max_iter
        self.del_treshold = del_treshold
        self.cov_type = cov_type
        self.cov_param = cov_param
        self.tol = tol
        self.allow_lowering_k = allow_lowering_k

        self.weights = None
        self.means = None
        self.covs = None

    def xentropy(self, cov):
        """
        Return cluster cost function for given covariance type

        :param cov_type: string
            Which covariance matrix type to use
        :param cov_param: int, float list or np.ndarray
            Parameter for fixed_covariance, fixed_spherical or eigenvalues cov_type
        :return: callable
            Cost function for given covaraince matrix type
        """
        def _standard(cov):
            return (cov.shape[0] / 2) * np.log(2 * np.pi * np.e) \
                   + 0.5 * np.log(np.linalg.det(cov))

        def _fixed_covariance(cov):
            """
            param: np.array of shape (d, d), where d is data dimensionality
                fixed non-singular covariance matrix,
            """
            assert cov.shape == self.cov_param.shape
            return (cov.shape[0] / 2) * np.log(2 * np.pi) \
                   + 0.5 * np.trace(np.linalg.inv(self.cov_param).dot(cov)) \
                   + 0.5 * np.log(np.linalg.det(self.cov_param))

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

        def _spherical(cov):

            return (cov.shape[0] / 2) * np.log(2 * np.pi * np.e / cov.shape[0]) \
                   + (cov.shape[0] / 2) * np.log(np.trace(cov))

        def _diagonal(cov):

            return (cov.shape[0] / 2) * np.log(2 * np.pi * np.e) \
                   + 0.5 * np.log(np.prod(np.diag(cov)))  # det of a diagonal matrix is the product of its diagonal

        def _eigenvalues(cov):
            """
            param: list, 1D np.array if length d, where d is data dimensionality
                list of eigenvalues
            """
            param = np.array(self.cov_param)

            assert param.ndim == 1 and param.shape[0] == cov.shape[0]
            assert (param == np.sort(param)).all(), "eigenvalues are NOT in ascending order"

            return (cov.shape[0] / 2) * np.log(2 * np.pi) \
                   + 0.5 * np.sum(np.linalg.eigvalsh(cov) / param) \
                   + 0.5 * np.log(np.prod(param))

        if self.cov_type == 'standard':
            return _standard(cov)
        elif self.cov_type == 'fixed_covariance':
            return _fixed_covariance(cov)
        elif self.cov_type == 'fixed_spherical':
            return _fixed_spherical(cov)
        elif self.cov_type == 'spherical':
            return _spherical(cov)
        elif self.cov_type == 'diagonal':
            return _diagonal(cov)
        elif self.cov_type == 'eigenvalues':
            return _eigenvalues(cov)
        else:
            raise ValueError("Wrong covariance type, cov_type={}".format(self.cov_type))

    def cec_cost(self, clust_id):

        if not self.equal:
            return self.weights[clust_id] * (-np.log(self.weights[clust_id]) + self.xentropy(self.covs[clust_id]))
        else:
            return self.weights[clust_id] * (-np.log(1 / self.k) + self.xentropy(self.covs[clust_id]))


    def get_cov(self):
        """
        Returns covariance transformation function for given covariance type
        :param cov_type: string
            Which covariance matrix type to use
        :param cov_param: int, float list or np.ndarray
            Parameter for fixed_covariance, fixed_spherical or eigenvalues cov_type
        :return: callable
            Transformation function for given covaraince matrix type
        """

        def _identity(cov):
            return cov

        def _fixed_covariance(cov):
            """
            param: np.array of shape (d, d), where d is data dimensionality
                fixed non-singular covariance matrix,
            """
            assert cov.shape == self.cov_param.shape
            return self.cov_param

        def _fixed_spherical(cov):
            """
            param: float
                fixed gaussian radius
            """
            assert isinstance(self.cov_param, float) or isinstance(self.cov_param, int)
            assert self.cov_param > 0

            return self.cov_param * np.eye(cov.shape[0])

        def _spherical(cov):

            return (np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])

        def _diagonal(cov):

            return np.diag(np.diag(cov)) # numpy <3

        def _eigenvalues(cov):
            """
            WARNING: better hug this fucntion with np.linalg.LinAlgError catching

            param: list, 1D np.array if length d, where d is data dimensionality
                list of eigenvalues
            """
            param = np.array(self.cov_param)

            assert param.ndim == 1 and param.shape[0] == cov.shape[0]
            assert (param == np.sort(param)).all(), "eigenvalues are NOT in ascending order"

            S = np.diag(param)
            # get SVD decomposition
            U, _, V = np.linalg.svd(cov)
            # substitute eigenvalues from svd with given and recalculate covariance matrix
            return np.dot(U, np.dot(S, V))

        if self.cov_type == 'standard':
            transform = _identity
        elif self.cov_type == 'fixed_covariance':
            transform = _fixed_covariance
        elif self.cov_type == 'fixed_spherical':
            transform = _fixed_spherical
        elif self.cov_type == 'spherical':
            transform = _spherical
        elif self.cov_type == 'diagonal':
            transform = _diagonal
        elif self.cov_type == 'eigenvalues':
            transform = _eigenvalues
        else:
            raise ValueError("Wrong covariance type, cov_type={}".format(self.cov_type))

        def _cov(cov):
            return transform(cov)

        return _cov

    def calculate_new_params(self, x, id_clust, add):

        coef = 1 if add else -1

        # calculate candidate_cl's energy after adding point x
        clust_count = np.sum(self.labels == id_clust)

        p_1 = clust_count / (clust_count + 1 * coef)
        p_2 = 1 / (clust_count + 1 * coef)

        self.weights[id_clust] = self.weights[id_clust] + coef * (1 / self.n)
        self.means[id_clust] = p_1 * self.means[id_clust] + coef * (p_2 * x)
        self.covs[id_clust] = p_1 * self.covs[id_clust] + coef * (p_1 * p_2 * np.outer((self.means[id_clust] - x), (self.means[id_clust] - x)))


    def __call__(self, X, k, seed):

        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.n = X.shape[0]
        self.k = k
        return self.solve(X)

    def solve(self, X):

        while True:
            if isinstance(self.init, np.ndarray):
                assert X.shape[1] == self.init.shape[1]
                assert self.init.shape[0] == self.k
                assert X.shape[0] > self.init.shape[0]
                init_means = self.init
            elif self.init == 'random':
                init_means = X[self.rng.choice(X.shape[0], size=self.k, replace=False)]
            elif self.init == 'k-means++':
                squared_norms = row_norms(X, squared=True)
                init_means = _k_init(X, n_clusters=self.k, x_squared_norms=squared_norms, random_state=self.rng)
            else:
                raise ValueError('Got unrecognized init parameter: {}'.format(self.init))

            self.labels = assign_to_closest(X, init_means)
            self.weights = np.array([np.sum(self.labels == i) / X.shape[0] for i in xrange(self.k)])


            if any(self.weights < self.del_treshold) or any(self.weights * X.shape[0] < X.shape[1] + 2):
                if self.allow_lowering_k:
                    self.k -= 1
                    if self.verbose:
                        logger.info("Failed initialization, decreasing k to {}".format(self.k))
                else:
                    self.seed += 1
                    self.rng = np.random.RandomState(self.seed)
            else:
                break

        self.means = np.array([np.mean(X[np.where(self.labels == i)[0]], axis=0) for i in xrange(self.k)])
        self.covs = np.array([np.cov(X[np.where(self.labels == i)[0]].T, bias=True) for i in xrange(self.k)])

        self.removed_clusters = self.k * [False]

        removed_now = False
        it = 0
        energies = []
        while it <= self.max_iter:

            change = False
            update_iter = False

            if removed_now:
                removed_now = False
                update_iter = True

            for idx, x in enumerate(X):

                for candidate_cl in xrange(self.k):

                    current_cl = self.labels[idx]

                    # skip removed cluster or x's current cluster
                    if self.removed_clusters[candidate_cl] or candidate_cl == current_cl:
                        continue

                    current_cost = self.cec_cost(current_cl) + self.cec_cost(candidate_cl)

                    old_weights = self.weights.copy()
                    old_means = self.means.copy()
                    old_covs = self.covs.copy()

                    # calculate evergy for candidte cluster
                    self.calculate_new_params(x, candidate_cl, add=True)
                    cost_added = self.cec_cost(candidate_cl)

                    # calculate energy for current cluster
                    if self.removed_clusters[current_cl]:
                        current_cost = np.inf
                        cost_removed = 0
                    else:
                        self.calculate_new_params(x, current_cl, add=False)
                        cost_removed = self.cec_cost(current_cl)


                    # check if changing x's cluster would result in lower energy
                    if (cost_removed + cost_added) < current_cost:
                        # assign x to new cluster
                        self.labels[idx] = candidate_cl
                        change = True

                        # delete small cluster
                        if not update_iter and not self.removed_clusters[current_cl]:
                            if self.weights[current_cl] < self.del_treshold or np.sum(self.labels == current_cl) < X.shape[1] + 2:
                                if self.verbose:
                                    logger.info("\t Deleting small cluster {}, running updating iteration".format(current_cl))
                                self.removed_clusters[current_cl] = True
                                removed_now = True
                                self.max_iter += 1

                    else:
                        self.weights = old_weights
                        self.means = old_means
                        self.covs = old_covs

            # pdb.set_trace()
            if not removed_now:
                energy = np.sum(np.array([self.cec_cost(i) for i in range(self.k) if not self.removed_clusters[i]]))

                it += 1

                if it == self.max_iter:
                    if self.verbose:
                        logger.warning("\t Maximum number of iterations reached, final energy: {}".format(energy))
                    break

                if self.verbose:
                    logger.info("\t Iter {} Enegry {}".format(it, energy))
                    # logger.info("Weights: {}".format(weights))

                if not change:
                    if self.verbose:
                        logger.info("\t No switch in clusters, done")
                    break

                energies.append(energy)

                if len(energies) > 3 and np.std(energies[-3:]) < self.tol:
                    if self.verbose:
                        logger.info("\t Energy change less than tolerance, done")
                    break

        alive_clusters = np.invert(self.removed_clusters)
        weights = self.weights[alive_clusters]
        means = self.means[alive_clusters]
        covs = self.covs[alive_clusters]

        return energy, self.labels, weights, means, covs



# TODO: refactor into a class

# def solve_lloyd(X,
#                 k,
#                 tol,
#                 max_iter,
#                 seed,
#                 del_treshold,
#                 verbose,
#                 cov_type,
#                 cov_param,
#                 init,
#                 del_clusters=True,
#                 **kwargs):
#
#     def _assign_point(x):
#         costs = [-np.log(weights[i])
#                  - np.log(multivariate_normal.pdf(x, mean=means[i], cov=covs[i]))
#                  for i in xrange(k)]
#
#         return np.argmin(costs)
#
#     # initlize parameters
#     rng = np.random.RandomState(seed)
#
#     cost = get_cost(cov_type, cov_param)
#     cov = get_cov(cov_type, cov_param)
#
#     if isinstance(init, np.ndarray):
#         assert X.shape[1] == init.shape[1]
#         assert init.shape[0] == k
#         assert X.shape[0] > init.shape[0]
#         means = init
#     elif init == 'random':
#         means = X[rng.choice(X.shape[0], size=k, replace=False)]
#     elif init == 'k-means++':
#         squared_norms = row_norms(X, squared=True)
#         means = _k_init(X, n_clusters=k, x_squared_norms=squared_norms, random_state=rng)
#     else:
#         raise ValueError('Got unrecognized init parameter: {}'.format(init))
#
#     weights = np.ones(k) / k  # p_i
#
#     covs = k * [cov(np.cov(X.T, bias=True))]
#
#     energy = [np.inf]  # h
#     labels = -np.ones(X.shape[0])  # cluster assignments
#
#     it = 0
#     while it <= max_iter:
#
#         # assign points to new clusters
#         for idx, x in enumerate(X):
#             try:
#                 y = _assign_point(x)
#             except np.linalg.LinAlgError:
#                 logger.error("\t Singular covariance matrix")
#                 energy.append(np.inf)
#                 return np.inf, labels, weights, means, covs
#
#             labels[idx] = y
#
#         # delete small clusters
#         if del_clusters:
#             id_cluster = 0
#             while id_cluster < k:
#                 if np.sum(labels == id_cluster) < del_treshold * X.shape[0] or np.sum(labels == id_cluster) < X.shape[1] + 2:
#
#                     if verbose:
#                         logger.info("\t Deleting small cluster {}".format(id_cluster))
#
#                     # delete cluster params
#                     k -= 1
#                     weights = np.delete(weights, id_cluster, axis=0)
#                     means = np.delete(means, id_cluster, axis=0)
#                     covs = np.delete(covs, id_cluster, axis=0)
#
#                     # reindex labels and reassign to nex clusters
#                     for idx, x in enumerate(X):
#                         # reassign point to new cluster
#                         if labels[idx] == id_cluster:
#                             try:
#                                 y = _assign_point(x)
#                             except np.linalg.LinAlgError:
#                                 logger.error("\t Singular covariance matrix")
#                                 energy.append(np.inf)
#                                 return energy, labels, weights, means, covs
#
#                             labels[idx] = y
#                         # reindex label
#                         elif labels[idx] > id_cluster:
#                             labels[idx] -= 1
#                 else:
#                     id_cluster += 1
#
#         # TODO: speed this up? less np.where, less list comprehensions
#         # compute new parameters
#         means = np.array([np.mean(X[np.where(labels == i)[0]], axis=0) for i in xrange(k)])
#         covs = np.array([cov(np.cov(X[np.where(labels == i)[0]].T, bias=True)) for i in xrange(k)])
#         weights = np.array([np.sum(labels == i) / X.shape[0] for i in xrange(k)])
#
#         energy.append(np.sum(np.array([cost(weights[i], covs[i]) for i in range(k)])))
#
#         it += 1
#
#         if verbose:
#             logger.info("\t Iter {} energy {}".format(it, energy[-1]))
#
#         if energy[-1] <= energy[-2] and np.abs(energy[-1] - energy[-2]) <= tol:
#             break
#
#     assert np.sum(labels == -1) == 0, "Some points were not assigned to a cluster somehow"
#
#     if it == max_iter:
#         logger.warning("\t Maximum number of iterations reached, final energy: {}".format(energy[-1]))
#
#     if verbose:
#         logger.info("\t Done, final energy: {}".format(energy[-1]))
#
#     return energy[-1], labels, weights, means, covs


