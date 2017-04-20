from __future__ import division
from six import string_types

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster.supervised import comb2
from scipy.misc import comb
from itertools import combinations
import os
from config import config
import subprocess
data_dir = config['data_dir']

results_dir = config['results_dir']

### PROCESSING, LOADING, SAVING


def get_key_params(params, keys):
    keys = ['link_weight', 'n_links', 'del_clusters', 'gauss_per_clust', 'ratio', 'probability', 'bad_links']
    key_params = []
    for key in params.keys():
        if key in keys:
            key_params.append(key)

    return key_params


def get_exp_path(key_params):

    path = "_".join([key + ':' + str(value) for key, value in key_params.iteritems()])
    path += '.csv'
    path = os.path.join(results_dir, path)

    return path

### MPCK-means functions

def weka_runner(params, verbose=False):
    assert 'filename' in params.keys()
    assert 'k' in params.keys()
    assert 'result_file' in params.keys()

    weka_path = 'java -classpath /home/sieradzki/wekaUT/weka-latest weka/clusterers/MPCKMeans'
    command = weka_path + " -D {}".format(params['filename'])
    if 'links' in params.keys():
        command += ' -C {}'.format(params['links'])
    command += ' -O {}'.format(params['result_file'])
    command += ' -T 4'
    command += ' -N {}'.format(params['k'])
    command += ' -K -1 -U'

    print command

    ret = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    if verbose:
        print ret

def generate_arff(data_fname, with_label=False):

    X = np.loadtxt(os.path.join(data_dir, data_fname + '_data.csv'), delimiter=',')
    if with_label:
        y = np.loadtxt(os.path.join(data_dir, data_fname + '_label.csv'), delimiter=',')

    arff = "@RELATION {}\n\n".format(data_fname)
    for i in range(X.shape[1]):
        arff += "@ATTRIBUTE {} NUMERIC\n".format(i)

    if with_label:
        arff += "@ATTRIBUTE class NUMERIC\n"

    arff += "\n@DATA\n"
    for i in range(X.shape[0]):
        if with_label:
            arff += ",".join([str(xi) for xi in X[i]])
            arff += ",{}\n".format(int(y[i]))
        else:
            arff += ",".join([str(xi) for xi in X[i]]) + "\n"

    arff += u"%\n%\n%\n"

    with open(os.path.join(data_dir, 'weka', "{}_data.arff".format(data_fname)), 'w') as f:
        f.write(arff)

### cGMM

def matlab_runner(script_path, params, verbose=False):
    matlab_path = '/usr/local/lib/MATLAB/R2014b/bin/glnxa64/MATLAB -nodisplay -nosplash -nodesktop -r '

    params_list = []

    for key, value in params.iteritems():
        assert isinstance(key, string_types)
        if isinstance(value, string_types):
            value = "'" + value + "'"
        else:
            value = str(value)
        params_list.append("{}={}".format(key, value))

    command = matlab_path + '"' + ";".join(
        params_list) + ';' + "try, run('" + script_path + '\'), catch, exit, end, exit"'

    ret = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

    if verbose:
        print ret

### CONSTRAINTS

def get_natural_constraints(y, n, seed=None, bad_prob=0):
    if seed is not None:
        assert isinstance(seed, int)
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(np.iinfo(np.int32).max)

    links = []
    for i in xrange(n):
        id1, id2 = rng.randint(0, y.shape[0], 2)

        p = int(y[id1] == y[id2])
        if rng.rand() <= bad_prob:
            p = (p + 1) % 2
        links.append([id1, id2, p])

    return np.array(links)


def get_balanced_constraints(y, n, ratio=0.5, seed=None):
    """
    ratio: float in [0,1]
        ratio must-be constraints to all, meaning number of must-be = ratio*n, number of cannot-be = (1-ratio)*n
    """

    assert ratio >= 0 and ratio <= 1
    assert isinstance(n, int)
    assert n > 0

    if seed is not None:
        assert isinstance(seed, int)
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(np.iinfo(np.int32).max)


    constraints = []
    classes = np.unique(y)

    # must-be
    for i in xrange(int(ratio * n)):
        l = rng.choice(classes)
        constraints.append(rng.choice(np.where(y == l)[0], 2).tolist() + [1])

    # cannot-be
    for i in xrange(int((1 - ratio) * n)):
        l1, l2 = rng.choice(classes, 2, replace=False)
        id1 = rng.choice(np.where(y == l1)[0])
        id2 = rng.choice(np.where(y == l2)[0])
        constraints.append([id1, id2, 0])

    rng.shuffle(constraints)
    return np.array(constraints)


def get_beta_distributed_links(y, n, ratio, beta=5, alpha=5, bad_prob=0., seed=None):

    if seed is not None:
        assert isinstance(seed, int)
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(np.iinfo(np.int32).max)

    links = []
    if ratio == 'natural':
        for i in xrange(n):
            id1, id2 = rng.randint(0, y.shape[0], 2)

            # correct link
            if rng.rand() > bad_prob:
                must_p = rng.beta(alpha, 1)
                cannot_p = rng.beta(1, beta)
            # bad link
            else:
                must_p = rng.beta(1, beta)
                cannot_p = rng.beta(alpha, 1)
                # must link
            if y[id1] == y[id2]:
                links.append([id1, id2, must_p])
            # cannot link
            else:
                links.append([id1, id2, cannot_p])

    elif isinstance(ratio, float):
        assert ratio >= 0 and ratio <= 1

        classes = np.unique(y)
        # must-be
        for i in xrange(int(ratio * n)):
            l = rng.choice(classes)
            # correct or bad link
            p = rng.beta(alpha, 1) if rng.rand() > bad_prob else rng.beta(1, beta)
            links.append(rng.choice(np.where(y == l)[0], 2).tolist() + [p])

        # cannot-be
        for i in xrange(int((1 - ratio) * n)):
            l1, l2 = rng.choice(classes, 2, replace=False)
            id1 = rng.choice(np.where(y == l1)[0])
            id2 = rng.choice(np.where(y == l2)[0])
            # correct or bad link
            p = rng.beta(1, beta) if rng.rand() > bad_prob else rng.beta(alpha, 1)

            links.append([id1, id2, p])

        rng.shuffle(links)
    else:
        raise TypeError("Wrong `ratio` argument: {}".format(ratio))

    return np.rec.array(links, dtype=[('id1', 'i4'), ('id2', 'i4'), ('p', 'f4')])


def get_all_links(y, seed=None):

    if seed is not None:
        assert isinstance(seed, int)
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(np.iinfo(np.int32).max)

    links = []

    for i, j in combinations(np.arange(len(y)), 2):
        t = 1 if y[i] == y[j] else -1
        links.append([i, j, t])

    rng.shuffle(links)
    return links


### METRICS


def ARI(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)


def RandIndex(labels_true, labels_pred):
    n_samples = len(labels_true)
    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    a = sum(comb2(n_ij) for n_ij in contingency.data)
    b = sum(comb2(n_c) for n_c in np.ravel(contingency.sum(axis=0))) - a
    c = sum(comb2(n_c) for n_c in np.ravel(contingency.sum(axis=1))) - a
    d = comb(n_samples, 2) - a - b - c
    return (a + d) / comb(n_samples, 2)

def links_accuracy(links, labels_pred):
    score = []
    for link in links:
        # must-be
        if link[-1] == 1:
            score.append(labels_pred[link[0]] == labels_pred[link[1]])
        # cannot-ne
        elif link[-1] == -1:
            score.append(labels_pred[link[0]] != labels_pred[link[1]])

    return sum(score) / len(links)


### CHUNKLETS

def get_chunklets(links, n):

    anti_chunks = links[np.where(links[:, 2] == 0)[0]][:, :2] + 1 # +1 for matlab indexing

    must_links = links[np.where(links[:, 2] == 1)[0]][:, :2]
    chunks = _get_chunklets(must_links, n)

    return chunks, anti_chunks

def _get_chunklets(must_links, n):

    sets = []
    for link in must_links:

        if len(sets) == 0:
            sets.append(set(link))
        else:
            for i in xrange(len(sets)):

                if link[0] in sets[i] or link[1] in sets[i]:
                    sets[i] = sets[i] | set(link)
                    break
                if i == len(sets) - 1:
                    sets.append(set(link))

    change = True
    while change:
        change = False
        for i, j in combinations(range(len(sets)), 2):
            if len(sets[i] & sets[j]) > 0:
                sets[i] = sets[i] | sets[j]
                del sets[j]
                change = True
                break

    # check for errors
    for i, j in combinations(range(len(sets)), 2):
        assert len(sets[i] & sets[j]) == 0

    chunklets = -np.ones(n, dtype=int)
    for i, s in enumerate(sets):
        chunklets[list(s)] = i

    return _reindex(chunklets)


def _reindex(chunklets):
    # change to one-indexing for matlab
    return [c + 1 if c != -1 else c for c in chunklets]



