import argparse
import os
import numpy as np
np.set_printoptions(threshold=np.nan)

from CEC import CEC

parser = argparse.ArgumentParser(description='Run CEC model /w link constraints')

parser.add_argument('-k',  type=int,
                            required=False,
                            default=5,
                            help='Number of clusters')

parser.add_argument('--data', '-D', type=str,
                                    required=True,
                                    help='Path to dataset stored in CSv format')

parser.add_argument('--links', '-L', type=str,
                                     required=True,
                                     help='Path to labels stored in CSV format')

parser.add_argument('--result_file', '-R', type=str,
                                           required=False,
                                           default=None,
                                           help='Path to file in which to store results')

parser.add_argument('--n_tries', '-N', type=int,
                            required=False,
                            default=1,
                            help='Number of times to run the algorithm')

parser.add_argument('--verbose', '-V', type=bool,
                                       required=False,
                                       default=True,
                                       help='Verbosity true/false')

args = parser.parse_args()

X = np.loadtxt(args.data, delimiter=',')
links = np.loadtxt(args.links, delimiter=',').astype(int)

# run CEC
cec = CEC(k=args.k,
          n_init=args.n_tries,
          verbose=args.verbose)

p = cec.fit_predict(X, links)

print "Cluster assignment:"
print p

if args.result_file is not None:
    np.savetxt(args.result_file, p, fmt="%d", delimiter=',')


