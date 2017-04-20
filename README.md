### Code supporting  "Flexible semi-supervised clustering with pairwise constraints reproduction" submission for ECML PKDD 2017

## Quick run
1. Install requirements listed in `requirement.txt`,
2. [OPTIONAL] Set `RESULTS_DIR` in the `.env` file if you want to run experiment codes,
3. Use `python cluster.py` to run the clustering.

## `Cluster.py` help

```
usage: cluster.py [-h] [-k K] --data DATA --links LINKS
                  [--result_file RESULT_FILE] [--n_tries N_TRIES]
                  [--verbose VERBOSE]

Run CEC model /w link constraints

optional arguments:
  -h, --help            show this help message and exit
  -k K                  Number of clusters
  --data DATA, -D DATA  Path to dataset stored in CSv format
  --links LINKS, -L LINKS
                        Path to labels stored in CSV format
  --result_file RESULT_FILE, -R RESULT_FILE
                        Path to file in which to store results
  --n_tries N_TRIES, -N N_TRIES
                        Number of times to run the algorithm
  --verbose VERBOSE, -V VERBOSE
                        Verbosity true/false
```
