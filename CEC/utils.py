
from itertools import combinations_with_replacement as comb, product, combinations


def must_consistent(part):
    must = []
    for i in range(len(part)):
        must += list(comb(part[i], 2))

    must += [(j, i) for i, j in must if i != j]
    return must


def cannot_consistent(part):
    cannot = []
    for i, j in combinations(range(len(part)), 2):
        cannot += list(product(part[i], part[j]))

    cannot += [(j, i) for i, j in cannot]
    return cannot