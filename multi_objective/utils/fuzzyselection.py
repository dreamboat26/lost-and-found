# -------------------------
# Author: Farzad Roozitalab
# -------------------------
from typing import List
import numpy as np

"""
Author's note:
I wrote this class explicitly for 2 objective problems. I will modify to be independent from the number of objectives.
So, currently if you want to use these techniques for more objective problems, you have to modify the codes.
"""


def fuzzy_selection(pareto_optimal_arr: np.ndarray) -> List:
    """This function returns the best compromised solution within a pareto array.
    Arguments:
    pareto_optimal_arr = The pareto array
    Returns:
    The best compromised solution
    """
    first_obj = pareto_optimal_arr[:, 0]
    second_obj = pareto_optimal_arr[:, 1]
    diff_first = np.max(first_obj) - np.min(first_obj)
    diff_second = np.max(first_obj) - np.min(first_obj)
    total_sum = np.sum(pareto_optimal_arr.flatten())
    membership_func = []
    for sol in pareto_optimal_arr:
        tmp = []
        if sol[0] <= np.min(first_obj):
            tmp.append(1)
        elif sol[0] >= np.max(first_obj):
            tmp.append(0)
        else:
            value = (np.max(first_obj) - sol[0]) / diff_first
            tmp.append(value)

        if sol[1] <= np.min(second_obj):
            tmp.append(1)
        elif sol[1] >= np.max(second_obj):
            tmp.append(0)
        else:
            value = (np.max(second_obj) - sol[1]) / diff_second
            tmp.append(value)
        membership_func.append(sum(tmp) / total_sum)

    idx_max = membership_func.index(max(membership_func))
    return pareto_optimal_arr[idx_max]
