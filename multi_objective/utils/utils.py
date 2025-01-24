# -------------------------
# Author: Farzad Roozitalab
# -------------------------

"""
Author's note:
I wrote this class explicitly for 2 objective problems, since there are already good clustering techniques available that are easier to use
than these two techniques. I wrote this class for university presentation. So, if you want to use these techniques for more objective problems, 
you have to modify the codes.
"""

import numpy as np


class Clustering:
    """
    class Arguments:
     - n_set: int
     number of desiredd sets
     - nondominated_solutions_arr: np.ndarray
     Current pareto set
     type:: str
        "centroid"
        or
        "simple"

    class.run_culster():
    Returns:
    The new (modified) current pareto set
    """

    def __init__(self, n_set, nondominated_solutions_arr, type="centroid") -> None:
        self.n_set = n_set
        self.arr = nondominated_solutions_arr
        self.type = type

    def __compute_euc_between_points(self, arr):
        dict_euc_dist = {}
        for sol in arr:
            idx = np.unique(np.where(arr == sol)[0])[0]
            b = arr[np.unique(np.where(arr != sol)[0])]
            euclidean_list = []
            for j in b:
                euc_dist = np.sqrt(((sol[0] - j[0]) ** 2) + (sol[1] - j[1]) ** 2)
                euclidean_list.append(euc_dist)
                dict_euc_dist[idx] = euclidean_list
        return dict_euc_dist

    def __compute_euc_dist_2d_2list(self, arr_1, arr_2):
        dict_euc_dist = {}
        for sol in arr_1:
            idx = np.unique(np.where(arr_1 == sol)[0])[0]
            euclidean_list = []
            for j in arr_2:
                euc_dist = np.sqrt(((sol[0] - j[0]) ** 2) + (sol[1] - j[1]) ** 2)
                euclidean_list.append(euc_dist)
                dict_euc_dist[idx] = euclidean_list
        return dict_euc_dist

    def __find_biggest_cluster(self, dict_clusters):
        maxcount = max(len(v) for v in dict_clusters.values())
        key = [k for k, v in dict_clusters.items() if len(v) == maxcount]
        value = dict_clusters[key[0]]
        return key, value

    def __compute_euc_dist_between_center_and_cluster(self, center, values):
        dict_euc_dist = {}
        euclidean_list = []
        for j in values:
            euc_dist = np.sqrt(((center[0] - j[0]) ** 2) + (center[1] - j[1]) ** 2)
            euclidean_list.append(euc_dist)
        key_name = euclidean_list.index(min(euclidean_list))
        dict_euc_dist[key_name] = euclidean_list
        return dict_euc_dist

    def __apply_simple_clustering(self, arr, n_set):
        """For 2D objective function"""
        # Simple clustering
        # dict_euc_dist = {}
        while len(arr) > n_set:
            # dict_euc_dist = {}
            dict_euc_dist = self.__compute_euc_between_points(arr=arr)
            # creaing the cluster
            # first we find the indices of the two closest objectives
            # min key is the index of the first two closest solutions
            first_obj_list_idx = min(dict_euc_dist, key=dict_euc_dist.get)
            # min value is the distance of the two closest solutions (shortest ditance)
            min_euc_value = np.min(dict_euc_dist[first_obj_list_idx])
            # second_min is the index of the second two closest solutions
            second_obj_list_idx = dict_euc_dist[first_obj_list_idx].index(min_euc_value)
            # cluster_indices = [int(first_obj_list_idx), second_obj_list_idx]

            # Note: During each iteration, we jump from calculating the distance of each element with itself. ==>
            # In order to keep it in mind, for finding the corect location of second index, we have to remove the corresponding obj
            # to the first index
            first_selected_obj = arr[int(first_obj_list_idx)]
            tmp_arr = np.delete(arr=arr, obj=[int(first_obj_list_idx)], axis=0)
            second_selected_obj = tmp_arr[second_obj_list_idx]
            random_index = np.random.randint(2, size=1)
            if random_index == 0:
                selected_obj_idx_to_remove = np.where(arr == first_selected_obj)
            else:
                selected_obj_idx_to_remove = np.where(arr == second_selected_obj)
            updated_nondominated_sol = arr[
                np.unique(np.where(arr != arr[selected_obj_idx_to_remove])[0])
            ]
            arr = updated_nondominated_sol
        return arr

    def __apply_centroid_clustering(self, arr, n_set):
        n_pareto = len(arr)
        if n_pareto > n_set:
            if n_set + 1 == n_pareto:  # if only one cluster is required
                dict_clusters = {}
                dict_dist_between_solutions = self.__compute_euc_between_points(arr=arr)
                first_obj_list_idx = min(
                    dict_dist_between_solutions, key=dict_dist_between_solutions.get
                )
                min_euc_value = np.min(dict_dist_between_solutions[first_obj_list_idx])
                second_obj_list_idx = dict_dist_between_solutions[
                    first_obj_list_idx
                ].index(min_euc_value)
                key_mean_values = [i for i in dict_clusters.keys()]
                if not key_mean_values:
                    first_selected_obj = arr[first_obj_list_idx]
                    arr = np.delete(arr=arr, obj=[first_obj_list_idx], axis=0)
                    second_selected_obj = arr[second_obj_list_idx]
                    arr = np.delete(arr=arr, obj=[second_obj_list_idx], axis=0)
                    mean_point = np.mean(
                        [first_selected_obj, second_selected_obj], axis=0
                    )
                    key = str(mean_point[0]) + "_" + str(mean_point[1])
                    dict_clusters[key] = [first_selected_obj, second_selected_obj]
                for remaining_sol in arr:
                    idx = np.unique(np.where(arr == remaining_sol)[0])[0]
                    dict_clusters[str(idx)] = [remaining_sol]
                return dict_clusters

            else:  # if more than one cluster is required
                dict_clusters = {}
                dict_dist_between_solutions = self.__compute_euc_between_points(arr=arr)
                first_obj_list_idx = min(
                    dict_dist_between_solutions, key=dict_dist_between_solutions.get
                )
                min_euc_value = np.min(dict_dist_between_solutions[first_obj_list_idx])
                second_obj_list_idx = dict_dist_between_solutions[
                    first_obj_list_idx
                ].index(min_euc_value)
                key_mean_values = [i for i in dict_clusters.keys()]
                if not key_mean_values:
                    first_selected_obj = arr[first_obj_list_idx]
                    arr = np.delete(arr=arr, obj=[first_obj_list_idx], axis=0)
                    second_selected_obj = arr[second_obj_list_idx]
                    arr = np.delete(arr=arr, obj=[second_obj_list_idx], axis=0)
                    mean_point = np.mean(
                        [first_selected_obj, second_selected_obj], axis=0
                    )
                    key = str(mean_point[0]) + "_" + str(mean_point[1])
                    dict_clusters[key] = [first_selected_obj, second_selected_obj]

                key_mean_values = list(dict_clusters.keys())

                while len(key_mean_values) < n_set:
                    if (len(key_mean_values) + len(arr)) == n_set:
                        for sol in arr:
                            idx = np.unique(np.where(arr == sol)[0])[0]
                            dict_clusters[str(idx)] = [sol]
                    else:
                        # compute the distance between points and take the indices of the best two and their values
                        dict_dist_between_remaining_solutions = (
                            self.__compute_euc_between_points(arr=arr)
                        )
                        first_point_idx = min(
                            dict_dist_between_remaining_solutions,
                            key=dict_dist_between_remaining_solutions.get,
                        )
                        min_euc_value = np.min(
                            dict_dist_between_remaining_solutions[first_point_idx]
                        )
                        second_point_idx = dict_dist_between_remaining_solutions[
                            first_point_idx
                        ].index(min_euc_value)
                        # reconstruct the cluster means in a list [[x1,y1], [x2, y2], etc.]
                        list_of_cluster_means = []
                        for i in key_mean_values:
                            value_0 = float(i.split("_")[0])
                            value_2 = float(i.split("_")[1])
                            list_of_cluster_means.append([value_0, value_2])
                        dict_dist_between_clusters_and_solutions = (
                            self.__compute_euc_dist_2d_2list(
                                arr_1=np.array(list_of_cluster_means), arr_2=arr
                            )
                        )
                        cluster_idx = min(
                            dict_dist_between_clusters_and_solutions,
                            key=dict_dist_between_clusters_and_solutions.get,
                        )  # returns index of first value
                        cluster_min_euc_value = np.min(
                            dict_dist_between_clusters_and_solutions[cluster_idx]
                        )  # returns min distance
                        closest_point_to_cluster_idx = (
                            dict_dist_between_clusters_and_solutions[cluster_idx].index(
                                cluster_min_euc_value
                            )
                        )  # returns the index of second value
                        if (
                            cluster_min_euc_value <= min_euc_value
                        ):  # add to the old cluster
                            point_to_add = arr[closest_point_to_cluster_idx]
                            dict_clusters[key_mean_values[cluster_idx]].append(
                                point_to_add
                            )
                            arr = np.delete(
                                arr=arr, obj=[closest_point_to_cluster_idx], axis=0
                            )
                            mean_point = np.mean(
                                dict_clusters[key_mean_values[cluster_idx]], axis=0
                            )
                            replace_key = str(mean_point[0]) + "_" + str(mean_point[1])
                            dict_clusters[str(replace_key)] = dict_clusters.pop(
                                key_mean_values[cluster_idx]
                            )

                        else:  # create new cluster in the dict
                            first_selected_obj = arr[first_point_idx]
                            arr = np.delete(arr=arr, obj=[first_point_idx], axis=0)
                            second_selected_obj = arr[second_point_idx]
                            arr = np.delete(arr=arr, obj=[second_point_idx], axis=0)
                            mean_point = np.mean(
                                [first_selected_obj, second_selected_obj], axis=0
                            )
                            new_key = str(mean_point[0]) + "_" + str(mean_point[1])
                            dict_clusters[new_key] = [
                                first_selected_obj,
                                second_selected_obj,
                            ]

                    key_mean_values = list(dict_clusters.keys())
                return dict_clusters

    def __remove_solutions_in_dense_areas(self, arr, n_set):
        dict_clusters = self.__apply_centroid_clustering(arr=arr, n_set=n_set)
        while len(arr) > n_set:
            key, values = self.__find_biggest_cluster(dict_clusters=dict_clusters)
            for i in key:
                value_0 = float(i.split("_")[0])
                value_2 = float(i.split("_")[1])
                center = [value_0, value_2]

            dict_euc_dist = self.__compute_euc_dist_between_center_and_cluster(
                center=center, values=values
            )
            idx_of_solution_to_be_removed = list(dict_euc_dist.keys())[0]
            solution_to_be_removed = dict_clusters[key[0]][
                idx_of_solution_to_be_removed
            ]
            arr = arr[np.unique(np.where(arr != solution_to_be_removed)[0])]
            del dict_clusters[key[0]][idx_of_solution_to_be_removed]
        return arr

    def run_cluster(self):
        if self.type == "simple":
            new_arr = self.__apply_simple_clustering(arr=self.arr, n_set=self.n_set)
            return new_arr
        elif self.type == "centroid":
            new_arr = self.__remove_solutions_in_dense_areas(
                arr=self.arr, n_set=self.n_set
            )
            return new_arr
        else:
            print("The requested clustering technique does not exist in the class")


def fuzzy_selection(pareto_optimal_arr):
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
