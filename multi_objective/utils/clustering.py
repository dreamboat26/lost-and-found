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
    This class contains two clustering techniques called simple and centroid.
    In multi-obj problems if the number of non-dominated solutions are higher than the defined n_set,
    the extra solutions have to be removed from the non-dominated olutions. For this purpose clustering is used.
    Different clustering methods can be used for this purpose. Here, I wrote two methods that I couldn't find anywhere.
    In simple clustering:
        - Step 1: Compute all possible distances between all elements.
        - Step 2: Take the two closest non-dominated solutions and consider them as cluster one.
        - Step 3: Remove one of them (randomly)
        - Step 4: Check whether number -f solutions == n_set
        - If not, go to step 1
        - If yes, return the modified non-dominated solutions set

    The problem of this clustering method is that it cannot keep the diversity of the solutions (which is important for us).

    Therefore, we apply the centroid clustering:
    This method is developed based on Metropolis et al. (1953).

    - Step 0: Check if number of non-dominated solutions is not < n_set -> got to step 1
    - Step 1: Compute all possible distances between all elements.
    - Step 2: Take the two closest non-dominated solutions and consider them as cluster one.
    - Step 3: Compute the mean solution in the new cluster
    - Step 4: Check num_cluster + number of remaining solutions
        - if == n_set - > go to step 8
        - else: go to step 5
    - Step 5: Repeat step 1 but this one conside the mean solution that were calculated in step 3 as one of the solutions
    - Step 6: Two scenario can happen
        - 1: The next two closest solutions are one solution with the mean value of the previous cluster:
            -> add that solution to the previous cluster
        - 2: The next two closest solutions are two differnt solutions (mean value of previous clusters not involved)
            -> Create a new cluster
    - Step 7: Go to step 3
    - Step 8: take each remaining solution and create a separate cluster
        - Up to this point we should have number of clusters == n_set
    - Step 9: While the number of solutions > n_set
        - Take the biggest cluster
        - Take the closest solution to that cluster
        - Remove that from the non-dominated solutions
    - Step 10: Return the modified non-dominated solutions


    Arguments:

     - n_set: int
     number of desiredd sets
     - nondominated_solutions_arr: np.ndarray
     Current pareto set
     type:: str
        "centroid"
        or
        "simple"

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
