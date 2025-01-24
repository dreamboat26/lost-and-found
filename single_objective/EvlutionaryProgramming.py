# -------------------------
# Author: Farzad Roozitalab
# Github: Farzad-R
# -------------------------
from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class EvlutionaryProgramming:
    def __init__(
        self,
        objective_function,
        upperbound,
        lowerbound,
        robust_trial=1,
        population_size=30,
        generation_size=100,
        beta=0.1,  # Crossover probability
        tournament_coeff=5,  # 1 <= tournament_coeff  < 2*population_size
    ):
        self.objective_function = objective_function
        self.robust_trial = robust_trial
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.num_parameters = len(self.upperbound)
        self.population_size = population_size
        self.generation_size = generation_size
        self.beta = beta
        self.tournament_coeff = tournament_coeff
        self.q = 2 * self.population_size - self.tournament_coeff  # q < (2*nP-1)

        self.best_obj_history: List = None
        self.best_solution_history: List = None
        self.df_obj_hist: pd.DataFrame = None
        self.df_solution_hist: pd.DataFrame = None

    def __generate_population(
        self, pop_size: int, min_values: np.ndarray, max_values: np.ndarray
    ) -> np.ndarray:
        """
        Returns: A numpy array containing the objective values
        """
        population = np.zeros((pop_size, len(min_values)))
        for i in range(pop_size):
            for j in range(len(min_values)):
                population[i, j] = np.random.uniform(min_values[j], max_values[j])
        return population

    def __evaluate_population_fitness(
        self, population_arr: np.ndarray
    ) -> Tuple[List, List]:
        obj_list = []
        for i in range(len(population_arr)):
            obj_list.append(self.objective_function(population_arr[i]))
        return obj_list

    def __apply_mutation(
        self, population, obj_list: List, obj_max: float, j: int
    ) -> np.ndarray:
        """
        In this step: Each parent generates an offspring.
        Returns:
        2nP size population (parents + their offsprings)
        """
        for i in range(0, self.population_size):
            temp_list = []
            for r in range(0, self.num_parameters):
                sigma = np.abs(
                    self.beta
                    * (obj_list[i] / obj_max)
                    * (self.upperbound[r] - self.lowerbound[r])
                )
                temp_list.append(population[i, r] + sigma * np.random.normal(0, 1, 1))
            if j == 0:
                population = np.append(
                    population, np.array(temp_list).reshape(1, 3), axis=0
                )
            else:
                population[i + self.population_size] = np.array(temp_list).flatten()
        return population

    def __check_boundary(self, population: np.ndarray) -> np.ndarray:
        """
        numpy.clip(a, a_min, a_max, out=None, **kwargs)[source]
        """
        return np.clip(population, self.lowerbound, self.upperbound, out=population)

    def __apply_tournament(self, obj_list: List, population: np.ndarray):
        """
        The first nP individuals with lower weights will be selected along with theri objective functions to represent
        the parents of the next generation
        """
        df_W = pd.DataFrame(columns=["w"])
        for i in range(len(population)):
            df_W.loc[i] = 1
            for m in range(self.q):
                RAND_K = np.random.permutation(len(population))
                d = RAND_K[0]
                if d == i:
                    if d < (2 * self.population_size - 1):
                        d = i + 1
                    else:
                        d = i - 1
                if np.random.uniform(0, 1, 1) > obj_list[i] / (
                    obj_list[i] + obj_list[d]
                ):
                    df_W.loc[i] = df_W.loc[i][0] + 1

        counter = 0
        x_Rank = population
        # This piece of code has 15 years of jail.
        for jj in range(self.q, -1, -1):
            for i in range(1, len(population)):
                if df_W.loc[i][0] == jj:
                    counter += 1
                    x_Rank[counter] = population[i]
        parents_of_new_generation = x_Rank
        return parents_of_new_generation, df_W

    def run_EP(self):
        # for k in range(0, self.robust_trial):
        # To keep track of the desired records
        self.best_obj_history = []
        self.best_solution_history = []

        population = self.__generate_population(
            pop_size=self.population_size,
            min_values=self.lowerbound,
            max_values=self.upperbound,
        )
        obj_list = self.__evaluate_population_fitness(population_arr=population)

        obj_min = min(obj_list)
        obj_max = max(obj_list)
        best_solution = population[obj_list.index(min(obj_list))]

        self.best_obj_history.append(obj_min)
        self.best_solution_history.append(best_solution)

        for j in range(self.generation_size):
            double_size_population = self.__apply_mutation(
                population=population, obj_list=obj_list, obj_max=obj_max, j=j
            )
            self.__check_boundary(population=double_size_population)
            obj_list = self.__evaluate_population_fitness(
                population_arr=double_size_population
            )

            obj_min = min(obj_list)
            obj_max = max(obj_list)
            best_solution = population[obj_list.index(min(obj_list))]
            # print(len(double_size_population))
            # print(len(obj_list))
            new_population, df_W = self.__apply_tournament(
                obj_list=obj_list, population=double_size_population
            )
            population = new_population

            if min(obj_list) < min(self.best_obj_history):
                self.best_obj_history.append(obj_min)
                self.best_solution_history.append(best_solution)
            else:
                self.best_obj_history.insert(
                    len(self.best_obj_history), self.best_obj_history[-1]
                )
                self.best_solution_history.insert(
                    len(self.best_solution_history), self.best_solution_history[-1]
                )
        self.df_obj_hist = pd.DataFrame(self.best_obj_history, columns=["obj_hist"])
        self.df_solution_hist = pd.DataFrame(self.best_solution_history)

        return self.df_obj_hist, self.df_solution_hist
