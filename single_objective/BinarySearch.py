# -------------------------
# Author: Farzad Roozitalab
# Github: Farzad-R
# -------------------------

import numpy as np
from typing import List, Tuple
import pandas as pd


class BinarySearch:
    def __init__(
        self,
        upperbound,
        lowerbound,
        objective_function,
        robust_trial=1,
        epsilon=0.001,
        population_size=10,
        generation_size=100,
    ):
        self.objective_function = objective_function
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.num_parameters = len(
            self.upperbound
        )  # number of parameters in one candidate solution (x1, x2, ..., x6)
        self.robust_trial = robust_trial
        self.epsilon = epsilon
        self.population_size = population_size
        self.generation_size = generation_size
        self.stop_condition = 0
        self.obj = None
        self.best_solution = None

    def __generate_population(
        self, pop_size: int, min_values: np.ndarray, max_values: np.ndarray
    ) -> np.ndarray:
        """
        Returns: A numpy array containing the objective values
        """
        population = np.zeros((pop_size, len(min_values)))
        for i in range(pop_size):
            for j in range(len(min_values)):
                population[i, j] = np.random.randint(min_values[j], max_values[j] + 1)
        return population

    def __evaluate_fitness(self, population):
        for i in range(self.population_size):
            obj = self.objective_function(population[i])
            if obj <= self.epsilon:
                self.stop_condition = 1
                return obj, population[i]
            else:
                return None, None

    def run_BS(self):
        # for k in range(self.robust_trial):
        for j in range(self.generation_size):
            population = self.__generate_population(
                pop_size=self.population_size,
                min_values=self.lowerbound,
                max_values=self.upperbound,
            )

            self.obj, self.best_solution = self.__evaluate_fitness(
                population=population
            )
            if self.stop_condition == 1:
                break

        print("Best generation:", j)
        return self.obj, self.best_solution
