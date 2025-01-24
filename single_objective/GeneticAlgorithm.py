# -------------------------
# Author: Farzad Roozitalab
# Github: Farzad-R
# -------------------------

import numpy as np
from typing import List, Tuple
import pandas as pd


class GeneticAlgorithm:
    def __init__(
        self,
        upperbound,
        lowerbound,
        objective_function,
        robust_trial=1,
        population_size=10,
        generation_size=100,
        crossover_probability=0.7,
        mutation_probability=0.3,
        B=2,
        lambda_coeff=0.2,
    ):
        self.robust_trial = robust_trial  # to test the robustness of the approach
        self.objective_function = objective_function
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.num_parameters = len(
            self.upperbound
        )  # number of parameters in one candidate solution (x1, x2, ..., x6)
        self.population_size = population_size
        self.generation_size = generation_size
        self.Pc = crossover_probability
        self.Pm = mutation_probability
        self.B = B
        self.lambda_coeff = lambda_coeff
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

    def __check_boundary(self, population: np.ndarray) -> np.ndarray:
        """
        numpy.clip(a, a_min, a_max, out=None, **kwargs)[source]
        """
        return np.clip(population, self.lowerbound, self.upperbound, out=population)

    def __apply_arithmetical_crossover(
        self, current_population: np.ndarray, previous_population: np.ndarray
    ) -> np.ndarray:
        for i in range(0, 2, self.population_size):
            if self.Pc > np.random.uniform(0, 1):
                for r in range(1, self.num_parameters):
                    current_population[i - 1, r] = (
                        self.lambda_coeff * previous_population[i - 1, r]
                        + (1 - self.lambda_coeff) * previous_population[i, r]
                    )
                    current_population[i, r] = (
                        self.lambda_coeff * previous_population[i, r]
                        + (1 - self.lambda_coeff) * previous_population[i - 1, r]
                    )
        return current_population

    def __apply_non_uniform_mutation(
        self, population: np.ndarray, generation_iter: int
    ) -> np.ndarray:
        for i in range(self.population_size):
            if self.Pm > np.random.uniform(0, 1):
                for r in range(0, self.num_parameters):
                    population[i, r] = population[i, r] - (
                        population[i, r] - self.lowerbound[r]
                    ) * (
                        1
                        - np.random.uniform(0, 1)
                        ** (1 - (generation_iter / self.generation_size) ** self.B)
                    )
            else:
                for r in range(0, self.num_parameters):
                    population[i, r] = population[i, r] + (
                        self.upperbound[r] - population[i, r]
                    ) * (
                        1
                        - np.random.uniform(0, 1)
                        ** (1 - (generation_iter / self.generation_size) ** self.B)
                    )
        return population

    def run_GA(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Note: In case of using robust trial, change the output to the desired format.

        Returns: -> Tuple[pd.DataFrame, pd.DataFrame]

        df_obj_hist: History of best objectives
        df_solution_hist: History of best solutions
        """

        # for r in range(self.robust_trial):
        # To keep track of the desired records
        self.best_obj_history = []
        self.best_solution_history = []

        current_population = self.__generate_population(
            pop_size=self.population_size,
            min_values=self.lowerbound,
            max_values=self.upperbound,
        )
        obj_list = self.__evaluate_population_fitness(current_population)

        self.best_obj_history.append(min(obj_list))
        self.best_solution_history.append(
            current_population[obj_list.index(min(obj_list))]
        )

        for j in range(self.generation_size):
            previous_population = current_population

            new_population = self.__apply_arithmetical_crossover(
                current_population=current_population,
                previous_population=previous_population,
            )
            self.__check_boundary(population=new_population)
            new_population = self.__apply_non_uniform_mutation(
                population=new_population, generation_iter=j
            )
            self.__check_boundary(population=new_population)
            obj_list = self.__evaluate_population_fitness(new_population)

            current_population = new_population

            if min(obj_list) < min(self.best_obj_history):
                self.best_obj_history.append(min(obj_list))
                self.best_solution_history.append(
                    current_population[obj_list.index(min(obj_list))]
                )
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
