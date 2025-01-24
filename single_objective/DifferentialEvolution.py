# -------------------------
# Author: Farzad Roozitalab
# Github: Farzad-R
# -------------------------

from typing import List, Tuple
import pandas as pd
import numpy as np


class DifferentialEvolution:
    def __init__(
        self,
        objective_function,
        upperbound,
        lowerbound,
        robust_trial=1,
        population_size=50,
        generation_size=100,
        mutation_factor=0.5,  # controls the speed of convergence
        crossover_rate=0.5,  # controls the solution convergence
    ) -> None:
        self.robust_trial = robust_trial  # to test robustness of the approach
        self.objective_function = objective_function
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.num_parameters = len(self.upperbound)
        self.population_size = population_size
        self.generation_size = generation_size  # number of generations (we chose it)
        self.mf = mutation_factor
        self.cr = crossover_rate
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

    def __apply_mutation(
        self,
        parent_population: np.ndarray,
        best_solution_arr: np.ndarray,
    ) -> np.ndarray:
        """
        This function applies the mutation formula to the parent population and generates the mutated population
        Args:
        population: The parent population
        best_solution_arr: is the best solution (kd, kp, ki) of the parent population
        Returns:
        mutated_population: the mutated population
        """
        # generating a copy of the previous population
        mutated_population = parent_population.copy()
        for i in range(self.population_size):
            # Extracting the best population's values
            two_random_numbers = np.random.randint(0, self.population_size, 2)
            x_r1, x_r2 = (
                parent_population[two_random_numbers[0]],
                parent_population[two_random_numbers[1]],
            )
            if not np.array_equal(x_r1, x_r2):
                # Applying one of the given formulas in the lecture and generating a mutated population
                mutated_population[i] = best_solution_arr + self.mf * (x_r1 - x_r2)
            else:
                x_r1, x_r2 = (
                    parent_population[two_random_numbers[0]],
                    parent_population[two_random_numbers[1]],
                )
                mutated_population[i] = best_solution_arr + self.mf * (x_r1 - x_r2)

        return mutated_population

    def __apply_crossover(
        self, parent_population: np.ndarray, mutated_population: np.ndarray
    ) -> np.ndarray:
        """
        This function applies crossover to generate the trial population

        Args:
        parent_population
        mutated_population

        Returns:
        trial_population_arr
        """
        for i in range(self.population_size):
            trial_vector = []
            for r in range(self.num_parameters):
                random_value = np.random.random()
                if random_value > self.cr:
                    trial_vector.append(parent_population[i][r])
                else:
                    trial_vector.append(mutated_population[i][r])
            if i == 0:
                trial_population_arr = np.array([trial_vector])
            else:
                trial_population_arr = np.concatenate(
                    (trial_population_arr, np.array([trial_vector])), axis=0
                )

        return trial_population_arr

    def __get_the_bestsolution_array(
        self, obj_list: List, population: np.ndarray
    ) -> np.ndarray:
        """
        Here we calculate the best solution of the population to use it in the mutation formula
        """
        min_obj_index = obj_list.index(min(obj_list))
        best_solution_arr = population[min_obj_index]
        return best_solution_arr

    def __select_the_better_population(
        self, parent_population: np.ndarray, trial_population: np.ndarray
    ) -> np.ndarray:
        """
        Here we have two sets of population ready: parent_population & trial_population
        we will compare them and svae the best values as the new population
        """
        selected_population = []
        for i in range(self.population_size):
            trial_obj = self.objective_function(trial_population[i])
            parent_obj = self.objective_function(parent_population[i])
            if trial_obj < parent_obj:
                selected_population.append(trial_population[i])
            else:
                selected_population.append(parent_population[i])
        return np.array(selected_population)

    def run_DE(self):
        # for r in range(self.robust_trial):
        # To keep track of the desired records
        self.best_obj_history = []
        self.best_solution_history = []

        population = self.__generate_population(
            pop_size=self.population_size,
            min_values=self.lowerbound,
            max_values=self.upperbound,
        )
        obj_list = self.__evaluate_population_fitness(population_arr=population)
        best_solution_arr = self.__get_the_bestsolution_array(
            obj_list=obj_list, population=population
        )
        self.best_obj_history.append(min(obj_list))
        self.best_solution_history.append(population[obj_list.index(min(obj_list))])
        for j in range(self.generation_size):
            parent_population = population
            mutated_population = self.__apply_mutation(
                parent_population=parent_population,
                best_solution_arr=best_solution_arr,
            )
            self.__check_boundary(population=mutated_population)
            trial_population_arr = self.__apply_crossover(
                parent_population=parent_population,
                mutated_population=mutated_population,
            )
            self.__check_boundary(population=trial_population_arr)

            population = self.__select_the_better_population(
                parent_population=parent_population,
                trial_population=trial_population_arr,
            )
            # evaluate the fitness of the new population
            obj_list = self.__evaluate_population_fitness(population_arr=population)
            # get the new bestsolution
            best_solution_arr = self.__get_the_bestsolution_array(
                obj_list=obj_list, population=population
            )
            if min(obj_list) < min(self.best_obj_history):
                self.best_obj_history.append(min(obj_list))
                self.best_solution_history.append(
                    population[obj_list.index(min(obj_list))]
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
