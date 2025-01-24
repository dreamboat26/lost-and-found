# -------------------------
# The main body of this code was copied from https://github.com/Valdecy/Metaheuristic-Salp_Swarm_Algorithm/blob/master/Python-MH-Salp%20Swarm%20Algorithm.py
# Author: Farzad Roozitalab
# Github: Farzad-R
# -------------------------

import numpy as np
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt


class SalpSwarmAlgorithm:
    def __init__(
        self,
        upperbound,
        lowerbound,
        objective_function,
        robust_trial=1,
        swarm_size=50,
        iterations=100,
    ):
        self.robust_trial = robust_trial
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.nx = len(self.upperbound)
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.objective_function = objective_function

    def initial_position(
        self, swarm_size: int, lowerbound: np.ndarray, upperbound: np.ndarray
    ) -> np.ndarray:
        """
        This function creates the initial population (swarm) and calculates the objective for each single indivudual

        Returns:
        A numpy array containing the objective values
        """
        position = np.zeros((swarm_size, len(lowerbound) + 1))
        for i in range(0, swarm_size):
            for j in range(0, len(lowerbound)):
                position[i, j] = np.random.uniform(lowerbound[j], upperbound[j])
            position[i, -1] = self.objective_function(
                position[i, 0 : position.shape[1] - 1]
            )
        return position

    def food_position(self, dimension: int) -> List:
        """
        This function calculates the food position (objective of each individual in the population)
        Args: dimension or nx the introduced nx (number of variables) in the lecture
        Returns:
        A List array containing the food (the variables and their corresponding objective values)
        """
        food = np.zeros((1, dimension + 1))
        for j in range(0, dimension):
            food[0, j] = 0.0
        food[0, -1] = self.objective_function(food[0, 0 : food.shape[1] - 1])
        return food

    def update_food(self, position: np.ndarray, food: np.ndarray) -> np.ndarray:

        """
        This function updates food (variables and their corresponding objective value) based on their fitness evaluation
        """
        for i in range(0, position.shape[0]):
            if food[0, -1] > position[i, -1]:
                for j in range(0, position.shape[1]):
                    food[0, j] = position[i, j]
        return food

    # Function: Updtade Position
    def update_position(
        self,
        position: np.ndarray,
        food: np.ndarray,
        lowerbound: np.ndarray,
        upperbound: np.ndarray,
        c1: float,
    ) -> np.ndarray:
        """
        This function impelements equation 3.1 from the paper.
        Args:
        position: previous position
        food: populations and their corresponding objective value
        lowerbound: lower boundary
        upperbound: upper boundary
        Returns:
        The updated values of the position
        """
        for i in range(0, position.shape[0]):
            if i <= position.shape[0] / 2:
                for j in range(0, len(lowerbound)):
                    c2 = np.random.random()
                    c3 = np.random.random()
                    if c3 >= 0.5:  # c3 < 0.5
                        position[i, j] = np.clip(
                            (
                                food[0, j]
                                + c1
                                * ((upperbound[j] - lowerbound[j]) * c2 + lowerbound[j])
                            ),
                            lowerbound[j],
                            upperbound[j],
                        )
                    else:
                        position[i, j] = np.clip(
                            (
                                food[0, j]
                                - c1
                                * ((upperbound[j] - lowerbound[j]) * c2 + lowerbound[j])
                            ),
                            lowerbound[j],
                            upperbound[j],
                        )
            elif i > position.shape[0] / 2 and i < position.shape[0] + 1:
                for j in range(0, len(lowerbound)):
                    position[i, j] = np.clip(
                        ((position[i - 1, j] + position[i, j]) / 2),
                        lowerbound[j],
                        upperbound[j],
                    )
            position[i, -1] = self.objective_function(
                position[i, 0 : position.shape[1] - 1]
            )
        return position

    # SSA Function
    def run_SSA(self):
        best_pop_history = []
        obj_history = []
        count = 0
        position = self.initial_position(
            swarm_size=self.swarm_size,
            lowerbound=self.lowerbound,
            upperbound=self.upperbound,
        )
        food = self.food_position(dimension=len(self.lowerbound))
        while count <= self.iterations:
            # C1 is calculated from eq. 3.2 from the paper
            c1 = 2 * np.exp(-((4 * (count / self.iterations)) ** 2))
            food = self.update_food(position, food)
            position = self.update_position(
                position=position,
                food=food,
                c1=c1,
                lowerbound=self.lowerbound,
                upperbound=self.upperbound,
            )
            count += 1
            best_pop_history.append(food[0][0:3].flatten())
            obj_history.append(food[0][-1])
        df_pop_history = pd.DataFrame(best_pop_history, columns=["kp", "kd", "ki"])
        df_obj_history = pd.DataFrame(obj_history, columns=["best_obj"])

        return df_pop_history, df_obj_history
