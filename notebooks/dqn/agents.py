"""
This seems to be a collection of the implemented agents for the bandit task.
"""
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, FlattenObservation
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import scipy.special
from typing import Optional


### AGENTS ###

class ForagingAgent:
    def __init__(self, 
                 n_arms=3,
                 alpha=0.4,
                 beta=100,
                 V0=(.7 + .25 + .25) / 3,
                reset_on_switch=False,
                b1=0.0,
                b2=0.0,
                b3=0.0,
                abandoned_bias=0,
                abandoned_decay=0
                ):
        
        self.n_arms = n_arms

        self.alpha = alpha
        self.beta = beta
        self.V0 = V0
       
        self.reset_on_switch = reset_on_switch
       
        self.spatial_bias_0 = np.array([b1, b2, b3])
        if len(self.spatial_bias_0) != n_arms:
            raise ValueError("Length of spatial_bias must be equal to n_arms.")
        self.abandoned_bias = abandoned_bias
        self.abandoned_decay = abandoned_decay

        # internal variables
        self.V = V0
        self.last_action = np.random.choice(n_arms)
        self.spatial_bias = self.spatial_bias_0.copy()

    def reset(self):
        """Reset the agent to its initial state."""
        self.V = self.V0
        self.last_action = np.random.choice(self.n_arms)
        self.spatial_bias = self.spatial_bias_0

    def _get_stay_proba(self) -> float:   
        # get the probability of shifting actions, according to the logistic function 
        proba_shift = 1 / (1 + np.exp(self.beta * (self.V - self.V0)))
        proba_stay = 1 - proba_shift

        return proba_stay
    
    def _get_min_max_stay_proba(self) -> tuple:
        min_proba_stay = 1 / (1 + np.exp(self.beta * (self.V0 - 0)))
        max_proba_stay = 1 / (1 + np.exp(self.beta * (self.V0 - 1)))

        return min_proba_stay, max_proba_stay

    def get_action_probas(self) -> int:
        """
        Returns the probability of each action according to the Softmax transformation.
        """

        # get the probability of shifting actions, according to the logistic function 
        proba_shift = 1 / (1 + np.exp(self.beta * (self.V - self.V0)))
        proba_stay = 1 - proba_shift

        # get action probas
        probas = np.ones(self.n_arms) * (proba_shift / (self.n_arms - 1))
        probas[self.last_action] = proba_stay

        # add spatial bias (includes bias against abandoned target)
        probas = probas * np.exp(self.spatial_bias)
        probas = probas / np.sum(probas)

        # --- avoid zeros that break log likelihood later ---
        eps = np.finfo(float).tiny  # ~2.225e-308, safely > 0
        probas = np.clip(probas, eps, 1.0)
        probas = probas / probas.sum()  # re-normalize after flooring
        # -----------------------------------

        return probas
    
    def act(self) -> int:
        """
        Picks a random action probabilistically.
        """
    
        # Apply the Softmax transformation
        prob_a = self.get_action_probas()

        # Choose an action according to the probability distribution
        action = np.random.choice(self.n_arms, p=prob_a)

        return action
    
    def update_values(self, action: None, reward: float):
        """Updates the V-value of an action."""

        switched = action != self.last_action

        # if the agent switches, reset the V-value
        if self.reset_on_switch and switched:
            rpe = reward - self.V0
            self.V = self.V0 + self.alpha * rpe
        else:
            # use RPE to update the V-value after a stay, but use baseline after a switch
            rpe = reward - self.V
            self.V += self.alpha * rpe

        # set bias against the abandoned option (add to base level of bias)
        for a in range(self.n_arms):
            if switched and a == self.last_action:
                self.spatial_bias[a] += self.abandoned_decay * (self.abandoned_bias - self.spatial_bias[a])
            else:
                self.spatial_bias[a] += self.abandoned_decay * (self.spatial_bias_0[a] - self.spatial_bias[a])

        # store the last action
        self.last_action = action
