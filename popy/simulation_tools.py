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


### ENVIRONMENTS ###

class ClassicalBanditTask(gym.Env):
    '''
    Excercise 2.5 from Sutton and Barto's book "Reinforcement Learning: An Introduction".
    '''
    def __init__(self, n_arms: int = 3, family: str = 'bernoulli'):
        self.n_arms = n_arms  # The size of the square grid
        self._block_length = 40  # The length of the current block
        self._trial_in_block_counter = 0  # The number of trials passed in the current block
        self._block_counter = 0  # The number of blocks passed in the current session

        self._family = family  # The family of the reward distribution, can be 'bernoulli' or 'normal'

        if self._family == 'bernoulli':
            self._reward_probas = np.ones(self.n_arms) / n_arms  # Define the reward probabilities for each arm (initially uniform)
        elif self._family == 'normal':
            self._reward_rates = np.random.normal(0, 1, n_arms)  # maybe its not necessary bcause we will update it in the reset method

        self.observation_space = gym.spaces.Discrete(1)  # Observations are not used in this environment
        self.action_space = gym.spaces.Discrete(self.n_arms)  # We have N actions, corresponding to the N arms

    def _get_info(self):
        if self._family == 'bernoulli':
            return {
                "block_id": self._block_counter,
                "trial_in_block_id": self._trial_in_block_counter,
                "reward_probas": self._reward_probas,
            }
        elif self._family == 'normal':
            return {
                "block_id": self._block_counter,
                "trial_in_block_id": self._trial_in_block_counter,
                "reward_rates": self._reward_rates,
                }
    
    def _update_reward_rates(self):
        if self._trial_in_block_counter == self._block_length - 1:
            if self._family == 'bernoulli':
                # randomize the reward probabilities
                self._reward_probas = np.random.dirichlet(np.ones(self.n_arms))
            elif self._family == 'normal':
                # Adjust the rates for the last trial of the block
                self._reward_rates = np.random.normal(0, 1, self.n_arms)
            self._trial_in_block_counter = 0
            self._block_counter += 1
        else:
            self._trial_in_block_counter += 1

    def _get_reward(self, action):
        # Sample the reward with variance 1
        if self._family == 'bernoulli':
            return np.random.choice([0, 1], p=[1 - self._reward_probas[action], self._reward_probas[action]])
        elif self._family == 'normal':
            return np.random.normal(self._reward_rates[action], 1)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._block_counter == 0
        self._trial_in_block_counter == 0

        # Reset the reward rates
        if self._family == 'bernoulli':
            self._reward_probas = np.ones(self.n_arms) / self.n_arms
        elif self._family == 'normal':
            self._reward_rates = np.random.normal(0, 1, self.n_arms)  # Define the reward rates for each arm

        observation = 0  # Observations are not used in this environment
        info = self._get_info()  # Get the info dictionary

        return observation, info

    def step(self, action):
        # An environment is completed if and only if the agent has reached the target
        terminated = False  # We don't terminate episodes in this environment, it can run forever
        truncated = False  # We don't truncate episodes in this environment, it can run forever
        observation = 0
        reward = self._get_reward(action)
        info = self._get_info()

        # Compute reward (only used for the last 5 trials of the block)
        self._update_reward_rates()

        return observation, reward, terminated, truncated, info


class MonkeyBanditTask(gym.Env):
    '''
    Implementation of the monkey bandit task of the project.
    '''

    def __init__(self, n_arms: int = 3):
        self.n_arms = n_arms  # The size of the square grid
        self._best_arm = 0  # Define the best arm
        self._next_best_arm = 0  # Define the next best arm
        self.rewards_in_block = np.zeros((self.n_arms, 40))  # Define the rewards in the block

        self.observation_space = gym.spaces.Discrete(1)  # Observations are not used in this environment
        self.action_space = gym.spaces.Discrete(self.n_arms)  # We have N actions, corresponding to the N arms

        self._trial_counter = 0  # The number of trials passed in the current session
        self._block_length = 40  # The length of the current block
        self._trial_in_block_counter = 0  # The number of trials passed in the current block
        self._block_counter = 0  # The number of blocks passed in the current session

        self.reset()

    def _get_info(self):
        return {
            "trial_id": self._trial_counter,
            "block_id": self._block_counter,
            #"trial_in_block_id": self._trial_in_block_counter,
            "best_arm": self._best_arm,
            #"block_length": self._block_length,
            #"reward_rates": self.rewards_in_block[:, self._trial_in_block_counter],
            }

    def _get_reward(self, action):
        # Sample the reward 
        return self.rewards_in_block[action, self._trial_in_block_counter]
    
    def _at_new_block(self):
        # Update the best arm, choose new next best arm, and reset the block length and trial counter
        self._best_arm = self._next_best_arm
        new_best_arm = np.random.randint(self.n_arms)
        while new_best_arm == self._best_arm:
            new_best_arm = np.random.randint(self.n_arms)
        self._next_best_arm = new_best_arm

        self._block_length = 40 + np.random.randint(-5, 6)
        self._trial_in_block_counter = 0
        self._block_counter += 1

        # Design rewards in block
        block_rewards_good_arm = np.zeros(self._block_length-5)
        n_rewarded = round((self._block_length-5) * .7)
        ids = np.random.choice(self._block_length-5, n_rewarded, replace=False)
        block_rewards_good_arm[ids] = 1

        block_rewards_bad_arm = np.zeros(self._block_length-5)
        n_rewarded = round((self._block_length-5) * .25)
        ids = np.random.choice(self._block_length-5, n_rewarded, replace=False)
        block_rewards_bad_arm[ids] = 1

        block_rewards_bad_arm_next = block_rewards_bad_arm.copy()
        block_rewards_bad_arm_not_next = block_rewards_bad_arm.copy()
        for rr in [0.625, 0.55 , 0.475, 0.4  , 0.325]:
            block_rewards_good_arm = np.append(block_rewards_good_arm, np.random.choice([1, 0], p=[rr, 1-rr]))
            block_rewards_bad_arm_next = np.append(block_rewards_bad_arm_next, np.random.choice([1, 0], p=[1-rr, rr]))
            block_rewards_bad_arm_not_next = np.append(block_rewards_bad_arm_not_next, np.random.choice([1, 0], p=[.25, .75]))

        self.rewards_in_block = np.array([block_rewards_bad_arm_not_next, block_rewards_bad_arm_not_next, block_rewards_bad_arm_not_next])
        self.rewards_in_block[self._best_arm] = block_rewards_good_arm
        self.rewards_in_block[self._next_best_arm] = block_rewards_bad_arm_next

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Set a new block, but reset the block counter
        self._at_new_block()
        self._block_counter = 0
        self._trial_counter = 0

        observation = 0
        info = self._get_info()

        return observation, info

    def step(self, action):
        # if new block, reset best arm, block length, and trial counter 
        if self._trial_in_block_counter >= self._block_length:
            self._at_new_block()

        # An environment is completed if and only if the agent has reached the target
        terminated = False  # We don't terminate episodes in this environment, it can run forever
        truncated = False  # We don't truncate episodes in this environment, it can run forever
        observation = 0
        reward = self._get_reward(action)
        info = self._get_info()
        
        # We need to track the number of steps 
        self._trial_in_block_counter += 1
        self._trial_counter += 1

        return observation, reward, terminated, truncated, info


class ChangeZeroRewardToNegativeOne(gym.RewardWrapper):
    def reward(self, reward):
        # Change reward from 0 to -1
        if reward == 0:
            return -1
        return reward
    

gym.register(
    id="zsombi/monkey-bandit-task-v0",
    entry_point=MonkeyBanditTask,
)

gym.register(
    id="zsombi/classical-bandit-task-v0",
    entry_point=ClassicalBanditTask,
)

### AGENTS ###


class RepeatingAgent:
    """
    An agent taht simply repeats the previous action.
    """
    def __init__(self, 
                 n_arms=3,
                 epsilon=0,
                ):
        
        self.n_arms = n_arms
        self.epsilon = epsilon

        self.last_action = np.random.choice(self.n_arms)
    
    def reset(self):
        self.last_action = np.random.choice(self.n_arms)

    def get_action_probas(self) -> np.ndarray:
        """
        Returns the probability of each action according to the Softmax transformation.
        """

        # if there was a reward in the last 3 trials, stay, i.e. set the probability of the previously chosen action to 1-epsilon + epsilon/n, and the rest to epsilon/n
        epsilon_per_n = self.epsilon / self.n_arms

        probas = np.ones(self.n_arms) * epsilon_per_n
        probas[self.last_action] = 1 - self.epsilon + epsilon_per_n

        return probas

    def act(self) -> int:
        """
        Picks a random action probabilistically.
        """
        # Apply the Softmax transformation
        probas = self.get_action_probas()

        # Choose an action according to the probability distribution
        return np.random.choice(self.n_arms, p=probas)
    
    def update_values(self, action: int, reward: float):
        """Updates the memory and last action of the agent."""
        # store the last action
        self.last_action = action


class WSLSAgent:
    """
    Simple WSLS agent, with softmax transformation.
    """
    def __init__(self, 
                 n_arms=3,
                 epsilon=0,
                ):
        
        self.n_arms = n_arms
        self.epsilon = epsilon

        self.last_action = np.random.choice(self.n_arms)
        self.last_outcome = 0  # memory of the last 2 trials
    
    def reset(self):
        self.last_action = np.random.choice(self.n_arms)
        self.last_outcome = 0

    def get_action_probas(self) -> np.ndarray:
        """
        Returns the probability of each action according to the Softmax transformation.
        """

        # if there was a reward in the last trials, stay, i.e. set the probability of the previously chosen action to 1-epsilon + epsilon/n, and the rest to epsilon/n
        epsilon_per_n = self.epsilon / self.n_arms
        if self.last_outcome == 1:
            probas = np.ones(self.n_arms) * epsilon_per_n
            probas[self.last_action] = 1 - self.epsilon + epsilon_per_n
        # set the probability of the previously chosen action to epsilon/n, and the rest to 1-epsilon/(n-1) --> epsilon/n + 2*[(1-epsilon)/(n-1)] = 1
        else:
            probas = np.ones(self.n_arms) * (((1 - self.epsilon)/(self.n_arms-1)) + epsilon_per_n)
            probas[self.last_action] = epsilon_per_n

        return probas

    def act(self) -> int:
        """
        Picks a random action probabilistically.
        """
        # Apply the Softmax transformation
        probas = self.get_action_probas()

        # Choose an action according to the probability distribution
        return np.random.choice(self.n_arms, p=probas)
    
    def update_values(self, action: int, reward: float):
        """Updates the memory and last action of the agent."""
        # update the memory
        self.last_outcome = reward

        # store the last action
        self.last_action = action


class WSLSAgent_custom:
    """
    An agent taht stays if there is a reward in the last 3 trials, and switches otherwise.
    """
    def __init__(self, 
                 n_arms=3,
                 epsilon=0,
                ):
        
        self.n_arms = n_arms
        self.epsilon = epsilon

        self.last_action = np.random.choice(self.n_arms)
        self.memory = [0, 0, 0]  # memory of the last 2 trials
    
    def reset(self):
        self.last_action = np.random.choice(self.n_arms)
        self.memory = [0, 0, 0]

    def get_action_probas(self) -> np.ndarray:
        """
        Returns the probability of each action according to the Softmax transformation.
        """

        # if there was a reward in the last 3 trials, stay, i.e. set the probability of the previously chosen action to 1-epsilon, and the rest to epsilon/n-1
        if sum(self.memory) > 0:  # stay
            probas = np.ones(self.n_arms) * ( self.epsilon / (self.n_arms - 1) )
            probas[self.last_action] = 1 - self.epsilon
        # set the probability of the previously chosen action to epsilon/n, and the rest to 1-epsilon/(n-1) --> epsilon/n + 2*[(1-epsilon)/(n-1)] = 1
        else:  # switch
            probas = np.ones(self.n_arms) * ((1 - self.epsilon)/(self.n_arms-1))
            probas[self.last_action] = self.epsilon

        return probas

    def act(self) -> int:
        """
        Picks a random action probabilistically.
        """
        # Apply the Softmax transformation
        probas = self.get_action_probas()

        # Choose an action according to the probability distribution
        return np.random.choice(self.n_arms, p=probas)
    
    def update_values(self, action: int, reward: float):
        """Updates the memory and last action of the agent."""
        # update the memory
        self.memory.pop(0)
        self.memory.append(reward)

        # store the last action
        self.last_action = action


class QLearner:
    def __init__(
        self,
        n_arms=3,
        alpha=.4,
        alpha_unchosen=None,
        beta=100,
        structure_aware=False,
        stickyness_bias=0.0,
        b1=0.0,
        b2=0.0,
        b3=0.0,
        forgetting_rate=0.0,
        forgetting_threshold=1/3,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of action values (q_values), a learning rate and an epsilon.
        """
        self.n_arms = n_arms
        
        self.alpha = alpha
        self.alpha_unchosen = alpha_unchosen if alpha_unchosen is not None else alpha
        self.beta = beta
        
        self.structure_aware = structure_aware
        
        self.stickyness_bias = float(stickyness_bias)
        self.spatial_bias = np.array([b1, b2, b3])
        if len(self.spatial_bias) != n_arms:
            raise ValueError("Length of spatial_bias must be equal to n_arms.")

        self.forgetting_rate = forgetting_rate
        self.forgetting_threshold = forgetting_threshold

        # internal variables
        self.q_values = np.ones(n_arms, dtype=np.float32) / n_arms  #np.zeros(n_arms, dtype=np.float32)  # 
        self.last_action = None  # will hold index of previous action

    def reset(self):
        self.q_values = np.zeros(self.n_arms, dtype=np.float32)
        self.last_action = None

    def get_action_probas(self) -> np.ndarray:
        """
        Returns the probability of each action according to the Softmax transformation.
        """
        # Stickiness bias: add extra bias for the last action
        bias_temp = self.spatial_bias.copy()
        if self.last_action is not None:
            bias_temp[self.last_action] += self.stickyness_bias

        # Apply the Softmax transformation to get probabilities of actions
        '''exp_Q = np.exp(self.beta * self.q_values + bias)
        return exp_Q / np.sum(exp_Q)'''
        # Softmax over logits = β * Q + bias  (use stable softmax)
        logits = self.beta * self.q_values + bias_temp
        z = logits - np.max(logits)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)
    
    def act(self) -> int:
        """
        Picks a random action probabilistically.
        """
        # Apply the Softmax transformation
        prob_a = self.get_action_probas()

        # Choose an action according to the probability distribution
        action = np.random.choice(self.n_arms, p=prob_a)
        return action

    @staticmethod
    def flip_reward(reward):
        """Encodes the reward."""
        return 0 if reward == 1 else 1

    def update_values(
        self,
        action: int,
        reward: float,
    ):
        """Updates the Q-value of an action."""
        # update selected arm
        rpe = reward - self.q_values[action]
        self.q_values[action] += self.alpha * rpe

        # update unchosen arms (if structure aware)
        if self.structure_aware:
            inverse_reward = self.flip_reward(reward)
            for a in range(self.n_arms):
                if not a == action:
                    rpe_unchosen = inverse_reward - self.q_values[a]
                    self.q_values[a] += self.alpha_unchosen * rpe_unchosen

        # forgetting towards uniform distribution (if forgetting rate > 0)
        if self.forgetting_rate > 0.0:
            for a in range(self.n_arms):
                if a != action:
                    self.q_values[a] += self.forgetting_rate * (self.forgetting_threshold - self.q_values[a])

        # store the last action
        self.last_action = action


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


class BayesianAgent:
    def __init__(
        self,
        n_arms=3,
        p_switch=None,  #1/40,
        transition_matrix = np.array([[39/40, .5/40, .5/40], [.5/40, 39/40, .5/40], [.5/40, .5/40, 39/40]]),
        emission_matrix_pos = np.array([[.7, .25, .25], [.25, .7, .25], [.25, .25, .7]]),
        emission_matrix_neg = np.array([[.3, .75, .75], [.75, .3, .75], [.75, .75, .3]]),
        beta=100,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of action values (q_values), a learning rate and an epsilon.
        """
        self.n_arms = n_arms

        if p_switch is not None:
            transition_matrix = np.array([[1 - p_switch, p_switch / 2, p_switch / 2],
                                          [p_switch / 2, 1 - p_switch, p_switch / 2],
                                          [p_switch / 2, p_switch / 2, 1 - p_switch]])
        self.transition_matrix = transition_matrix
        self.emission_matrix_pos = emission_matrix_pos
        self.emission_matrix_neg = emission_matrix_neg
        self.beta = beta
        
        self.posterior = np.ones(n_arms, dtype=np.float32) / n_arms


    def reset(self):
        self.posterior = np.ones(self.n_arms, dtype=np.float32) / self.n_arms

    def get_action_probas(self) -> int:
        """
        Returns the probability of each action according to the Softmax transformation.
        """
        # softmax action selection
        exp_Q = np.exp(self.beta * self.posterior)
        return exp_Q / np.sum(exp_Q)

    def act(self) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # Apply the Softmax transformation
        prob_a = self.get_action_probas()

        # Choose an action according to the probability distribution
        action = np.random.choice(self.n_arms, p=prob_a)
        return action

    def update_values(
        self,
        action: int,
        reward: float,
    ):
        """Updates th eposteriors of the agent (inference and state transition)."""
        # update the posterior
        self.posterior = self.posterior * self.emission_matrix_pos[action] if reward > 0 else self.posterior * self.emission_matrix_neg[action]
        self.posterior = self.posterior / np.sum(self.posterior)

        # update the transition matrix
        self.posterior = np.dot(self.posterior, self.transition_matrix)



