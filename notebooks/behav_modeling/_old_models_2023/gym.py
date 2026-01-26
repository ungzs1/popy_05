import numpy as np
import pandas as pd


class SimpleBanditTask:
    """
    Multi armed bandit environmet.

    Parameters of the model:
    - N_TRIALS: number of trials
    - N_ARMS: number of arms
    - probabs: probability of reward for each arm
    - BLOCK_LEN: number of trials in each block (reward probabilities are fixed within each block).
    - BLOCK_TRANSITION_LEN: number of trials in each block's end when reward probabilities are shifting gradually.

    Consistency rules:
    - N_TRIALS mod BLOCK_LEN = 0

    """

    def __init__(self, N_TRIALS=10000, BLOCK_LEN=40, BLOCK_TRANSITION_LEN=5):
        self.N_TRIALS = N_TRIALS
        #self.N_ARMS = 3

        self.BLOCK_LEN = BLOCK_LEN
        self.BLOCK_TRANSITION_LEN = BLOCK_TRANSITION_LEN
       
        # shuffled reward probabilities
        self.probabs = np.array([.25, .25, .7])
        np.random.shuffle(self.probabs) 
        self.probab_transition = None
        self.update_transition_probabs()  # term of probabs transition
        self.best_arm = np.argmax(self.probabs)  # best arm in current block

        # current trial and block info
        self.trial_id = 0  # current trial (absolute measure)
        self.block_id = 0  # current block (absolute measure)
        self.trial_in_block = 0  # current trial in block (relative measure)
        #self.best_arm = np.random.choice(self.n_arms)  # best arm in current block
        #self.reward = None  # reward of current trial

        # history of choices and rewards
        self.rewards = np.zeros(self.N_TRIALS)-1
        self.choices = np.zeros(self.N_TRIALS)-1
        self.best_arms = np.zeros(self.N_TRIALS)-1
        self.block_ids = np.zeros(self.N_TRIALS)-1
        self.trial_in_blocks = np.zeros(self.N_TRIALS)-1


    def draw_reward(self, action):
        # get reward (binomial distribution)
        chosen_arm_probab = self.probabs[action]
        return np.random.binomial(1, chosen_arm_probab)

    def update_history(self, action, reward):
        # update history
        self.choices[self.trial_id] = action
        self.rewards[self.trial_id] = reward
        self.block_ids[self.trial_id] = self.block_id
        self.best_arms[self.trial_id] = self.best_arm
        self.trial_in_blocks[self.trial_id] = self.trial_in_block

    def update_transition_probabs(self):
        """Compute transition probabilities for each arm.

        Parameters
        ----------
        probabs : array-like
            Reward probabilities for each arm.
        """
        # get next probabs (different from current probabs)
        arms = np.arange(len(self.probabs))
        best_arm = np.argmax(self.probabs)
        other_arms = arms[arms != best_arm]
        new_best_arm = np.random.choice(other_arms)

        next_probabs = np.ones(len(self.probabs)) * .25
        next_probabs[new_best_arm] = .7

        self.probab_transition = (next_probabs - self.probabs) / self.BLOCK_TRANSITION_LEN

    def step(self, action):
        # on task end
        if self.trial_id == self.N_TRIALS:
            return -1
        
        # get reward
        reward = self.draw_reward(action)

        # update history
        self.update_history(action, reward)

        # update current trial and block info
        self.trial_id += 1
        self.trial_in_block += 1

        # update probabs during block transition
        if self.trial_in_block > self.BLOCK_LEN - self.BLOCK_TRANSITION_LEN:
            # gradually shift probabs
            self.probabs += self.probab_transition

        # on block end
        if self.trial_in_block == self.BLOCK_LEN:
            # update and reset block info
            self.trial_in_block = 0
            self.block_id += 1
            self.best_arm = np.argmax(self.probabs)

            # update probab transition term
            self.update_transition_probabs()

        # return reward
        return reward

    def extract_info(self):
        #  pandas with columns [trial_id, choice, reward, best_arm], and fill with data
        df = pd.DataFrame()
        df['trial_id'] = np.arange(self.N_TRIALS)
        df['block_id'] = self.block_ids
        df['trial_in_block'] = self.trial_in_blocks
        df['target'] = self.choices
        df['feedback'] = self.rewards
        df['best_target'] = self.best_arms
        return df