import numpy as np


def softmax(values, beta):
    """Compute softmax values for each sets of scores in x.
    
    Parameters
    ----------
    values : array-like
        Values to be softmaxed.
    beta : float
        Inverse temperature parameter.
    """

    values = np.array(values)

    sum_vals = np.exp(beta * values).sum()
    return np.array([np.exp(beta * value) / sum_vals for value in values])


class FrustratedRescorlaAgent:
    def __init__(self):
        # constants
        self.N_ARMS = 3
        self.V_SHIFT = (.25 + .25 + .7) / 3
        self.ALPHA = .1
        self.BETA = 50

        # last value and action
        self.value = self.V_SHIFT
        self.previous_action = np.random.choice([0, 1, 2])
        self.state = 'explore'  # explore or exploit
        
        # to be able to access past information (agent can't use this information)
        self._past_values = []
        self._past_actions = []
        self._past_states = []

    def update_value(self, reward):
        '''Update value based on reward using the Rescorla-Wagner rule.'''
        self.value = self.value + self.ALPHA * (reward - self.value)

    def update_state(self):
        '''Updates state based on current value. State can be exploration or exploitation.'''
        p_shift, p_stay = softmax([self.V_SHIFT, self.value], self.BETA)

        # select action (switch or stay) with probabilities p_shift and p_stay
        self.state = np.random.choice(['explore', 'exploit'], p=[p_shift, p_stay])

    def make_choice(self):
        '''Select action based on state. During exploraiton, the agent switches to another arm, while during exploitation it stays with the current arm.'''
        # if stay then stay with current action, otherwise switch to another arm (randomly)
        if self.state == 'exploit':
            action = self.previous_action
        elif self.state == 'explore':
            arms = np.arange(self.N_ARMS)
            action = np.random.choice(arms[arms != self.previous_action])

        return action

    def act(self, reward=None):
        '''Act based on reward. First update value, then update state and finally make a choice.'''
    
        # update value
        self.update_value(reward)

        # update state
        self.update_state()

        # make choice
        action = self.make_choice()

        # save value, action and state
        self.previous_action = action
        self._past_values.append(self.value)
        self._past_actions.append(action)
        self._past_states.append(self.state)

        return action
    

class HMM_DMaker:
    def __init__(self):
        # hmm params
        self.N_ARMS = 3
        self.transition_matrix = np.array[[39/40, 1/80, 1/80], [1/80, 39/40, 1/80], [1/80, 1/80, 39/40]]
        self.emission_matrix = ...
        self.posterior = np.ones(self.N_ARMS) / self.N_ARMS
        self.state = np.random.choice([0, 1, 2])
        self.action = np.random.choice([0, 1, 2])

        # for action selection
        strategy = 'softmax'  # softmax or argmax
        self.BETA = 50

        # to be able to access past information (agent can't use this information)
        self._past_values = []
        self._past_actions = []
        self._past_states = []
    
    def act(self):
        '''Select action based on state. During exploraiton, the agent switches to another arm, while during exploitation it stays with the current arm.'''
        # if stay then stay with current action, otherwise switch to another arm (randomly)
        if self.strategy == 'argmax':
            action = np.argmax(self.posterior)
        elif self.strategy == 'softmax':
            action = np.random.choice(np.arange(self.N_ARMS), p=softmax(self.posterior, self.BETA))

        return action
    
    def update_posterior(self, action, reward):
        '''Update posterior based on action and reward.'''
        # get likelihood
        likelihood = self.emission_matrix[action]

        # update posterior
        self.posterior = self.transition_matrix[action] * likelihood
        self.posterior /= self.posterior.sum()


class HiddenMarkovDecisionProcess:
    def __init__(self, n_arms=3, n_states=2, learning_rate=0.01, discount_factor=0.95):
        self.n_arms = n_arms
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Initialize transition probabilities
        self.transition_probs = np.ones((n_states, n_states)) / n_states

        # Initialize emission probabilities
        self.emission_probs = np.ones((n_states, n_arms)) / n_arms

        # Initialize state values
        self.state_values = np.zeros(n_states)

        # Initialize current belief state
        self.belief_state = np.ones(n_states) / n_states

    def update_belief_state(self, action, reward):
        # Update belief state based on action and reward
        emission_likelihoods = self.emission_probs[:, action] * (reward * 0.7 + (1 - reward) * 0.3)
        new_belief = self.belief_state @ self.transition_probs * emission_likelihoods
        self.belief_state = new_belief / new_belief.sum()

    def act(self, epsilon=0.1):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            return np.random.randint(self.n_arms)
        else:
            expected_rewards = self.belief_state @ self.emission_probs
            return np.argmax(expected_rewards)

    def update_model(self, action, reward):
        # Update transition probabilities
        new_transition_probs = self.transition_probs + self.learning_rate * (
            np.outer(self.belief_state, self.belief_state) - self.transition_probs
        )
        self.transition_probs = new_transition_probs / new_transition_probs.sum(axis=1, keepdims=True)

        # Update emission probabilities
        new_emission_probs = self.emission_probs.copy()
        new_emission_probs[:, action] += self.learning_rate * self.belief_state * (
            reward - self.emission_probs[:, action]
        )
        self.emission_probs = new_emission_probs / new_emission_probs.sum(axis=1, keepdims=True)

        # Update state values using TD learning
        expected_reward = self.emission_probs[:, action] @ self.belief_state
        td_error = reward + self.discount_factor * self.state_values @ self.belief_state - expected_reward
        self.state_values += self.learning_rate * td_error * self.belief_state

