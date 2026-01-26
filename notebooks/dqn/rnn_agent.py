"""
RNN agent for the 3-armed bandit task using reinforcement learning.

The agent learns to decide whether to stay or switch based on reward feedback.
This version implements:
- Tiny RNN architecture (default 4 hidden units) for interpretability
- Actor-critic with value baseline for stable training
- Truncated BPTT for learning temporal dependencies
- Proper sampling using torch.distributions
- Device handling and reproducibility utilities
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
from collections import deque
import random
from typing import Optional, Tuple, Dict, List


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SwitchRNN(nn.Module):
    """
    Tiny RNN network for interpretable cognitive strategy modeling.
    
    Takes reward feedback and previous action as input, outputs:
    - Probability of switching on the next trial (actor)
    - Value estimate of current state (critic)
    
    Designed for interpretability:
    - Small hidden size (default 4) for trajectory visualization
    - Supports vanilla RNN or GRU
    - Optional weight regularization for simple dynamics
    """
    
    def __init__(
        self, 
        n_arms: int = 3,
        hidden_size: int = 4,
        rnn_type: str = 'GRU',
        use_action_input: bool = True,
        device: str = 'cpu'
    ):
        super(SwitchRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_arms = n_arms
        self.rnn_type = rnn_type
        self.use_action_input = use_action_input
        self.device = device
        
        # Input: [reward (1)] + optional [previous_action one-hot (n_arms)]
        self.input_size = 1 + (n_arms if use_action_input else 0)
        
        # RNN layer
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                nonlinearity='tanh'
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}. Choose 'RNN' or 'GRU'.")
        
        # Actor head: probability of switching
        self.fc_actor = nn.Linear(hidden_size, 1)
        
        # Critic head: value estimate
        self.fc_critic = nn.Linear(hidden_size, 1)
        
        self.to(device)
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None,
        return_all_hidden: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
        hidden : torch.Tensor, optional
            Hidden state from previous timestep
        return_all_hidden : bool
            If True, return hidden states at all timesteps
            
        Returns
        -------
        switch_prob : torch.Tensor
            Probability of switching (batch_size, 1)
        value : torch.Tensor
            Value estimate (batch_size, 1)
        hidden : torch.Tensor
            Updated hidden state (num_layers, batch_size, hidden_size)
        hidden_all : torch.Tensor, optional
            Hidden states at all timesteps (batch_size, seq_len, hidden_size)
            Only returned if return_all_hidden=True
        """
        # Pass through RNN
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Take the output from the last timestep
        rnn_out_last = rnn_out[:, -1, :]
        
        # Compute actor (switch probability) and critic (value)
        switch_logit = self.fc_actor(rnn_out_last)
        switch_prob = torch.sigmoid(switch_logit)
        
        value = self.fc_critic(rnn_out_last)
        
        if return_all_hidden:
            return switch_prob, value, hidden, rnn_out
        else:
            return switch_prob, value, hidden, None
    
    def forward_sequence(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass that returns outputs at all timesteps for analysis.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
        hidden : torch.Tensor, optional
            Initial hidden state
            
        Returns
        -------
        outputs : dict
            Dictionary containing:
            - 'switch_probs': (batch_size, seq_len, 1)
            - 'values': (batch_size, seq_len, 1)
            - 'hidden_states': (batch_size, seq_len, hidden_size)
            - 'final_hidden': (num_layers, batch_size, hidden_size)
        """
        # Pass through RNN
        rnn_out, final_hidden = self.rnn(x, hidden)
        
        # Compute outputs at all timesteps
        batch_size, seq_len, _ = rnn_out.shape
        
        # Reshape to (batch_size * seq_len, hidden_size) for efficient linear layers
        rnn_out_flat = rnn_out.reshape(-1, self.hidden_size)
        
        switch_logits = self.fc_actor(rnn_out_flat)
        switch_probs = torch.sigmoid(switch_logits).reshape(batch_size, seq_len, 1)
        
        values = self.fc_critic(rnn_out_flat).reshape(batch_size, seq_len, 1)
        
        return {
            'switch_probs': switch_probs,
            'values': values,
            'hidden_states': rnn_out,
            'final_hidden': final_hidden
        }
    
    def reset_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Reset the hidden state."""
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


class RNNAgent:
    """
    RL agent that uses a tiny RNN to learn the stay/switch decision.
    
    Improvements over original:
    - Actor-critic architecture with value baseline
    - Truncated BPTT for learning temporal dependencies
    - Proper sampling using torch.distributions with entropy bonus
    - Device handling and reproducibility
    - Configurable RNN architecture
    
    The agent:
    1. Receives binary reward feedback (0 or 1)
    2. Uses RNN to output probability of switching (actor) and value estimate (critic)
    3. Samples action from Bernoulli distribution (no external noise)
    4. If switch: chooses random alternative action
    5. If stay: repeats previous action
    6. Learns using actor-critic with entropy regularization
    """
    
    def __init__(
        self,
        n_arms: int = 3,
        hidden_size: int = 4,
        rnn_type: str = 'GRU',
        use_action_input: bool = True,
        learning_rate: float = 0.001,
        gamma: float = 1.0,  # discount factor (1.0 for bandits)
        entropy_coef: float = 0.01,  # entropy bonus coefficient
        value_coef: float = 0.5,  # value loss coefficient
        feedback_duration: int = 1,  # timesteps to present feedback
        decision_delay: int = 0,  # delay between feedback and decision
        bptt_truncation: int = 20,  # truncate BPTT every K trials
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        self.n_arms = n_arms
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.feedback_duration = feedback_duration
        self.decision_delay = decision_delay
        self.bptt_truncation = bptt_truncation
        self.device = device
        
        if seed is not None:
            set_seed(seed)
        
        # Initialize the RNN
        self.network = SwitchRNN(
            n_arms=n_arms,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            use_action_input=use_action_input,
            device=device
        )
        
        # Optimizer (separate from network)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Initialize state
        self.last_action = np.random.choice(n_arms)
        self.hidden = self.network.reset_hidden(batch_size=1)
        
        # Buffers for truncated BPTT
        self.trial_inputs = []  # list of input tensors
        self.trial_log_probs = []  # list of log probabilities
        self.trial_entropies = []  # list of entropies
        self.trial_values = []  # list of value estimates
        self.trial_rewards = []  # list of rewards
        self.trial_count = 0  # count trials since last update
        
        # For monitoring
        self.episode_rewards = []
        self.episode_losses = []
    
    def reset(self):
        """Reset the agent to initial state."""
        self.last_action = np.random.choice(self.n_arms)
        self.hidden = self.network.reset_hidden(batch_size=1)
        self.trial_inputs = []
        self.trial_log_probs = []
        self.trial_entropies = []
        self.trial_values = []
        self.trial_rewards = []
        self.trial_count = 0
    
    def _make_input(self, reward: float) -> torch.Tensor:
        """
        Construct network input from reward and previous action.
        
        Parameters
        ----------
        reward : float
            Binary reward (0 or 1)
            
        Returns
        -------
        network_input : torch.Tensor
            Input tensor of shape (1, 1, input_size)
        """
        if self.network.use_action_input:
            # One-hot encode previous action
            action_onehot = torch.zeros(self.n_arms, device=self.device)
            action_onehot[self.last_action] = 1.0
            # Concatenate reward and action
            network_input = torch.cat([
                torch.tensor([reward], device=self.device),
                action_onehot
            ])
        else:
            network_input = torch.tensor([reward], device=self.device)
        
        # Add batch and sequence dimensions
        return network_input.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)
    
    def act(
        self, 
        reward: float, 
        return_hidden_states: bool = False
    ) -> Tuple[int, bool, Optional[np.ndarray]]:
        """
        Decide whether to stay or switch based on the reward feedback.
        
        Parameters
        ----------
        reward : float
            Binary reward (0 or 1) from the previous action
        return_hidden_states : bool
            If True, return hidden states at each timestep
            
        Returns
        -------
        action : int
            The chosen action (0 to n_arms-1)
        switched : bool
            Whether the agent switched from the previous action
        hidden_states : np.ndarray, optional
            Hidden states over time (total_timesteps, hidden_size)
            Only returned if return_hidden_states=True
        """
        # Prepare input for the network
        network_input = self._make_input(reward)
        
        # Repeat input for feedback_duration timesteps to simulate temporal dynamics
        network_input = network_input.repeat(1, self.feedback_duration, 1)  # (1, feedback_duration, input_size)
        
        # Add decision delay
        if self.decision_delay > 0:
            delay_input = torch.zeros(1, self.decision_delay, network_input.shape[2], device=self.device)
            network_input = torch.cat([network_input, delay_input], dim=1)
        
        # Store input for BPTT
        self.trial_inputs.append(network_input)
        
        # Forward pass through network
        if return_hidden_states:
            # Use forward_sequence to get all hidden states
            outputs = self.network.forward_sequence(network_input, self.hidden)
            switch_prob = outputs['switch_probs'][:, -1, :]  # (1, 1)
            value = outputs['values'][:, -1, :]  # (1, 1)
            self.hidden = outputs['final_hidden']
            hidden_states = outputs['hidden_states'].squeeze(0).detach().cpu().numpy()  # (seq_len, hidden_size)
        else:
            switch_prob, value, self.hidden, _ = self.network(network_input, self.hidden)
            hidden_states = None
        
        # Sample action using Bernoulli distribution (no external noise)
        dist = Bernoulli(probs=switch_prob)
        should_switch_tensor = dist.sample()
        should_switch = bool(should_switch_tensor.item())
        
        # Store log probability, entropy, and value for training
        log_prob = dist.log_prob(should_switch_tensor)
        entropy = dist.entropy()
        
        self.trial_log_probs.append(log_prob)
        self.trial_entropies.append(entropy)
        self.trial_values.append(value)
        
        # Execute action
        if should_switch:
            # Switch to a random alternative action
            alternatives = [a for a in range(self.n_arms) if a != self.last_action]
            action = np.random.choice(alternatives)
        else:
            # Stay with previous action
            action = self.last_action
        
        self.last_action = action
        self.trial_count += 1
        
        if return_hidden_states:
            return action, should_switch, hidden_states
        else:
            return action, should_switch
    
    def store_reward(self, reward: float):
        """Store reward for training."""
        self.trial_rewards.append(reward)
    
    def should_update(self) -> bool:
        """Check if it's time to update (reached truncation boundary)."""
        return self.trial_count >= self.bptt_truncation
    
    def update(self, force: bool = False) -> Dict[str, float]:
        """
        Update the network using actor-critic with truncated BPTT.
        
        This should be called:
        - Every `bptt_truncation` trials (automatic if using should_update())
        - At the end of an episode with force=True
        
        Parameters
        ----------
        force : bool
            If True, update even if haven't reached truncation boundary
            
        Returns
        -------
        metrics : dict
            Dictionary with 'loss', 'policy_loss', 'value_loss', 'entropy'
        """
        if len(self.trial_rewards) == 0:
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        if not force and not self.should_update():
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(self.trial_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Stack values, log_probs, entropies
        values = torch.cat(self.trial_values, dim=0)  # (n_trials, 1)
        log_probs = torch.cat(self.trial_log_probs, dim=0)  # (n_trials, 1)
        entropies = torch.cat(self.trial_entropies, dim=0)  # (n_trials, 1)
        
        # Compute advantages (i.e., do baseline subtraction???)
        advantages = returns.unsqueeze(1) - values.detach()  # (n_trials, 1)
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor loss: policy gradient with advantage
        policy_loss = -(log_probs * advantages).mean()
        
        # Critic loss: MSE between value predictions and returns
        value_loss = ((values - returns.unsqueeze(1)) ** 2).mean()
        
        # Entropy bonus for exploration
        entropy_bonus = entropies.mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Store episode metrics
        episode_reward = sum(self.trial_rewards)
        self.episode_rewards.append(episode_reward)
        self.episode_losses.append(loss.item())
        
        # Clear buffers and detach hidden state at truncation boundary
        self.trial_inputs = []
        self.trial_log_probs = []
        self.trial_entropies = []
        self.trial_values = []
        self.trial_rewards = []
        self.trial_count = 0
        self.hidden = self.hidden.detach()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_bonus.item()
        }
    
    def get_switch_probability(self, reward: float) -> float:
        """
        Get the switch probability without taking an action (for analysis).
        
        Parameters
        ----------
        reward : float
            Binary reward (0 or 1)
            
        Returns
        -------
        switch_prob : float
            Probability of switching
        """
        with torch.no_grad():
            network_input = self._make_input(reward)
            network_input = network_input.repeat(1, self.feedback_duration, 1)
            
            if self.decision_delay > 0:
                delay_input = torch.zeros(1, self.decision_delay, network_input.shape[2], device=self.device)
                network_input = torch.cat([network_input, delay_input], dim=1)
            
            switch_prob, _, _, _ = self.network(network_input, self.hidden)
            return switch_prob.item()
    
    def save(self, filepath: str):
        """Save model and optimizer state."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
        }, filepath)
    
    def load(self, filepath: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_losses = checkpoint.get('episode_losses', [])


class RNNAgentRecorder:
    """
    Helper class to record RNN agent behavior and internal states during simulation.
    """
    
    def __init__(self):
        self.recording = []
    
    def record(
        self, 
        action: int, 
        reward: float, 
        info: dict, 
        agent: RNNAgent, 
        switched: bool, 
        switch_prob: Optional[float] = None
    ):
        """
        Record the environment and agent states.
        
        Parameters
        ----------
        action : int
            The action taken by the agent
        reward : float
            The reward received
        info : dict
            Environment information
        agent : RNNAgent
            The agent
        switched : bool
            Whether the agent switched actions
        switch_prob : float, optional
            The switch probability output by the network
        """
        trial_behav = {
            "action": action,
            "reward": reward,
            "switched": int(switched),
        }
        
        if switch_prob is not None:
            trial_behav["switch_prob"] = switch_prob
        
        # Save environment info
        for key, value in info.items():
            trial_behav[key] = value.copy() if isinstance(value, np.ndarray) else value
        
        self.recording.append(trial_behav)
    
    def get_recording(self):
        """Get the recording as a pandas DataFrame."""
        import pandas as pd
        recordings_df = pd.DataFrame(self.recording)
        
        # Reorder columns
        first_cols = ["trial_id", "block_id", "best_arm", "action", "reward", "switched", "switch_prob"]
        other_cols = [col for col in recordings_df.columns if col not in first_cols]
        first_cols = [col for col in first_cols if col in recordings_df.columns]
        recordings_df = recordings_df[first_cols + other_cols]
        
        return recordings_df


# ==================== Analysis Utilities ====================

class RNNAnalyzer:
    """
    Analysis tools for interpreting tiny RNN dynamics and discovered strategies.
    
    Provides:
    - Fixed-point analysis
    - Trajectory visualization
    - Behavioral strategy characterization (P(switch|reward), etc.)
    - Vector field visualization for 2D hidden spaces
    """
    
    def __init__(self, agent: RNNAgent):
        self.agent = agent
        self.network = agent.network
        self.device = agent.device
    
    def compute_behavioral_stats(self, recording_df) -> Dict[str, float]:
        """
        Compute behavioral statistics from recorded trials.
        
        Parameters
        ----------
        recording_df : pd.DataFrame
            Recording from RNNAgentRecorder
            
        Returns
        -------
        stats : dict
            Behavioral statistics including:
            - p_stay_given_reward: P(stay | reward=1)
            - p_switch_given_no_reward: P(switch | reward=0)
            - p_stay_given_no_reward: P(stay | reward=0)
            - switch_rate: overall switch rate
            - reward_rate: overall reward rate
        """
        import pandas as pd
        
        # Shift reward to align with next trial's decision
        recording_df = recording_df.copy()
        recording_df['prev_reward'] = recording_df['reward'].shift(1)
        recording_df = recording_df.dropna()
        
        rewarded_trials = recording_df[recording_df['prev_reward'] == 1]
        unrewarded_trials = recording_df[recording_df['prev_reward'] == 0]
        
        stats = {
            'p_stay_given_reward': 1 - rewarded_trials['switched'].mean() if len(rewarded_trials) > 0 else np.nan,
            'p_switch_given_no_reward': unrewarded_trials['switched'].mean() if len(unrewarded_trials) > 0 else np.nan,
            'p_stay_given_no_reward': 1 - unrewarded_trials['switched'].mean() if len(unrewarded_trials) > 0 else np.nan,
            'switch_rate': recording_df['switched'].mean(),
            'reward_rate': recording_df['reward'].mean(),
        }
        
        return stats
    
    def find_fixed_points(
        self, 
        reward: float = 0.0, 
        action: int = 0,
        n_inits: int = 10,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> List[np.ndarray]:
        """
        Find fixed points of the RNN dynamics for a given input condition.
        
        For vanilla RNN: h* = tanh(W_hh @ h* + W_ih @ u + b)
        For GRU: more complex, but we can still find fixed points numerically.
        
        Parameters
        ----------
        reward : float
            Input reward value
        action : int
            Input action (arm index)
        n_inits : int
            Number of random initializations
        max_iter : int
            Maximum iterations for fixed-point search
        tol : float
            Convergence tolerance
            
        Returns
        -------
        fixed_points : list of np.ndarray
            List of unique fixed points found
        """
        # Prepare constant input
        if self.network.use_action_input:
            action_onehot = torch.zeros(self.agent.n_arms, device=self.device)
            action_onehot[action] = 1.0
            u = torch.cat([torch.tensor([reward], device=self.device), action_onehot])
        else:
            u = torch.tensor([reward], device=self.device)
        
        u = u.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)
        
        fixed_points = []
        
        for _ in range(n_inits):
            # Random initialization
            h = torch.randn(1, 1, self.network.hidden_size, device=self.device) * 0.5
            
            with torch.no_grad():
                for it in range(max_iter):
                    h_old = h.clone()
                    
                    # One step of RNN dynamics
                    _, h = self.network.rnn(u, h)
                    
                    # Check convergence
                    delta = torch.norm(h - h_old).item()
                    if delta < tol:
                        # Found a fixed point
                        fp = h.squeeze().cpu().numpy()
                        
                        # Check if it's a new fixed point (not duplicate)
                        is_new = True
                        for existing_fp in fixed_points:
                            if np.linalg.norm(fp - existing_fp) < tol * 10:
                                is_new = False
                                break
                        
                        if is_new:
                            fixed_points.append(fp)
                        break
        
        return fixed_points
    
    def compute_jacobian_at_fixed_point(
        self, 
        fixed_point: np.ndarray, 
        reward: float = 0.0, 
        action: int = 0
    ) -> np.ndarray:
        """
        Compute the Jacobian of the RNN dynamics at a fixed point.
        
        Eigenvalues of the Jacobian tell us about stability:
        - All eigenvalues with |λ| < 1: stable attractor
        - Some eigenvalues with |λ| > 1: saddle point or unstable
        
        Parameters
        ----------
        fixed_point : np.ndarray
            Hidden state fixed point (hidden_size,)
        reward : float
            Input reward value
        action : int
            Input action (arm index)
            
        Returns
        -------
        jacobian : np.ndarray
            Jacobian matrix (hidden_size, hidden_size)
        """
        # Prepare input
        if self.network.use_action_input:
            action_onehot = torch.zeros(self.agent.n_arms, device=self.device)
            action_onehot[action] = 1.0
            u = torch.cat([torch.tensor([reward], device=self.device), action_onehot])
        else:
            u = torch.tensor([reward], device=self.device)
        
        u = u.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)
        
        # Convert fixed point to tensor
        h = torch.tensor(fixed_point, dtype=torch.float32, device=self.device)
        h = h.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
        h.requires_grad = True
        
        # Forward pass
        _, h_next = self.network.rnn(u, h)
        h_next = h_next.squeeze()  # (hidden_size,)
        
        # Compute Jacobian using autograd
        jacobian = []
        for i in range(self.network.hidden_size):
            # Gradient of h_next[i] with respect to h
            grad = torch.autograd.grad(
                h_next[i], 
                h, 
                retain_graph=True, 
                create_graph=False
            )[0]
            jacobian.append(grad.squeeze().detach().cpu().numpy())
        
        jacobian = np.array(jacobian)  # (hidden_size, hidden_size)
        
        return jacobian
    
    def plot_vector_field_2d(
        self, 
        reward: float = 0.0, 
        action: int = 0,
        xlim: Tuple[float, float] = (-2, 2),
        ylim: Tuple[float, float] = (-2, 2),
        n_grid: int = 20,
        ax=None
    ):
        """
        Plot vector field of RNN dynamics in 2D hidden space.
        
        Only works if hidden_size == 2.
        
        Parameters
        ----------
        reward : float
            Input reward value
        action : int
            Input action
        xlim : tuple
            X-axis limits
        ylim : tuple
            Y-axis limits
        n_grid : int
            Number of grid points per dimension
        ax : matplotlib axis, optional
            Axis to plot on
        """
        import matplotlib.pyplot as plt
        
        if self.network.hidden_size != 2:
            raise ValueError("Vector field plot only works for hidden_size=2")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create grid
        x = np.linspace(xlim[0], xlim[1], n_grid)
        y = np.linspace(ylim[0], ylim[1], n_grid)
        X, Y = np.meshgrid(x, y)
        
        # Compute vector field
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        # Prepare input
        if self.network.use_action_input:
            action_onehot = torch.zeros(self.agent.n_arms, device=self.device)
            action_onehot[action] = 1.0
            u = torch.cat([torch.tensor([reward], device=self.device), action_onehot])
        else:
            u = torch.tensor([reward], device=self.device)
        
        u = u.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)
        
        with torch.no_grad():
            for i in range(n_grid):
                for j in range(n_grid):
                    h = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32, device=self.device)
                    h = h.unsqueeze(0)  # (1, 1, 2)
                    
                    _, h_next = self.network.rnn(u, h)
                    h_next = h_next.squeeze().cpu().numpy()
                    
                    # Vector field is h_next - h
                    U[i, j] = h_next[0] - X[i, j]
                    V[i, j] = h_next[1] - Y[i, j]
        
        # Plot
        ax.quiver(X, Y, U, V, alpha=0.6)
        ax.set_xlabel('Hidden unit 1')
        ax.set_ylabel('Hidden unit 2')
        ax.set_title(f'Vector field (reward={reward}, action={action})')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_trajectory(
        self, 
        hidden_states: np.ndarray,
        ax=None,
        **kwargs
    ):
        """
        Plot trajectory of hidden states over time.
        
        For 2D hidden states, plots in 2D space.
        For higher dimensions, plots first 2 PCs.
        
        Parameters
        ----------
        hidden_states : np.ndarray
            Hidden states over time (n_timesteps, hidden_size)
        ax : matplotlib axis, optional
            Axis to plot on
        **kwargs : dict
            Additional arguments for plotting
        """
        import matplotlib.pyplot as plt
        
        if self.network.hidden_size == 2:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 8))
            
            ax.plot(hidden_states[:, 0], hidden_states[:, 1], **kwargs)
            ax.scatter(hidden_states[0, 0], hidden_states[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
            ax.scatter(hidden_states[-1, 0], hidden_states[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
            ax.set_xlabel('Hidden unit 1')
            ax.set_ylabel('Hidden unit 2')
            ax.set_title('Hidden state trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # Use PCA to project to 2D
            from sklearn.decomposition import PCA
            
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 8))
            
            pca = PCA(n_components=2)
            hidden_2d = pca.fit_transform(hidden_states)
            
            ax.plot(hidden_2d[:, 0], hidden_2d[:, 1], **kwargs)
            ax.scatter(hidden_2d[0, 0], hidden_2d[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
            ax.scatter(hidden_2d[-1, 0], hidden_2d[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
            ax.set_title('Hidden state trajectory (PCA projection)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        return ax


def plot_learning_curves(agent: RNNAgent, window: int = 50):
    """
    Plot learning curves: reward and loss over episodes.
    
    Parameters
    ----------
    agent : RNNAgent
        Trained agent
    window : int
        Moving average window size
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reward curve
    if len(agent.episode_rewards) > 0:
        rewards_df = pd.DataFrame({'episode': range(len(agent.episode_rewards)), 
                                   'reward': agent.episode_rewards})
        rewards_df['reward_ma'] = rewards_df['reward'].rolling(window=window, min_periods=1).mean()
        
        axes[0].plot(rewards_df['episode'], rewards_df['reward'], alpha=0.3, label='Raw')
        axes[0].plot(rewards_df['episode'], rewards_df['reward_ma'], linewidth=2, label=f'MA({window})')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Learning Curve: Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Loss curve
    if len(agent.episode_losses) > 0:
        losses_df = pd.DataFrame({'episode': range(len(agent.episode_losses)), 
                                  'loss': agent.episode_losses})
        losses_df['loss_ma'] = losses_df['loss'].rolling(window=window, min_periods=1).mean()
        
        axes[1].plot(losses_df['episode'], losses_df['loss'], alpha=0.3, label='Raw')
        axes[1].plot(losses_df['episode'], losses_df['loss_ma'], linewidth=2, label=f'MA({window})')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Learning Curve: Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_strategy_analysis(recording_df):
    """
    Plot behavioral strategy analysis: P(switch|reward), P(stay|reward), etc.
    
    Parameters
    ----------
    recording_df : pd.DataFrame
        Recording from RNNAgentRecorder
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Shift reward to align with next trial's decision
    df = recording_df.copy()
    df['prev_reward'] = df['reward'].shift(1)
    df = df.dropna()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # P(switch | previous reward)
    switch_given_reward = df.groupby('prev_reward')['switched'].agg(['mean', 'sem'])
    axes[0].bar([0, 1], switch_given_reward['mean'], yerr=switch_given_reward['sem'], 
                capsize=5, alpha=0.7)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['No reward', 'Reward'])
    axes[0].set_ylabel('P(switch)')
    axes[0].set_title('Switch probability by outcome')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Switch rate over time
    df['trial_bin'] = pd.cut(df.index, bins=10, labels=False)
    switch_over_time = df.groupby('trial_bin')['switched'].mean()
    axes[1].plot(switch_over_time.index, switch_over_time.values, marker='o')
    axes[1].set_xlabel('Trial bin')
    axes[1].set_ylabel('Switch rate')
    axes[1].set_title('Switch rate over time')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    # Reward rate by action
    reward_by_action = df.groupby('action')['reward'].agg(['mean', 'sem', 'count'])
    axes[2].bar(reward_by_action.index, reward_by_action['mean'], 
                yerr=reward_by_action['sem'], capsize=5, alpha=0.7)
    axes[2].set_xlabel('Action (arm)')
    axes[2].set_ylabel('Reward rate')
    axes[2].set_title('Reward rate by arm')
    axes[2].set_ylim([0, 1])
    
    # Add count labels
    for i, (idx, row) in enumerate(reward_by_action.iterrows()):
        axes[2].text(i, row['mean'] + 0.05, f"n={int(row['count'])}", 
                    ha='center', va='bottom', fontsize=9)
    
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig
