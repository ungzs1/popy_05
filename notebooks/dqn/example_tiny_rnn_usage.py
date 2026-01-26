"""
Example usage of the improved tiny RNN agent for 3-armed bandit task.

This script demonstrates:
1. Training the agent with actor-critic and truncated BPTT
2. Analysis of learned strategies (fixed points, trajectories)
3. Behavioral characterization (P(switch|reward), etc.)
4. Visualization of 2D hidden dynamics

Run this to see the improvements in action!
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from notebooks.dqn.rnn_agent import (
    RNNAgent, RNNAgentRecorder, RNNAnalyzer,
    plot_learning_curves, plot_strategy_analysis, set_seed
)


# ==================== Simple Bandit Environment ====================

class ThreeArmedBandit:
    """
    Simple 3-armed bandit environment for testing.
    One arm is the best (highest reward probability).
    """
    
    def __init__(self, reward_probs=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        if reward_probs is None:
            # Default: arm 0 is best (80% reward), others are worse (20%)
            self.reward_probs = np.array([0.8, 0.2, 0.2])
        else:
            self.reward_probs = np.array(reward_probs)
        
        self.n_arms = len(self.reward_probs)
        self.trial_id = 0
        self.best_arm = np.argmax(self.reward_probs)
    
    def reset(self):
        """Reset environment."""
        self.trial_id = 0
        return self._get_info()
    
    def step(self, action):
        """
        Take an action and return reward.
        
        Returns
        -------
        reward : int
            Binary reward (0 or 1)
        info : dict
            Trial information
        """
        # Sample reward
        reward = 1 if np.random.random() < self.reward_probs[action] else 0
        
        self.trial_id += 1
        
        info = self._get_info()
        
        return reward, info
    
    def _get_info(self):
        return {
            'trial_id': self.trial_id,
            'best_arm': self.best_arm,
            'reward_probs': self.reward_probs.copy()
        }


# ==================== Training Function ====================

def train_agent(
    agent,
    env,
    n_trials=1000,
    verbose=True,
    record=True
):
    """
    Train the agent on the bandit environment.
    
    Parameters
    ----------
    agent : RNNAgent
        Agent to train
    env : ThreeArmedBandit
        Environment
    n_trials : int
        Number of trials to run
    verbose : bool
        Print progress
    record : bool
        Record behavior for analysis
        
    Returns
    -------
    recorder : RNNAgentRecorder or None
        Recorder with trial-by-trial data if record=True
    """
    if record:
        recorder = RNNAgentRecorder()
    else:
        recorder = None
    
    agent.reset()
    info = env.reset()
    
    # Initial random action
    action = agent.last_action
    
    for trial in range(n_trials):
        # Get reward from environment
        reward, info = env.step(action)
        
        # Agent decides next action
        action, switched = agent.act(reward)
        
        # Store reward for training
        agent.store_reward(reward)
        
        # Record if requested
        if record:
            switch_prob = agent.get_switch_probability(reward)
            recorder.record(action, reward, info, agent, switched, switch_prob)
        
        # Update agent if at truncation boundary
        if agent.should_update():
            metrics = agent.update()
            if verbose and trial % 100 == 0:
                print(f"Trial {trial}/{n_trials}, Loss: {metrics['loss']:.4f}, "
                      f"Policy: {metrics['policy_loss']:.4f}, "
                      f"Value: {metrics['value_loss']:.4f}, "
                      f"Entropy: {metrics['entropy']:.4f}")
    
    # Final update if any trials remaining
    agent.update(force=True)
    
    if verbose:
        print(f"\nTraining complete! Total episodes: {len(agent.episode_rewards)}")
        if len(agent.episode_rewards) > 0:
            print(f"Mean reward (last 50 episodes): {np.mean(agent.episode_rewards[-50:]):.2f}")
    
    return recorder


# ==================== Main Example ====================

def main():
    """Run the full example."""
    
    print("=" * 60)
    print("Tiny RNN Agent Example: 3-Armed Bandit")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create environment
    print("\n1. Creating environment...")
    env = ThreeArmedBandit(reward_probs=[0.8, 0.2, 0.2], seed=42)
    print(f"   Best arm: {env.best_arm} (reward prob: {env.reward_probs[env.best_arm]:.1%})")
    
    # Create agent with tiny RNN (4 hidden units for interpretability)
    print("\n2. Creating tiny RNN agent...")
    agent = RNNAgent(
        n_arms=3,
        hidden_size=4,  # Tiny for interpretability!
        rnn_type='GRU',
        use_action_input=True,
        learning_rate=0.001,
        gamma=1.0,  # No discounting for bandits
        entropy_coef=0.01,
        value_coef=0.5,
        bptt_truncation=20,
        device='cpu',
        seed=42
    )
    print(f"   Hidden size: {agent.network.hidden_size}")
    print(f"   RNN type: {agent.network.rnn_type}")
    print(f"   Input size: {agent.network.input_size}")
    
    # Train agent
    print("\n3. Training agent...")
    recorder = train_agent(agent, env, n_trials=200_000, verbose=True, record=True)
    
    # Get recording
    recording_df = recorder.get_recording()
    
    # Plot learning curves
    print("\n4. Plotting learning curves...")
    fig_learning = plot_learning_curves(agent, window=50)
    plt.savefig('//Users/zsombi/Library/CloudStorage/GoogleDrive-uuungvarszi@gmail.com/Other computers/My Mac/ZSOMBI/SBRI/PoPy/notebooks/dqn/tmp/learning_curves.png', dpi=150, bbox_inches='tight')
    print("   Saved to: /tmp/learning_curves.png")
    
    # Plot strategy analysis
    print("\n5. Analyzing behavioral strategy...")
    fig_strategy = plot_strategy_analysis(recording_df)
    plt.savefig('//Users/zsombi/Library/CloudStorage/GoogleDrive-uuungvarszi@gmail.com/Other computers/My Mac/ZSOMBI/SBRI/PoPy/notebooks/dqn/tmp/strategy_analysis.png', dpi=150, bbox_inches='tight')
    print("   Saved to: /tmp/strategy_analysis.png")
    
    # Compute behavioral stats
    analyzer = RNNAnalyzer(agent)
    stats = analyzer.compute_behavioral_stats(recording_df)
    print("\n   Behavioral statistics:")
    print(f"   - P(stay | reward):        {stats['p_stay_given_reward']:.2%}")
    print(f"   - P(switch | no reward):   {stats['p_switch_given_no_reward']:.2%}")
    print(f"   - Overall switch rate:     {stats['switch_rate']:.2%}")
    print(f"   - Overall reward rate:     {stats['reward_rate']:.2%}")
    
    # Find fixed points
    print("\n6. Analyzing RNN dynamics (fixed points)...")
    for reward_val in [0.0, 1.0]:
        for action_val in range(3):
            fps = analyzer.find_fixed_points(reward=reward_val, action=action_val, n_inits=5)
            print(f"   Input: reward={reward_val}, action={action_val} -> {len(fps)} fixed point(s)")
            
            for i, fp in enumerate(fps):
                # Compute Jacobian and eigenvalues
                jac = analyzer.compute_jacobian_at_fixed_point(fp, reward=reward_val, action=action_val)
                eigenvalues = np.linalg.eigvals(jac)
                max_abs_eig = np.max(np.abs(eigenvalues))
                stability = "stable" if max_abs_eig < 1.0 else "unstable"
                print(f"     FP {i+1}: {fp}, max|λ|={max_abs_eig:.3f} ({stability})")
    
    # Visualize trajectory
    print("\n7. Recording trajectory of hidden states...")
    agent.reset()
    env.reset()
    
    # Run a short sequence and record hidden states
    action = agent.last_action
    all_hidden_states = []
    
    for trial in range(50):
        reward, info = env.step(action)
        action, switched, hidden_states = agent.act(reward, return_hidden_states=True)
        all_hidden_states.append(hidden_states)
    
    # Concatenate all hidden states
    all_hidden_states = np.concatenate(all_hidden_states, axis=0)
    
    # Plot trajectory
    fig, ax = plt.subplots(figsize=(8, 8))
    analyzer.plot_trajectory(all_hidden_states, ax=ax, alpha=0.7, linewidth=1.5)
    plt.savefig('/Users/zsombi/Library/CloudStorage/GoogleDrive-uuungvarszi@gmail.com/Other computers/My Mac/ZSOMBI/SBRI/PoPy/notebooks/dqn/tmp/hidden_trajectory.png', dpi=150, bbox_inches='tight')
    print("   Saved to: /tmp/hidden_trajectory.png")
    
    # If hidden_size == 2, also plot vector field
    if agent.network.hidden_size == 2:
        print("\n8. Plotting vector field (2D hidden space)...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        analyzer.plot_vector_field_2d(reward=0.0, action=0, ax=axes[0])
        axes[0].set_title('Vector field: no reward')
        
        analyzer.plot_vector_field_2d(reward=1.0, action=0, ax=axes[1])
        axes[1].set_title('Vector field: reward')
        
        plt.savefig('/Users/zsombi/Library/CloudStorage/GoogleDrive-uuungvarszi@gmail.com/Other computers/My Mac/ZSOMBI/SBRI/PoPy/notebooks/dqn/tmp/vector_field.png', dpi=150, bbox_inches='tight')
        print("   Saved to: /tmp/vector_field.png")
    else:
        print("\n8. Skipping vector field plot (hidden_size != 2)")
    
    # Save model
    print("\n9. Saving trained model...")
    agent.save('/Users/zsombi/Library/CloudStorage/GoogleDrive-uuungvarszi@gmail.com/Other computers/My Mac/ZSOMBI/SBRI/PoPy/notebooks/dqn/tmp/tiny_rnn_agent.pth')
    print("   Saved to: /tmp/tiny_rnn_agent.pth")
    
    # Test loading
    print("\n10. Testing model loading...")
    agent_loaded = RNNAgent(
        n_arms=3,
        hidden_size=4,
        rnn_type='GRU',
        use_action_input=True,
        device='cpu'
    )
    agent_loaded.load('/Users/zsombi/Library/CloudStorage/GoogleDrive-uuungvarszi@gmail.com/Other computers/My Mac/ZSOMBI/SBRI/PoPy/notebooks/dqn/tmp/tiny_rnn_agent.pth')
    print("   Model loaded successfully!")
    
    print("\n" + "=" * 60)
    print("Example complete! Check /tmp/ for output plots.")
    print("=" * 60)


if __name__ == '__main__':
    main()
