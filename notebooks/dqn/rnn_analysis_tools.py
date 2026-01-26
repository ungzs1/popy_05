"""
Visualization and analysis tools for RNN agent.

This module provides utilities for:
- Analyzing hidden state dynamics
- Visualizing learned representations
- Comparing strategies across conditions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import List, Tuple, Optional


def extract_hidden_states(agent, trial_sequence: List[Tuple[int, int]]) -> np.ndarray:
    """
    Extract hidden states from the RNN for a sequence of trials.
    
    Parameters
    ----------
    agent : RNNAgent
        The trained RNN agent
    trial_sequence : List[Tuple[int, int]]
        List of (reward, previous_switch) tuples
        
    Returns
    -------
    hidden_states : np.ndarray
        Array of hidden states (n_trials, hidden_size)
    """
    agent.network.reset_hidden(batch_size=1)
    hidden_states = []
    
    with torch.no_grad():
        for reward, prev_switch in trial_sequence:
            # Prepare input
            network_input = torch.FloatTensor([[reward, prev_switch]])
            network_input = network_input.unsqueeze(0)
            network_input = network_input.repeat(1, agent.feedback_duration, 1)
            
            # Forward pass
            _, hidden = agent.network(network_input, agent.network.hidden)
            agent.network.hidden = hidden
            
            # Store hidden state
            hidden_states.append(hidden.squeeze().numpy())
    
    return np.array(hidden_states)


def plot_hidden_state_dynamics(hidden_states: np.ndarray, 
                               rewards: List[int],
                               switches: List[int],
                               n_components: int = 3,
                               figsize: Tuple[int, int] = (15, 5)):
    """
    Plot the dynamics of hidden states over time.
    
    Parameters
    ----------
    hidden_states : np.ndarray
        Hidden states from RNN (n_trials, hidden_size)
    rewards : List[int]
        Reward on each trial
    switches : List[int]
        Whether agent switched on each trial
    n_components : int
        Number of top principal components to plot
    """
    from sklearn.decomposition import PCA
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    hidden_pca = pca.fit_transform(hidden_states)
    
    fig, axes = plt.subplots(1, n_components, figsize=figsize)
    if n_components == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # Color by reward
        scatter = ax.scatter(range(len(hidden_pca)), hidden_pca[:, i],
                           c=rewards, cmap='RdYlGn', alpha=0.6, s=20)
        
        # Mark switches
        switch_idx = [j for j, s in enumerate(switches) if s == 1]
        ax.scatter([switch_idx], hidden_pca[switch_idx, i],
                  marker='x', c='black', s=100, alpha=0.8, label='Switch')
        
        ax.set_xlabel('Trial')
        ax.set_ylabel(f'PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)')
        ax.set_title(f'Principal Component {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=axes[-1], label='Reward')
    plt.tight_layout()
    
    return fig, hidden_pca


def plot_hidden_state_clustering(hidden_states: np.ndarray,
                                 rewards: List[int],
                                 switches: List[int],
                                 method: str = 'tsne',
                                 figsize: Tuple[int, int] = (12, 5)):
    """
    Visualize hidden state clustering using dimensionality reduction.
    
    Parameters
    ----------
    hidden_states : np.ndarray
        Hidden states from RNN
    rewards : List[int]
        Reward on each trial
    switches : List[int]
        Whether agent switched
    method : str
        'tsne' or 'umap' for dimensionality reduction
    """
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'umap':
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("method must be 'tsne' or 'umap'")
    
    # Reduce dimensions
    embedded = reducer.fit_transform(hidden_states)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot colored by reward
    scatter1 = axes[0].scatter(embedded[:, 0], embedded[:, 1],
                              c=rewards, cmap='RdYlGn', alpha=0.6, s=20)
    axes[0].set_title(f'Hidden States ({method.upper()}) - Colored by Reward')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    plt.colorbar(scatter1, ax=axes[0], label='Reward')
    
    # Plot colored by switch
    scatter2 = axes[1].scatter(embedded[:, 0], embedded[:, 1],
                              c=switches, cmap='coolwarm', alpha=0.6, s=20)
    axes[1].set_title(f'Hidden States ({method.upper()}) - Colored by Switch')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    plt.colorbar(scatter2, ax=axes[1], label='Switched')
    
    plt.tight_layout()
    return fig


def analyze_switch_probability_landscape(agent,
                                        reward_range: List[int] = [0, 1],
                                        prev_switch_range: List[int] = [0, 1],
                                        figsize: Tuple[int, int] = (10, 8)):
    """
    Visualize how switch probability varies with inputs.
    
    Parameters
    ----------
    agent : RNNAgent
        Trained RNN agent
    reward_range : List[int]
        Range of reward values to test
    prev_switch_range : List[int]
        Range of previous switch values to test
    """
    # Create grid of inputs
    n_points = 20
    reward_vals = np.linspace(min(reward_range), max(reward_range), n_points)
    prev_switch_vals = np.linspace(min(prev_switch_range), max(prev_switch_range), n_points)
    
    switch_probs = np.zeros((len(reward_vals), len(prev_switch_vals)))
    
    # Compute switch probability for each input combination
    with torch.no_grad():
        for i, reward in enumerate(reward_vals):
            for j, prev_switch in enumerate(prev_switch_vals):
                agent.network.reset_hidden(batch_size=1)
                network_input = torch.FloatTensor([[reward, prev_switch]])
                network_input = network_input.unsqueeze(0)
                network_input = network_input.repeat(1, agent.feedback_duration, 1)
                
                switch_prob, _ = agent.network(network_input, agent.network.hidden)
                switch_probs[i, j] = switch_prob.item()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(switch_probs, aspect='auto', origin='lower',
                   extent=[prev_switch_vals[0], prev_switch_vals[-1],
                          reward_vals[0], reward_vals[-1]],
                   cmap='RdYlBu_r')
    
    ax.set_xlabel('Previous Switch')
    ax.set_ylabel('Reward')
    ax.set_title('Switch Probability Landscape')
    
    # Add contour lines
    contours = ax.contour(prev_switch_vals, reward_vals, switch_probs,
                         levels=[0.25, 0.5, 0.75], colors='black',
                         linewidths=1, alpha=0.5)
    ax.clabel(contours, inline=True, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Switch Probability')
    plt.tight_layout()
    
    return fig


def compare_agent_strategies(rec_dict: dict,
                            agent_names: List[str],
                            figsize: Tuple[int, int] = (15, 10)):
    """
    Compare strategies across multiple agents.
    
    Parameters
    ----------
    rec_dict : dict
        Dictionary mapping agent names to their recording DataFrames
    agent_names : List[str]
        Names of agents to compare
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Plot 1: Mean reward
    mean_rewards = [rec_dict[name]['reward'].mean() for name in agent_names]
    axes[0].bar(agent_names, mean_rewards)
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Overall Performance')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Switch rate
    switch_rates = [rec_dict[name]['switched'].mean() if 'switched' in rec_dict[name].columns
                   else np.nan for name in agent_names]
    axes[1].bar(agent_names, switch_rates)
    axes[1].set_ylabel('Switch Rate')
    axes[1].set_title('Overall Switch Rate')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Learning curves
    window = 500
    for name in agent_names:
        running_avg = rec_dict[name]['reward'].rolling(window=window).mean()
        axes[2].plot(running_avg, label=name, alpha=0.7)
    axes[2].set_xlabel('Trial')
    axes[2].set_ylabel(f'Running Avg (window={window})')
    axes[2].set_title('Learning Curves')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Switch behavior after reward
    for name in agent_names:
        rec = rec_dict[name]
        if 'switch_prob' in rec.columns:
            reward_switch = rec[rec['reward'] == 1]['switch_prob']
            axes[3].hist(reward_switch, bins=20, alpha=0.5, label=name)
    axes[3].set_xlabel('Switch Probability')
    axes[3].set_ylabel('Count')
    axes[3].set_title('Switch Prob After Reward')
    axes[3].legend()
    
    # Plot 5: Switch behavior after no reward
    for name in agent_names:
        rec = rec_dict[name]
        if 'switch_prob' in rec.columns:
            no_reward_switch = rec[rec['reward'] == 0]['switch_prob']
            axes[4].hist(no_reward_switch, bins=20, alpha=0.5, label=name)
    axes[4].set_xlabel('Switch Probability')
    axes[4].set_ylabel('Count')
    axes[4].set_title('Switch Prob After No Reward')
    axes[4].legend()
    
    # Plot 6: Block-wise performance
    for name in agent_names:
        block_rewards = rec_dict[name].groupby('block_id')['reward'].mean()
        axes[5].plot(block_rewards, label=name, alpha=0.7)
    axes[5].set_xlabel('Block ID')
    axes[5].set_ylabel('Mean Reward')
    axes[5].set_title('Performance by Block')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_trial_by_trial_analysis(rec: pd.DataFrame,
                                 start_trial: int = 0,
                                 n_trials: int = 100,
                                 figsize: Tuple[int, int] = (15, 8)):
    """
    Detailed trial-by-trial analysis of agent behavior.
    
    Parameters
    ----------
    rec : pd.DataFrame
        Recording DataFrame
    start_trial : int
        Trial to start from
    n_trials : int
        Number of trials to plot
    """
    end_trial = min(start_trial + n_trials, len(rec))
    rec_slice = rec.iloc[start_trial:end_trial]
    
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    trials = range(start_trial, end_trial)
    
    # Plot 1: Actions and best arm
    axes[0].plot(trials, rec_slice['action'], 'o-', label='Chosen Action', markersize=3)
    axes[0].plot(trials, rec_slice['best_arm'], '--', label='Best Arm', linewidth=2)
    axes[0].set_ylabel('Action')
    axes[0].set_title('Actions Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Rewards
    axes[1].stem(trials, rec_slice['reward'], basefmt=' ')
    axes[1].set_ylabel('Reward')
    axes[1].set_title('Rewards Received')
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Switches
    if 'switched' in rec_slice.columns:
        axes[2].stem(trials, rec_slice['switched'], basefmt=' ')
        axes[2].set_ylabel('Switched')
        axes[2].set_title('Switch Events')
        axes[2].set_ylim([-0.1, 1.1])
        axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Switch probability
    if 'switch_prob' in rec_slice.columns:
        axes[3].plot(trials, rec_slice['switch_prob'], 'o-', markersize=3)
        axes[3].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        axes[3].set_xlabel('Trial')
        axes[3].set_ylabel('Switch Prob')
        axes[3].set_title('Switch Probability Over Time')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("RNN Agent Analysis Tools")
    print("=" * 50)
    print("\nAvailable functions:")
    print("- extract_hidden_states: Extract RNN hidden states")
    print("- plot_hidden_state_dynamics: Visualize hidden state evolution")
    print("- plot_hidden_state_clustering: Cluster analysis of hidden states")
    print("- analyze_switch_probability_landscape: Switch prob as function of inputs")
    print("- compare_agent_strategies: Compare multiple agents")
    print("- plot_trial_by_trial_analysis: Detailed trial-by-trial plots")
    print("\nImport these functions in your notebook for analysis!")
