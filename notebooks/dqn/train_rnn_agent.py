"""
Train the RNN agent on the 3-armed bandit task.

This script:
1. Initializes the environment and RNN agent
2. Trains the agent for a specified number of steps
3. Saves the trained model
4. Runs a full simulation with the trained agent
5. Saves the training losses and simulation results

Usage:
    python train_rnn_agent.py
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import os

from rnn_agent import RNNAgent, RNNAgentRecorder
from popy.simulation_tools import MonkeyBanditTask


def train_agent(
    n_training_steps=10_000_000,
    update_interval=100,
    n_arms=3,
    hidden_size=32,
    learning_rate=0.001,
    gamma=0.95,
    exploration_noise=0.05,
    feedback_duration=3,
    decision_delay=1,
    save_dir='results'
):
    """
    Train the RNN agent.
    
    Parameters
    ----------
    n_training_steps : int
        Number of training steps
    update_interval : int
        Update network every N steps
    n_arms : int
        Number of arms in the bandit task
    hidden_size : int
        Size of RNN hidden state
    learning_rate : float
        Learning rate for optimizer
    gamma : float
        Discount factor
    exploration_noise : float
        Noise in decision making
    feedback_duration : int
        Timesteps to present feedback
    decision_delay : int
        Delay between feedback and decision
    save_dir : str
        Directory to save results
        
    Returns
    -------
    agent : RNNAgent
        Trained agent
    losses : list
        Training losses at each update
    mean_rewards : list
        Mean rewards at each update
    """
    print("="*60)
    print("RNN AGENT TRAINING")
    print("="*60)
    print(f"Training steps: {n_training_steps:,}")
    print(f"Update interval: {update_interval}")
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*60)
    
    # Create environment
    print("\n1. Creating environment...")
    env = gym.make("zsombi/monkey-bandit-task-v0", n_arms=n_arms, max_episode_steps=n_training_steps)
    print("   ✓ Environment created")
    
    # Create agent
    print("\n2. Initializing RNN agent...")
    agent = RNNAgent(
        n_arms=n_arms,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        gamma=gamma,
        exploration_noise=exploration_noise,
        feedback_duration=feedback_duration,
        decision_delay=decision_delay
    )
    print("   ✓ Agent initialized")
    
    # Training loop
    print("\n3. Starting training...")
    recorder = RNNAgentRecorder()
    losses = []
    mean_rewards = []
    
    obs, info = env.reset()
    agent.reset()
    
    last_reward = 0
    step_count = 0
    episode_count = 0
    
    while step_count < n_training_steps:
        # Agent acts
        action, switched = agent.act(last_reward)
        
        # Environment step
        obs, reward, terminated, done, info = env.step(action)
        
        # Store reward for learning
        agent.store_reward(reward)
        
        # Get switch probability for recording
        switch_prob = agent.get_switch_probability(last_reward)
        
        # Record behavior
        recorder.record(action, reward, info, agent, switched, switch_prob)
        
        last_reward = reward
        step_count += 1
        
        # Update network periodically
        if step_count % update_interval == 0:
            loss = agent.update()
            losses.append(loss)
            
            # Compute mean reward over last interval
            recent_rewards = [recorder.recording[i]['reward'] 
                             for i in range(max(0, len(recorder.recording) - update_interval), 
                                           len(recorder.recording))]
            mean_reward = np.mean(recent_rewards) if recent_rewards else 0
            mean_rewards.append(mean_reward)
            
            episode_count += 1
            
            # Progress update
            if episode_count % 50 == 0:
                print(f"   Episode {episode_count:,}, Steps: {step_count:,}, "
                      f"Mean Reward: {mean_reward:.3f}, Loss: {loss:.4f}")
    
    print(f"\n   ✓ Training complete!")
    print(f"   Total steps: {step_count}")
    print(f"   Total episodes: {episode_count}")
    print(f"   Final mean reward: {mean_rewards[-1]:.3f}")
    
    return agent, losses, mean_rewards


def simulate_with_trained_agent(
    agent,
    n_simulation_steps=100_000,
    n_arms=3,
    save_dir='results'
):
    """
    Run a full simulation with the trained agent.
    
    Parameters
    ----------
    agent : RNNAgent
        Trained agent
    n_simulation_steps : int
        Number of simulation steps
    n_arms : int
        Number of arms
    save_dir : str
        Directory to save results
        
    Returns
    -------
    recording_df : pd.DataFrame
        Recording of the simulation
    hidden_states_array : np.ndarray
        Hidden states for all trials, shape (n_trials, n_neurons, n_timesteps)
    """
    print("\n4. Running simulation with trained agent...")
    print(f"   Simulation steps: {n_simulation_steps}")
    
    # Create fresh environment for simulation
    env = gym.make("zsombi/monkey-bandit-task-v0", n_arms=n_arms, max_episode_steps=n_simulation_steps)
    
    # Reset agent
    agent.reset()
    agent.exploration_noise = 0.0  # Disable exploration for evaluation
    
    # Record simulation
    recorder = RNNAgentRecorder()
    hidden_states_list = []  # List to store hidden states for each trial
    
    obs, info = env.reset()
    last_reward = 0
    step_count = 0
    
    while step_count < n_simulation_steps:
        # Agent acts (no learning, just inference) and get hidden states
        action, switched, hidden_states = agent.act(last_reward, return_hidden_states=True)
        
        # Environment step
        obs, reward, terminated, done, info = env.step(action)
        
        # Get switch probability
        switch_prob = agent.get_switch_probability(last_reward)
        
        # Record behavior
        recorder.record(action, reward, info, agent, switched, switch_prob)
        
        # Store hidden states for this trial
        hidden_states_list.append(hidden_states)  # shape: (n_timesteps, n_neurons)
        
        last_reward = reward
        step_count += 1
        
        # Progress update
        if step_count % 20000 == 0:
            recent_rewards = [recorder.recording[i]['reward'] 
                             for i in range(max(0, len(recorder.recording) - 1000), 
                                           len(recorder.recording))]
            mean_reward = np.mean(recent_rewards)
            print(f"   Step {step_count}/{n_simulation_steps}, "
                  f"Recent mean reward: {mean_reward:.3f}")
    
    # Get recording
    recording_df = recorder.get_recording()
    
    # Convert hidden states to array: (n_trials, n_neurons, n_timesteps)
    # First, we need to transpose each trial's hidden states from (n_timesteps, n_neurons) to (n_neurons, n_timesteps)
    hidden_states_array = np.array([h.T for h in hidden_states_list])
    
    print(f"\n   ✓ Simulation complete!")
    print(f"   Total trials: {len(recording_df)}")
    print(f"   Overall mean reward: {recording_df['reward'].mean():.3f}")
    print(f"   Hidden states shape: {hidden_states_array.shape} (trials, neurons, timesteps)")
    
    return recording_df, hidden_states_array


def save_results(agent, losses, mean_rewards, recording_df, hidden_states_array, save_dir='results'):
    """
    Save all results to disk.
    
    Parameters
    ----------
    agent : RNNAgent
        Trained agent
    losses : list
        Training losses
    mean_rewards : list
        Mean rewards during training
    recording_df : pd.DataFrame
        Simulation recording
    hidden_states_array : np.ndarray
        Hidden states array (n_trials, n_neurons, n_timesteps)
    save_dir : str
        Directory to save results
    """
    print("\n5. Saving results...")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'rnn_agent_trained.pth')
    torch.save(agent.network.state_dict(), model_path)
    print(f"   ✓ Model saved to: {model_path}")
    
    # Save training losses
    training_data = pd.DataFrame({
        'episode': range(1, len(losses) + 1),
        'loss': losses,
        'mean_reward': mean_rewards
    })
    training_path = os.path.join(save_dir, 'training_history.csv')
    training_data.to_csv(training_path, index=False)
    print(f"   ✓ Training history saved to: {training_path}")
    
    # Save simulation recording
    recording_path = os.path.join(save_dir, 'simulation_recording.csv')
    recording_df.to_csv(recording_path, index=False)
    print(f"   ✓ Simulation recording saved to: {recording_path}")
    
    # Save agent parameters
    params = {
        'n_arms': agent.n_arms,
        'hidden_size': agent.network.hidden_size,
        'learning_rate': agent.network.optimizer.param_groups[0]['lr'],
        'gamma': agent.gamma,
        'exploration_noise': agent.exploration_noise,
        'feedback_duration': agent.feedback_duration,
        'decision_delay': agent.decision_delay,
    }
    params_df = pd.DataFrame([params])
    params_path = os.path.join(save_dir, 'agent_parameters.csv')
    params_df.to_csv(params_path, index=False)
    print(f"   ✓ Agent parameters saved to: {params_path}")
    
    # Save hidden states as numpy array
    hidden_states_path = os.path.join(save_dir, 'hidden_states.npy')
    np.save(hidden_states_path, hidden_states_array)
    print(f"   ✓ Hidden states saved to: {hidden_states_path}")
    print(f"     Shape: {hidden_states_array.shape} (trials, neurons, timesteps)")
    
    print(f"\n   All results saved to: {save_dir}/")


def main():
    """Main training pipeline."""
    # Training parameters
    n_training_steps = 10_000_000 
    n_simulation_steps = 100_000
    update_interval = 1000
    save_dir = '/Users/zsombi/Library/CloudStorage/GoogleDrive-uuungvarszi@gmail.com/Other computers/My Mac/ZSOMBI/SBRI/PoPy/notebooks/dqn/results'
    
    # Agent parameters
    agent_params = {
        'n_arms': 3,
        'hidden_size': 32,
        'learning_rate': 0.001,
        'gamma': 0.95,
        'exploration_noise': 0.05,
        'feedback_duration': 3,  
        'decision_delay': 1, 
    }
    
    # Train agent
    agent, losses, mean_rewards = train_agent(
        n_training_steps=n_training_steps,
        update_interval=update_interval,
        save_dir=save_dir,
        **agent_params
    )
    
    # Simulate with trained agent
    recording_df, hidden_states_array = simulate_with_trained_agent(
        agent,
        n_simulation_steps=n_simulation_steps,
        n_arms=agent_params['n_arms'],
        save_dir=save_dir
    )
    
    # Save all results
    save_results(agent, losses, mean_rewards, recording_df, hidden_states_array, save_dir)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nTo analyze results, run the notebook 'results.ipynb'")
    print(f"or load the files from: {save_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
