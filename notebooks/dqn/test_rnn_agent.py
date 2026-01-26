"""
Quick test script for the RNN agent.

Run this to verify the installation and see the agent in action.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} installed")
except ImportError:
    print("✗ PyTorch not found. Install with: pip install torch")
    exit(1)

from rnn_agent import RNNAgent, RNNAgentRecorder

print("\n" + "="*60)
print("RNN Agent Quick Test")
print("="*60)

# Create environment
print("\n1. Creating 3-armed bandit environment...")
env = gym.make("zsombi/monkey-bandit-task-v0", n_arms=3, max_episode_steps=5000)
print("   ✓ Environment created")

# Create agent
print("\n2. Initializing RNN agent...")
agent = RNNAgent(
    n_arms=3,
    hidden_size=32,
    learning_rate=0.001,
    gamma=0.95,
    exploration_noise=0.05,
    feedback_duration=3,
    decision_delay=1
)
print("   ✓ Agent initialized")

# Quick training run
print("\n3. Running short training (5000 steps)...")
recorder = RNNAgentRecorder()
obs, info = env.reset()
agent.reset()

last_reward = 0
update_interval = 100
n_steps = 5000

for step in range(n_steps):
    # Agent acts
    action, switched = agent.act(last_reward)
    
    # Environment responds
    obs, reward, terminated, done, info = env.step(action)
    agent.store_reward(reward)
    
    # Record
    switch_prob = agent.get_switch_probability(last_reward)
    recorder.record(action, reward, info, agent, switched, switch_prob)
    
    last_reward = reward
    
    # Update network
    if step > 0 and step % update_interval == 0:
        loss = agent.update()
        
    # Progress indicator
    if step > 0 and step % 1000 == 0:
        print(f"   Step {step}/{n_steps}")

print("   ✓ Training complete")

# Analyze results
print("\n4. Results:")
rec = recorder.get_recording()

mean_reward = rec['reward'].mean()
switch_rate = rec['switched'].mean()
switch_after_reward = rec[rec['reward'] == 1]['switch_prob'].mean()
switch_after_no_reward = rec[rec['reward'] == 0]['switch_prob'].mean()

print(f"   Mean reward: {mean_reward:.3f}")
print(f"   Switch rate: {switch_rate:.3f}")
print(f"   Switch prob after reward: {switch_after_reward:.3f}")
print(f"   Switch prob after no reward: {switch_after_no_reward:.3f}")

# Quick plot
print("\n5. Generating plot...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Learning curve
window = 200
running_avg = rec['reward'].rolling(window=window).mean()
axes[0].plot(running_avg)
axes[0].set_xlabel('Trial')
axes[0].set_ylabel(f'Running Avg Reward (window={window})')
axes[0].set_title('Learning Curve')
axes[0].grid(True, alpha=0.3)

# Plot 2: Switch behavior
reward_trials = rec[rec['reward'] == 1]['switch_prob']
no_reward_trials = rec[rec['reward'] == 0]['switch_prob']

axes[1].hist([reward_trials, no_reward_trials], bins=20, alpha=0.6,
             label=['After Reward', 'After No Reward'])
axes[1].set_xlabel('Switch Probability')
axes[1].set_ylabel('Count')
axes[1].set_title('Switch Behavior by Outcome')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_agent_test_results.png', dpi=150, bbox_inches='tight')
print("   ✓ Plot saved as 'rnn_agent_test_results.png'")

plt.show()

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)
print("\nNext steps:")
print("- Open results.ipynb for detailed training and analysis")
print("- Read RNN_AGENT_README.md for documentation")
print("- Experiment with different hyperparameters")
print("="*60)
