# Tiny RNN Agent for Cognitive Strategy Discovery

This module implements a **tiny recurrent neural network (RNN) agent** for reinforcement learning on bandit tasks, designed for interpretability and dynamical systems analysis. 

## Key Features

### 🎯 Core Improvements (v2.0)

1. **Actor-Critic Architecture**
   - Value baseline for variance reduction
   - Separate actor (policy) and critic (value) heads
   - Stabilized training with advantage estimation

2. **Truncated Backpropagation Through Time (BPTT)**
   - Learn long-term dependencies across trials
   - Configurable truncation boundary (default: 20 trials)
   - Proper gradient flow through hidden states

3. **Correct Sampling with Entropy Regularization**
   - Uses `torch.distributions.Bernoulli` for principled sampling
   - No external noise injection (was causing bias)
   - Entropy bonus prevents premature convergence

4. **Tiny Architecture for Interpretability**
   - Default: 4 hidden units (down from 32)
   - Supports vanilla RNN or GRU
   - Designed for fixed-point analysis and trajectory visualization

5. **Improved Input Representation**
   - Uses `[previous_action (one-hot), reward]` instead of `[reward, previous_switch]`
   - Preserves which arm was chosen (not just switch/stay)
   - Enables learning arm-specific preferences

6. **Device Handling & Reproducibility**
   - Proper CPU/GPU device management
   - Seed setting for reproducible experiments
   - Model checkpointing (save/load)

### 🔬 Analysis Tools

**`RNNAnalyzer` class** provides:
- **Fixed-point finder**: Discover attractor states under different input conditions
- **Jacobian analysis**: Compute stability of fixed points (eigenvalue decomposition)
- **Vector field visualization**: Plot dynamics in 2D hidden space
- **Trajectory plotting**: Visualize hidden state evolution over trials
- **Behavioral statistics**: Compute P(stay|reward), P(switch|no_reward), etc.

**Plotting utilities**:
- `plot_learning_curves()`: Track reward and loss during training
- `plot_strategy_analysis()`: Characterize learned strategy (WSLS, perseveration, etc.)

## Quick Start

### Basic Usage

```python
from rnn_agent import RNNAgent, set_seed

# Set seed for reproducibility
set_seed(42)

# Create agent with tiny RNN (4 hidden units)
agent = RNNAgent(
    n_arms=3,
    hidden_size=4,          # Tiny for interpretability
    rnn_type='GRU',         # or 'RNN' for vanilla RNN
    learning_rate=0.001,
    gamma=1.0,              # No discounting for bandits
    entropy_coef=0.01,      # Entropy bonus for exploration
    bptt_truncation=20,     # Truncate BPTT every 20 trials
    device='cpu',
    seed=42
)

# Training loop
agent.reset()
for trial in range(1000):
    reward = get_reward_from_environment(action)
    action, switched = agent.act(reward)
    agent.store_reward(reward)
    
    # Update at truncation boundaries
    if agent.should_update():
        metrics = agent.update()

# Final update
agent.update(force=True)

# Save trained model
agent.save('trained_agent.pth')
```

### Analysis

```python
from rnn_agent import RNNAnalyzer, plot_learning_curves, plot_strategy_analysis

# Plot learning curves
plot_learning_curves(agent, window=50)

# Create analyzer
analyzer = RNNAnalyzer(agent)

# Compute behavioral statistics
stats = analyzer.compute_behavioral_stats(recording_df)
print(f"P(stay | reward): {stats['p_stay_given_reward']:.2%}")

# Find fixed points
fps = analyzer.find_fixed_points(reward=1.0, action=0)
for fp in fps:
    jac = analyzer.compute_jacobian_at_fixed_point(fp, reward=1.0, action=0)
    eigenvalues = np.linalg.eigvals(jac)
    print(f"Fixed point: {fp}, max|λ|={np.max(np.abs(eigenvalues)):.3f}")

# Plot trajectory
action = agent.last_action
hidden_states = []
for trial in range(50):
    reward, _ = env.step(action)
    action, switched, h = agent.act(reward, return_hidden_states=True)
    hidden_states.append(h)

all_hidden = np.concatenate(hidden_states, axis=0)
analyzer.plot_trajectory(all_hidden)

# Plot vector field (if hidden_size == 2)
if agent.network.hidden_size == 2:
    analyzer.plot_vector_field_2d(reward=1.0, action=0)
```

## Architecture

### SwitchRNN (Network)

```
Input: [reward, prev_action_onehot]  → (1 + n_arms dimensions)
   ↓
RNN/GRU (tiny: 4 hidden units)
   ↓
Actor head → sigmoid → P(switch)
Critic head → linear → V(state)
```

**Key parameters**:
- `hidden_size`: Number of hidden units (default: 4 for interpretability)
- `rnn_type`: 'RNN' (vanilla) or 'GRU' (default)
- `use_action_input`: Include previous action in input (default: True)

### RNNAgent

**Training algorithm**: Actor-Critic with truncated BPTT

1. **Forward pass**: 
   - Build input from `[reward, prev_action_onehot]`
   - Run through RNN for `feedback_duration` steps
   - Optional `decision_delay` with zero input
   - Sample action from `Bernoulli(P(switch))`

2. **Backward pass** (every K trials):
   - Compute returns: R_t = Σ γ^k r_{t+k}
   - Compute advantages: A_t = R_t - V(s_t)
   - Actor loss: -log π(a|s) * A
   - Critic loss: MSE(V(s), R)
   - Entropy bonus: -β * H(π)
   - Gradient clipping (max_norm=1.0)

**Key parameters**:
- `gamma`: Discount factor (default: 1.0 for bandits)
- `entropy_coef`: Entropy bonus coefficient (default: 0.01)
- `value_coef`: Value loss coefficient (default: 0.5)
- `bptt_truncation`: Truncate BPTT every K trials (default: 20)

## What Changed from v1.0?

### Fixed Bugs 🐛

1. **BPTT was broken**: Hidden state was detached every trial → no learning of long-term dependencies
2. **Sampling/log_prob mismatch**: Added Gaussian noise to sampling but computed log_prob from un-noised Bernoulli → biased policy gradient
3. **Decision delay bug**: In analysis branch, computed switch_prob before running delay steps
4. **Inappropriate discounting**: Used γ=0.99 for bandits (should be 1.0)
5. **Device handling missing**: Hidden state always created on CPU

### Improvements 🚀

1. **Variance reduction**: Added value baseline (actor-critic) instead of pure REINFORCE
2. **Exploration**: Entropy bonus instead of external noise injection
3. **Interpretability**: Smaller default hidden size (4 vs 32)
4. **Input representation**: `[reward, action_onehot]` preserves arm identity
5. **Analysis tools**: Fixed-point finder, Jacobian, trajectories, vector fields
6. **Reproducibility**: Seeding, save/load, device management
7. **API cleanup**: Removed optimizer from network, unified forward paths

## Theoretical Background

This implementation is inspired by:

> **Ji-an Li et al. (2025)**. "Discovering cognitive strategies with tiny recurrent neural networks."

**Key ideas**:
1. Use **tiny RNNs** (2-4 hidden units) to discover interpretable strategies
2. Analyze learned strategies via **dynamical systems tools** (fixed points, Jacobian eigenvalues, phase portraits)
3. Different regularization strengths → different strategy regimes (simple WSLS ↔ complex inference)
4. Fixed points correspond to **behavioral modes** (explore, exploit, switch, stay)
5. Separatrices in hidden space define **decision boundaries**

## Example: 3-Armed Bandit

See `example_tiny_rnn_usage.py` for a complete example:

```bash
python notebooks/dqn/example_tiny_rnn_usage.py
```

This will:
1. Train a tiny RNN agent (4 hidden units) on a 3-armed bandit
2. Plot learning curves (reward, loss)
3. Analyze behavioral strategy (P(switch|reward), etc.)
4. Find fixed points for different input conditions
5. Visualize hidden state trajectories
6. Save trained model and plots

## Tips for Interpretability

### For Tiny RNNs (2-4 hidden units):

1. **Use vanilla RNN** for simpler dynamics (easier fixed-point analysis)
2. **Visualize vector fields** if hidden_size=2
3. **Sweep hidden sizes** (2, 3, 4, 8) to see simplicity-performance tradeoff
4. **Add weight decay** on recurrent weights to encourage sparse dynamics
5. **Train multiple seeds** and compare discovered strategies

### For Strategy Discovery:

1. **Fit to real data**: Use behavior cloning loss + RL to match observed choices
2. **Compare to baselines**: WSLS (win-stay-lose-shift), ε-greedy, UCB
3. **Probe with specific sequences**: Test reward → switch, reward → stay, etc.
4. **Cluster fixed points**: Different input conditions → different attractor states
5. **Analyze Jacobian eigenvectors**: Directions of contraction/expansion

## API Reference

### RNNAgent

**Methods**:
- `reset()`: Reset agent state
- `act(reward, return_hidden_states=False)`: Decide action given reward
- `store_reward(reward)`: Buffer reward for training
- `should_update()`: Check if at truncation boundary
- `update(force=False)`: Update network (actor-critic + BPTT)
- `get_switch_probability(reward)`: Query policy without acting
- `save(filepath)` / `load(filepath)`: Checkpoint model

### RNNAnalyzer

**Methods**:
- `compute_behavioral_stats(recording_df)`: P(switch|reward), etc.
- `find_fixed_points(reward, action, n_inits, max_iter, tol)`: Find attractors
- `compute_jacobian_at_fixed_point(fp, reward, action)`: Stability analysis
- `plot_vector_field_2d(reward, action, xlim, ylim, n_grid)`: Visualize dynamics
- `plot_trajectory(hidden_states)`: Plot hidden state evolution

### Utilities

- `set_seed(seed)`: Set random seeds (numpy, torch, random)
- `plot_learning_curves(agent, window)`: Plot reward and loss
- `plot_strategy_analysis(recording_df)`: Plot behavioral strategy

## Common Issues

**Q: Agent doesn't learn / reward stays low**
- Check that `update()` is called regularly (every `bptt_truncation` trials)
- Try smaller learning rate (0.0001) or larger entropy bonus (0.05)
- Ensure environment is learnable (at least one arm has high reward probability)

**Q: Fixed-point finder returns no results**
- Increase `n_inits` (try 20-50)
- Check that network is trained (not random initialization)
- Try different input conditions (reward=0/1, different actions)

**Q: Hidden state trajectories look chaotic**
- Add weight decay to regularize recurrent weights
- Use vanilla RNN instead of GRU for simpler dynamics
- Reduce hidden size to 2-3 for more constrained dynamics

**Q: ValueError: tensor on different devices**
- Ensure `device` parameter matches your hardware (use 'cpu' if no GPU)
- Check that all inputs are created with `device=agent.device`

## Citation

If you use this code, please cite:

```bibtex
@article{ji2025discovering,
  title={Discovering cognitive strategies with tiny recurrent neural networks},
  author={Ji-an Li and others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License (or match your project's license)

---

**Version**: 2.0  
**Last updated**: November 2025  
**Maintainer**: [Your name]
