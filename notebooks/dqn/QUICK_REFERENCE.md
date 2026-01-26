# Quick Reference: Tiny RNN Agent

## Installation
```python
# No installation needed - just import from notebooks.dqn
from notebooks.dqn.rnn_agent import RNNAgent, RNNAnalyzer, set_seed
```

## Basic Training (5 lines)
```python
set_seed(42)
agent = RNNAgent(n_arms=3, hidden_size=4, seed=42)
for trial in range(1000):
    action, _ = agent.act(reward)
    agent.store_reward(reward)
    if agent.should_update(): agent.update()
agent.update(force=True)
```

## Key Parameters

| Parameter | Default | Use Case |
|-----------|---------|----------|
| `hidden_size=2` | - | Maximum interpretability (vector fields) |
| `hidden_size=4` | ✓ | Good balance |
| `hidden_size=8` | - | More capacity |
| `rnn_type='RNN'` | - | Simpler dynamics (easier analysis) |
| `rnn_type='GRU'` | ✓ | Better performance |
| `gamma=1.0` | ✓ | Bandits (no delayed rewards) |
| `gamma<1.0` | - | MDPs (delayed rewards) |
| `entropy_coef=0.01` | ✓ | Exploration bonus |
| `bptt_truncation=20` | ✓ | Update every K trials |

## Common Patterns

### Train and Save
```python
agent = RNNAgent(n_arms=3, hidden_size=4, seed=42)
# ... training loop ...
agent.save('my_agent.pth')
```

### Load and Evaluate
```python
agent = RNNAgent(n_arms=3, hidden_size=4)
agent.load('my_agent.pth')
action, switched = agent.act(reward)
```

### Analyze Strategy
```python
analyzer = RNNAnalyzer(agent)
stats = analyzer.compute_behavioral_stats(recording_df)
print(f"P(stay|reward) = {stats['p_stay_given_reward']:.2%}")
```

### Find Fixed Points
```python
fps = analyzer.find_fixed_points(reward=1.0, action=0, n_inits=10)
for fp in fps:
    jac = analyzer.compute_jacobian_at_fixed_point(fp, reward=1.0, action=0)
    eigenvalues = np.linalg.eigvals(jac)
    print(f"FP: {fp}, max|λ|={np.max(np.abs(eigenvalues)):.3f}")
```

### Visualize Trajectory
```python
action = agent.last_action
hidden_states = []
for trial in range(50):
    reward, _ = env.step(action)
    action, _, h = agent.act(reward, return_hidden_states=True)
    hidden_states.append(h)
all_hidden = np.concatenate(hidden_states, axis=0)
analyzer.plot_trajectory(all_hidden)
```

### Vector Field (2D only)
```python
# Only works if hidden_size=2
analyzer.plot_vector_field_2d(reward=1.0, action=0)
```

## Outputs

### Recording Data
```python
recorder = RNNAgentRecorder()
# In training loop:
recorder.record(action, reward, info, agent, switched, switch_prob)
# After training:
df = recorder.get_recording()
# Columns: trial_id, action, reward, switched, switch_prob, ...
```

### Behavioral Stats
```python
stats = analyzer.compute_behavioral_stats(df)
# Returns dict with:
# - p_stay_given_reward
# - p_switch_given_no_reward
# - switch_rate
# - reward_rate
```

### Update Metrics
```python
metrics = agent.update()
# Returns dict with:
# - loss (total)
# - policy_loss (actor)
# - value_loss (critic)
# - entropy (exploration)
```

## Plotting

### Learning Curves
```python
from notebooks.dqn.rnn_agent import plot_learning_curves
plot_learning_curves(agent, window=50)
```

### Strategy Analysis
```python
from notebooks.dqn.rnn_agent import plot_strategy_analysis
plot_strategy_analysis(recording_df)
```

## Examples

### Run Full Example (4D)
```bash
python notebooks/dqn/example_tiny_rnn_usage.py
```
Outputs: learning curves, strategy analysis, trajectories, checkpoints

### Run 2D Example (Maximum Interpretability)
```bash
python notebooks/dqn/example_2d_rnn.py
```
Outputs: 9-panel comprehensive analysis with vector fields

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Low reward rate | Increase `entropy_coef` or training time |
| Training unstable | Decrease learning rate to 0.0001 |
| No fixed points | Increase `n_inits=50`, train longer |
| Device errors | Set `device='cpu'` explicitly |
| Not reproducible | Call `set_seed(42)` before creating agent |

## Architecture Summary

```
Input: [reward, prev_action_onehot]
  ↓
RNN/GRU (tiny: 2-4 hidden units)
  ↓
├─ Actor:  P(switch) via sigmoid
└─ Critic: V(state) via linear

Loss = -log π * A + 0.5 * MSE(V, R) - 0.01 * H(π)
```

Where:
- A = advantage = R - V(s)
- R = discounted return
- H(π) = entropy of policy

## Files

| File | Purpose |
|------|---------|
| `rnn_agent.py` | Core implementation |
| `README_tiny_rnn.md` | Full documentation |
| `IMPROVEMENTS.md` | Before/after comparison |
| `example_tiny_rnn_usage.py` | 4D agent example |
| `example_2d_rnn.py` | 2D agent with vector fields |

## Resources

- Paper: Ji-an Li et al. (2025) "Discovering cognitive strategies with tiny RNNs"
- Docs: See `README_tiny_rnn.md` for full API reference
- Examples: Both example scripts are self-contained and runnable

---

**Version**: 2.0  
**Last updated**: November 2025
