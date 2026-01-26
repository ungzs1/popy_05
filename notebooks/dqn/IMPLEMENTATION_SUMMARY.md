# RNN Agent Implementation Summary

## What Was Created

I've implemented a complete RNN-based reinforcement learning agent for the 3-armed bandit task. Here's what was delivered:

### Core Implementation Files

1. **`rnn_agent.py`** (Main Implementation)
   - `SwitchRNN`: Neural network class (GRU-based)
   - `RNNAgent`: Complete RL agent with REINFORCE algorithm
   - `RNNAgentRecorder`: Recording utility for tracking behavior
   
   **Key Features:**
   - Inputs: Binary reward + previous switch indicator
   - Temporal dynamics: Configurable feedback duration and decision delay
   - Learning: Policy gradient (REINFORCE) with gradient clipping
   - Exploration: Gaussian noise in decision making
   - Output: Probability of switching actions

2. **`results.ipynb`** (Updated Notebook)
   - Added comprehensive tutorial cells for RNN agent
   - Training loop with progress monitoring
   - Performance analysis and visualization
   - Comparison with rule-based ForagingAgent
   - Block-wise and trial-wise analysis
   - Advanced analysis section (optional)

### Documentation Files

3. **`RNN_AGENT_README.md`** (Complete Documentation)
   - Architecture explanation
   - Learning algorithm details
   - Usage examples
   - Customization guide
   - Troubleshooting tips
   - References to relevant literature

4. **`requirements_rnn.txt`** (Dependencies)
   - PyTorch and dependencies
   - Installation instructions for CPU and GPU
   - Verification steps

### Utility Files

5. **`test_rnn_agent.py`** (Quick Test Script)
   - Standalone script to verify installation
   - Runs a short training session
   - Generates quick visualizations
   - Useful for debugging

6. **`rnn_analysis_tools.py`** (Advanced Analysis)
   - Hidden state extraction and visualization
   - PCA/t-SNE/UMAP dimensionality reduction
   - Switch probability landscape analysis
   - Multi-agent comparison tools
   - Trial-by-trial detailed plotting

## Agent Design Overview

### The Concept

The agent learns a **stay/switch policy** based on reward feedback:

1. **Receives outcome**: Binary feedback (0 or 1)
2. **Processes with RNN**: Temporal integration of feedback history
3. **Outputs switch probability**: How likely to switch actions
4. **Makes decision**: Stay (repeat) or Switch (random alternative)
5. **Learns from experience**: Updates network weights using policy gradient

### Key Differences from Rule-Based Agent

| Aspect | ForagingAgent (Rule-Based) | RNNAgent (Learning-Based) |
|--------|---------------------------|---------------------------|
| Value Update | Explicit: V += α(r - V) | Implicit: Learned by RNN |
| Switch Decision | Threshold: σ(β(V - V₀)) | Neural: RNN → sigmoid |
| Strategy | Hand-designed | Discovered through learning |
| Parameters | Must be tuned | Self-adapts during training |
| Temporal Processing | None | RNN temporal dynamics |
| Flexibility | Fixed rules | Can learn complex patterns |

### Temporal Dynamics

The RNN includes realistic temporal processing:

```
Trial Timeline:
|--Feedback Period--|--Delay--|--Decision--|
   (3 timesteps)    (1 step)   (action)
```

- **Feedback Period**: Reward presented for N timesteps (simulates sustained neural response)
- **Decision Delay**: Optional pause before decision (models deliberation)
- **RNN Processing**: Continuous hidden state updates throughout

## How to Use

### Quick Start

```bash
# 1. Install PyTorch
pip install torch

# 2. Run quick test
python test_rnn_agent.py

# 3. Open notebook for full analysis
jupyter notebook results.ipynb
```

### Training Example

```python
from rnn_agent import RNNAgent, RNNAgentRecorder
import gymnasium as gym

# Create environment
env = gym.make("zsombi/monkey-bandit-task-v0", n_arms=3)

# Create agent
agent = RNNAgent(
    n_arms=3,
    hidden_size=32,
    learning_rate=0.001,
    gamma=0.95,
    feedback_duration=3,
    decision_delay=1
)

# Training loop
recorder = RNNAgentRecorder()
obs, info = env.reset()
agent.reset()
last_reward = 0

for step in range(50000):
    action, switched = agent.act(last_reward)
    obs, reward, _, done, info = env.step(action)
    agent.store_reward(reward)
    
    # Record
    switch_prob = agent.get_switch_probability(last_reward)
    recorder.record(action, reward, info, agent, switched, switch_prob)
    
    last_reward = reward
    
    # Update every 100 steps
    if step % 100 == 0:
        loss = agent.update()

# Analyze
rec = recorder.get_recording()
print(f"Mean reward: {rec['reward'].mean():.3f}")
```

## Customization Examples

### 1. Use LSTM Instead of GRU

In `rnn_agent.py`, change the RNN type:
```python
self.rnn = nn.LSTM(
    input_size=self.input_size,
    hidden_size=hidden_size,
    num_layers=1,
    batch_first=True
)
```

### 2. Add More Input Features

```python
# Expand input to include action history
self.input_size = 4  # reward, prev_switch, last_action, trial_in_block

# In act():
network_input = torch.FloatTensor([[
    reward,
    self.previous_switch,
    self.last_action / self.n_arms,
    (step % 40) / 40
]])
```

### 3. Implement Actor-Critic

```python
class ActorCriticRNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.GRU(input_size=2, hidden_size=hidden_size)
        self.actor = nn.Linear(hidden_size, 1)   # Policy
        self.critic = nn.Linear(hidden_size, 1)  # Value
    
    def forward(self, x, hidden=None):
        rnn_out, hidden = self.rnn(x, hidden)
        switch_prob = torch.sigmoid(self.actor(rnn_out[:, -1, :]))
        value = self.critic(rnn_out[:, -1, :])
        return switch_prob, value, hidden
```

## Expected Results

After training for ~50,000 steps:

- **Mean reward**: 0.55 - 0.65 (depends on task difficulty)
- **Switch rate**: 0.2 - 0.4 (adapts to task)
- **Switch after reward**: Low probability (~0.2)
- **Switch after no reward**: High probability (~0.7)

The agent should learn to:
1. Stay with rewarding actions
2. Switch away from unrewarding actions
3. Adapt within blocks as reward contingencies change
4. Achieve similar or better performance than rule-based agent

## Analysis Capabilities

The implementation includes tools for:

1. **Performance Analysis**
   - Learning curves
   - Block-wise performance
   - Switch behavior by outcome

2. **Hidden State Analysis**
   - PCA visualization of hidden states
   - t-SNE/UMAP clustering
   - Temporal dynamics tracking

3. **Strategy Comparison**
   - Multi-agent comparisons
   - Trial-by-trial detailed analysis
   - Switch probability landscapes

## Next Steps

### Research Extensions

1. **Test Generalization**
   - Train on 3-armed task, test on 4-armed
   - Vary reward probabilities
   - Change block lengths

2. **Architectural Variations**
   - Compare GRU vs LSTM vs vanilla RNN
   - Multi-layer networks
   - Attention mechanisms

3. **Learning Algorithms**
   - Actor-Critic for variance reduction
   - Proximal Policy Optimization (PPO)
   - Experience replay

4. **Neuroscience Connections**
   - Compare hidden states to neural recordings
   - Analyze learned representations
   - Model fitting to behavioral data

## File Structure

```
notebooks/dqn/
├── rnn_agent.py              # Main implementation
├── rnn_analysis_tools.py     # Analysis utilities
├── test_rnn_agent.py         # Quick test script
├── results.ipynb             # Training notebook
├── RNN_AGENT_README.md       # Documentation
├── requirements_rnn.txt      # Dependencies
└── IMPLEMENTATION_SUMMARY.md # This file
```

## Technical Details

### Network Architecture
- Input layer: 2D (reward, previous_switch)
- RNN layer: GRU with 32 hidden units
- Output layer: 1D through sigmoid (switch probability)
- Total parameters: ~3,000 (depends on hidden_size)

### Training
- Algorithm: REINFORCE (policy gradient)
- Optimizer: Adam with learning rate 0.001
- Gradient clipping: Max norm = 1.0
- Return normalization: Z-score normalization per episode

### Computational Requirements
- Training time: ~2-5 minutes for 50k steps (CPU)
- Memory: < 100 MB
- No GPU required (but can use if available)

## Troubleshooting

### Common Issues

1. **PyTorch not found**: `pip install torch`
2. **Poor performance**: Increase training duration or adjust learning rate
3. **Unstable training**: Decrease learning rate, increase update interval
4. **Not learning**: Check that rewards are being stored and returns computed correctly

## References

### Implemented Methods
- REINFORCE: Williams (1992)
- GRU: Cho et al. (2014)
- Policy gradient: Sutton & Barto (2018)

### Relevant Research
- Song et al. (2017): Training RNNs for cognitive tasks
- Wang et al. (2018): Meta-RL with RNNs
- Mante et al. (2013): Context-dependent computation in RNNs

---

**Created by**: GitHub Copilot
**Date**: November 10, 2025
**Project**: PoPy (Prefrontal cortex Oscillations in Python)
