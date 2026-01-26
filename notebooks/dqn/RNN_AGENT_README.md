# RNN Agent for 3-Armed Bandit Task

## Overview

This implementation provides an RNN-based agent that learns to solve the 3-armed bandit task using reinforcement learning. Unlike the rule-based `ForagingAgent` that uses explicit delta learning rules and threshold comparisons, this agent learns the stay/switch strategy end-to-end through experience.

## Architecture

### Input
The RNN receives a 2-dimensional input vector at each timestep:
- **Reward** (0 or 1): Binary feedback from the previous action
- **Previous Switch** (0 or 1): Whether the agent switched on the previous trial

### Network Structure
```
Input (2D) → GRU/RNN → Hidden State (32D) → Fully Connected → Sigmoid → Switch Probability
```

- **RNN Layer**: GRU (Gated Recurrent Unit) with 32 hidden units
  - Processes temporal sequences of rewards
  - Maintains memory of past outcomes
  - Can be replaced with LSTM or vanilla RNN

- **Output Layer**: Single sigmoid neuron
  - Outputs probability of switching (0 to 1)
  - 1 = switch to random alternative action
  - 0 = stay with previous action

### Temporal Dynamics

The implementation includes temporal processing to model neural/cognitive delays:

1. **Feedback Duration** (default: 3 timesteps)
   - The reward signal is presented for multiple timesteps
   - Simulates sustained neural responses to feedback
   - The RNN processes this temporal input

2. **Decision Delay** (default: 1 timestep)
   - Optional delay period between feedback and decision
   - During delay, input is zeros (models deliberation time)
   - RNN continues processing internal representations

### Decision Making

1. Network outputs switch probability `p_switch`
2. Add exploration noise: `p_noisy = p_switch + N(0, σ)`
3. Sample action: 
   - With probability `p_noisy`: **Switch** → choose random alternative
   - With probability `1 - p_noisy`: **Stay** → repeat previous action

## Learning Algorithm

The agent uses **REINFORCE** (policy gradient method):

### Training Loop
```
for each trial:
    1. Observe reward from previous action
    2. RNN computes switch probability
    3. Sample action (stay or switch)
    4. Store log probability and reward
    
every N trials:
    1. Compute discounted returns
    2. Calculate policy gradient loss: -∑(log π(a) × R)
    3. Backpropagate through RNN
    4. Update weights with gradient descent
```

### Key Parameters
- **Learning rate** (α): 0.001
- **Discount factor** (γ): 0.95 - balances immediate vs future rewards
- **Update interval**: 100 trials between network updates
- **Exploration noise** (σ): 0.05 - adds stochasticity to decisions

## Usage

### Quick Start: Train and Analyze

**Step 1: Train the agent**
```bash
python train_rnn_agent.py
```

This will:
- Train for 50,000 steps
- Save the trained model to `results/rnn_agent_trained.pth`
- Run a 100,000 trial simulation
- Save all results to `results/`

**Step 2: Analyze results**
```bash
jupyter notebook results.ipynb
```

Then run all cells to visualize training curves and performance metrics.

### Manual Training (Advanced)
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

# Save model
import torch
torch.save(agent.network.state_dict(), 'my_trained_agent.pth')

# Analyze
rec = recorder.get_recording()
print(f"Mean reward: {rec['reward'].mean():.3f}")
```

## Comparison with Rule-Based Agent

### ForagingAgent (Rule-Based)
- Explicit value computation: `V += α(reward - V)`
- Deterministic threshold comparison: `p_switch = σ(β(V - V₀))`
- Hand-crafted switching logic
- Parameters must be manually tuned

### RNNAgent (Learning-Based)
- End-to-end learning from experience
- Discovers optimal strategy through trial and error
- Flexible temporal processing
- Can learn complex, non-linear strategies
- Adapts to task statistics automatically

## Customization

### Change RNN Type
In `rnn_agent.py`, replace GRU with LSTM:
```python
self.rnn = nn.LSTM(
    input_size=self.input_size,
    hidden_size=hidden_size,
    num_layers=1,
    batch_first=True
)
```

### Add More Input Features
Expand the input to include additional information:
```python
# In __init__:
self.input_size = 4  # reward, previous_switch, action_history, etc.

# In act():
network_input = torch.FloatTensor([[
    reward, 
    self.previous_switch,
    self.last_action / self.n_arms,  # normalized
    self.trial_count % 40 / 40  # position in block
]])
```

### Multi-Layer RNN
```python
self.rnn = nn.GRU(
    input_size=self.input_size,
    hidden_size=hidden_size,
    num_layers=3,  # Stack 3 GRU layers
    batch_first=True,
    dropout=0.2  # Add dropout between layers
)
```

### Add Value Learning (Actor-Critic)
```python
class ActorCriticRNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, 1)  # Policy (switch prob)
        self.critic = nn.Linear(hidden_size, 1)  # Value estimate
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hidden=None):
        rnn_out, hidden = self.rnn(x, hidden)
        rnn_out_last = rnn_out[:, -1, :]
        
        switch_prob = self.sigmoid(self.actor(rnn_out_last))
        value = self.critic(rnn_out_last)
        
        return switch_prob, value, hidden
```

## Expected Behavior

### Learning Trajectory
- **Early training** (0-10k trials): Random exploration, unstable rewards
- **Mid training** (10k-30k trials): Strategy emergence, improving rewards
- **Late training** (30k+ trials): Stable strategy, near-optimal performance

### Learned Strategy
The RNN should learn to:
1. **Stay after reward**: Low switch probability when `reward = 1`
2. **Switch after no reward**: High switch probability when `reward = 0`
3. **Track block structure**: Adapt switch behavior based on recent history
4. **Exploit best arm**: Converge to high-reward actions within blocks

### Performance Metrics
- Mean reward: ~0.55-0.65 (depends on task difficulty)
- Switch rate: ~0.2-0.4 (adapts to task statistics)
- Block-wise adaptation: Increasing accuracy within blocks

## Troubleshooting

### Poor Performance
- Increase training duration (more trials)
- Adjust learning rate (try 0.0001 - 0.01)
- Increase hidden size (try 64 or 128)
- Reduce exploration noise
- Increase update interval for more stable gradients

### Unstable Training
- Decrease learning rate
- Add gradient clipping (already implemented)
- Normalize returns (already implemented)
- Use smaller batch sizes for updates

### Not Learning
- Check that rewards are being stored correctly
- Verify policy gradients are non-zero
- Ensure RNN hidden state is maintained across trials
- Check for vanishing gradients (try LSTM instead of GRU)

## Dependencies

```bash
pip install torch numpy pandas matplotlib gymnasium
```

## Files

- `rnn_agent.py`: Main implementation (RNNAgent, SwitchRNN, RNNAgentRecorder)
- `results.ipynb`: Training and analysis notebook
- `agents.py`: Rule-based ForagingAgent for comparison

## References

### Reinforcement Learning
- Sutton & Barto (2018). Reinforcement Learning: An Introduction
- Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning

### RNNs for Decision Making
- Song et al. (2017). Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks
- Wang et al. (2018). Prefrontal cortex as a meta-reinforcement learning system
- Mante et al. (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex

## License

Part of the PoPy project. See main repository for license information.
