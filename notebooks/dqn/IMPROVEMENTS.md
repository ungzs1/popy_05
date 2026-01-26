# Summary of Improvements to RNN Agent

## Critical Fixes

### 1. BPTT Was Broken ❌ → ✅
**Before:**
```python
self.network.hidden = hidden_new.detach()  # Called EVERY trial!
# ❌ Gradients blocked across trials → can't learn temporal dependencies
```

**After:**
```python
# Only detach at truncation boundaries (every K trials)
if agent.should_update():
    agent.update()  # Backprop through K trials
    self.hidden = self.hidden.detach()  # NOW detach
```

**Impact**: Agent can now learn multi-trial strategies (e.g., "explore for 5 trials, then exploit").

---

### 2. Sampling/Log-Prob Mismatch ❌ → ✅
**Before:**
```python
# Add noise to probability
noisy_switch_prob = np.clip(switch_prob + np.random.normal(0, 0.1), 0, 1)
should_switch = np.random.random() < noisy_switch_prob

# But compute log_prob from UN-NOISED probability! 😱
log_prob = torch.log(switch_prob + 1e-10)  # ❌ WRONG!
```

**After:**
```python
# Sample from the actual distribution
dist = Bernoulli(probs=switch_prob)
should_switch = dist.sample()
log_prob = dist.log_prob(should_switch)  # ✅ CORRECT!
```

**Impact**: Policy gradient is now unbiased. Previous version was learning the wrong gradient.

---

### 3. Decision Delay Bug ❌ → ✅
**Before (in analysis branch):**
```python
rnn_out, hidden_new = self.network.rnn(network_input, hidden)
# Compute switch_prob from feedback-phase hidden state
switch_logit = self.network.fc_out(rnn_out[:, -1, :])

# THEN add delay
if self.decision_delay > 0:
    delay_input = torch.zeros(1, self.decision_delay, 2)
    rnn_out_delay, hidden_new = self.network.rnn(delay_input, hidden_new)
    # ❌ But switch_prob is already computed! Doesn't use delay!
```

**After:**
```python
# Concatenate feedback + delay into single sequence
if self.decision_delay > 0:
    delay_input = torch.zeros(1, self.decision_delay, input_size, device=device)
    network_input = torch.cat([network_input, delay_input], dim=1)

# Forward through ENTIRE sequence
outputs = self.network.forward_sequence(network_input, hidden)
switch_prob = outputs['switch_probs'][:, -1, :]  # ✅ From final timestep
```

**Impact**: Analysis branch now computes switch probability from the correct hidden state.

---

### 4. Inappropriate Discounting ❌ → ✅
**Before:**
```python
gamma = 0.99  # ❌ Unnecessary for bandits
```

**After:**
```python
gamma = 1.0  # ✅ Bandits are non-sequential (no delayed rewards)
```

**Impact**: Simpler credit assignment; better for bandit tasks.

---

### 5. High-Variance REINFORCE ❌ → ✅
**Before:**
```python
# Pure REINFORCE: -log π(a|s) * R
policy_loss = -log_prob * R
# ❌ High variance → slow/unstable learning
```

**After:**
```python
# Actor-critic with value baseline
advantage = R - V(s).detach()  # ✅ Variance reduction
policy_loss = -log_prob * advantage
value_loss = (V(s) - R)^2
total_loss = policy_loss + value_coef * value_loss - entropy_coef * H(π)
```

**Impact**: Faster, more stable learning; added value head to network.

---

## Major Enhancements

### 6. Tiny Architecture for Interpretability
**Before:**
```python
hidden_size = 32  # Too large for fixed-point analysis
input = [reward, previous_switch]  # Loses arm identity
```

**After:**
```python
hidden_size = 4  # Tiny! Can visualize trajectories, find fixed points
input = [reward, prev_action_onehot]  # Preserves which arm was chosen
```

**Impact**: Can use dynamical systems tools (fixed points, Jacobian, vector fields).

---

### 7. Entropy Regularization
**Before:**
```python
# Exploration via external noise (caused log_prob mismatch)
noisy_switch_prob = switch_prob + np.random.normal(0, 0.1)
```

**After:**
```python
# Entropy bonus in loss function
entropy = dist.entropy()
loss -= entropy_coef * entropy  # Encourages exploration
```

**Impact**: Principled exploration; prevents premature convergence.

---

### 8. Analysis Tools
**Before:**
- No analysis utilities

**After:**
- `RNNAnalyzer` class:
  - `find_fixed_points()`: Discover attractor states
  - `compute_jacobian_at_fixed_point()`: Stability analysis
  - `plot_vector_field_2d()`: Visualize dynamics
  - `plot_trajectory()`: Show hidden state evolution
  - `compute_behavioral_stats()`: P(switch|reward), etc.
- Plotting utilities:
  - `plot_learning_curves()`
  - `plot_strategy_analysis()`

**Impact**: Can interpret what strategy the agent learned (WSLS? perseveration? inference?).

---

### 9. Device Handling & Reproducibility
**Before:**
```python
self.hidden = torch.zeros(1, batch_size, hidden_size)  # ❌ Always CPU
# No seeding, no save/load
```

**After:**
```python
self.hidden = torch.zeros(1, batch_size, hidden_size, device=device)
set_seed(42)  # Reproducibility
agent.save('checkpoint.pth')  # Checkpointing
agent.load('checkpoint.pth')
```

**Impact**: Can use GPU; reproducible experiments; can resume training.

---

## Comparison Table

| Feature | Before (v1.0) | After (v2.0) |
|---------|---------------|--------------|
| **BPTT** | Broken (detach every trial) | ✅ Truncated BPTT (every K trials) |
| **Sampling** | Noisy probability | ✅ torch.distributions |
| **Log-prob** | Mismatched | ✅ Aligned with sampling |
| **Baseline** | None (pure REINFORCE) | ✅ Value head (actor-critic) |
| **Exploration** | External noise | ✅ Entropy bonus |
| **Hidden size** | 32 (hard to interpret) | ✅ 4 (tiny, interpretable) |
| **Input** | [reward, prev_switch] | ✅ [reward, prev_action_onehot] |
| **Gamma** | 0.99 | ✅ 1.0 (for bandits) |
| **Analysis tools** | None | ✅ Fixed points, Jacobian, plots |
| **Device** | CPU only | ✅ CPU/GPU |
| **Seeding** | None | ✅ set_seed() |
| **Save/load** | None | ✅ Checkpointing |
| **Decision delay bug** | Yes | ✅ Fixed |

---

## Performance Impact

**Training stability**: 🔴 Poor → 🟢 Good
- Value baseline reduces variance
- Entropy prevents collapse
- Gradient clipping prevents explosions

**Learning speed**: 🔴 Slow → 🟢 Fast
- BPTT learns temporal dependencies
- Correct gradients (no sampling bias)

**Interpretability**: 🔴 None → 🟢 Excellent
- Tiny hidden size (4 units)
- Fixed-point analysis
- Trajectory visualization
- Vector field plots (2D)

**Reproducibility**: 🔴 None → 🟢 Full
- Seeding all RNGs
- Checkpointing
- Device handling

---

## Usage Changes

### Training Loop
**Before:**
```python
for trial in range(n_trials):
    action, switched = agent.act(reward)
    agent.store_reward(reward)
    
agent.update()  # ❌ Only at end
```

**After:**
```python
for trial in range(n_trials):
    action, switched = agent.act(reward)
    agent.store_reward(reward)
    
    if agent.should_update():  # ✅ Every K trials
        agent.update()
        
agent.update(force=True)  # Final update
```

### Configuration
**Before:**
```python
agent = RNNAgent(
    n_arms=3,
    hidden_size=32,
    learning_rate=0.001,
    gamma=0.99,
    exploration_noise=0.1
)
```

**After:**
```python
agent = RNNAgent(
    n_arms=3,
    hidden_size=4,           # ✅ Tiny!
    rnn_type='GRU',          # ✅ Configurable
    use_action_input=True,   # ✅ Better input
    learning_rate=0.001,
    gamma=1.0,               # ✅ Correct for bandits
    entropy_coef=0.01,       # ✅ Entropy instead of noise
    value_coef=0.5,          # ✅ Actor-critic
    bptt_truncation=20,      # ✅ Truncated BPTT
    device='cpu',            # ✅ Device handling
    seed=42                  # ✅ Reproducibility
)
```

---

## Testing

Run the example:
```bash
python notebooks/dqn/example_tiny_rnn_usage.py
```

Expected output:
- Agent learns to prefer best arm (reward rate ~70-80%)
- P(stay | reward) > P(stay | no reward) (win-stay-lose-shift)
- Smooth learning curves
- Stable fixed points under different input conditions
- Interpretable hidden state trajectories

---

## Next Steps

1. **For interpretability**: Use hidden_size=2, plot vector fields
2. **For performance**: Use hidden_size=8-16
3. **For strategy discovery**: Fit to real behavior data, analyze fixed points
4. **For comparison**: Run multiple seeds, compare discovered strategies
5. **For advanced analysis**: Add spectral regularization, low-rank parameterization

---

## Questions?

See `README_tiny_rnn.md` for full documentation.
