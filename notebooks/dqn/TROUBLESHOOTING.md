# Troubleshooting Guide

## Common Issues and Solutions

### 1. RuntimeError: "one of the variables needed for gradient computation has been modified"

**Error Message:**
```
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
```

**Cause:** This happens when PyTorch's autograd graph is broken by in-place operations on tensors that are part of the computation graph.

**Solution:**
1. **Restart the kernel**: In VS Code or Jupyter, go to Kernel → Restart Kernel
2. **Re-run all cells from the beginning** in order
3. Make sure you have the latest version of `rnn_agent.py` (the hidden state should be detached with `.detach()`)

**Prevention:**
- Always run notebook cells in order
- Don't re-run training cells without restarting
- The fix in `rnn_agent.py` detaches the hidden state to prevent this issue

---

### 2. ImportError: "No module named 'torch'"

**Error Message:**
```
ImportError: No module named 'torch'
```

**Solution:**
```bash
pip install torch
```

Or with conda:
```bash
conda install pytorch -c pytorch
```

**Verify installation:**
```python
import torch
print(torch.__version__)
```

---

### 3. Training produces poor results (mean reward < 0.5)

**Possible Causes:**
- Not enough training steps
- Learning rate too high or too low
- Network too small

**Solutions:**

**A. Increase training duration:**
```python
n_runs = 100_000  # instead of 50_000
```

**B. Adjust learning rate:**
```python
rnn_agent = RNNAgent(
    learning_rate=0.003  # try 0.001, 0.003, or 0.01
)
```

**C. Increase network size:**
```python
rnn_agent = RNNAgent(
    hidden_size=64  # or 128
)
```

---

### 4. Training is unstable or loss explodes

**Symptoms:**
- Loss values increase instead of decrease
- NaN values in outputs
- Agent performance degrades over time

**Solutions:**

**A. Reduce learning rate:**
```python
rnn_agent = RNNAgent(
    learning_rate=0.0005  # lower than default
)
```

**B. Update less frequently:**
```python
update_interval = 200  # instead of 100
```

**C. Check gradient clipping (already implemented):**
The code already includes gradient clipping, but you can adjust it in `rnn_agent.py`:
```python
torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
```

---

### 5. Kernel crashes or runs out of memory

**Symptoms:**
- Kernel dies during training
- "Out of memory" errors

**Solutions:**

**A. Reduce network size:**
```python
rnn_agent = RNNAgent(
    hidden_size=16  # smaller network
)
```

**B. Update more frequently (smaller batches):**
```python
update_interval = 50  # smaller update interval
```

**C. Reduce total steps:**
```python
n_runs = 25_000  # fewer training steps
```

---

### 6. Agent doesn't learn (flat learning curve)

**Symptoms:**
- Reward rate doesn't improve over time
- Switch probability doesn't change
- Loss stays constant

**Debugging Steps:**

**A. Verify rewards are being stored:**
```python
# Add this after the training loop
print(f"Rewards collected: {len(rnn_agent.rewards)}")
print(f"Log probs collected: {len(rnn_agent.saved_log_probs)}")
```

**B. Check if updates are happening:**
```python
# Add this in the training loop
if step_count % update_interval == 0:
    loss = rnn_agent.update()
    print(f"Update at step {step_count}, loss: {loss:.4f}")  # Add this line
```

**C. Increase learning rate:**
```python
rnn_agent = RNNAgent(
    learning_rate=0.005  # higher learning rate
)
```

**D. Reduce exploration noise:**
```python
rnn_agent = RNNAgent(
    exploration_noise=0.01  # less noise = clearer signal
)
```

---

### 7. "AttributeError" or "KeyError" when analyzing results

**Error Examples:**
```
KeyError: 'switch_prob'
AttributeError: 'DataFrame' object has no attribute 'switched'
```

**Cause:** Recording wasn't done properly or cells weren't run in order.

**Solution:**
1. Make sure the training cell completed successfully
2. Check that `recorder.record()` is being called in the training loop
3. Verify you're using `RNNAgentRecorder`, not `MakeRecording`

---

### 8. Results differ between runs

**Symptoms:**
- Different performance each time
- Can't reproduce results

**Cause:** This is normal! Neural networks have random initialization and stochastic training.

**Solutions (for reproducibility):**

```python
# Set random seeds at the beginning
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Then create agent and train
```

---

### 9. Notebook cells appear out of order or broken

**Solution:**
1. Close and reopen the notebook
2. Restart kernel
3. Clear all outputs: Edit → Clear All Outputs
4. Run all cells from top to bottom

---

### 10. Can't import from `rnn_agent.py`

**Error:**
```
ModuleNotFoundError: No module named 'rnn_agent'
```

**Solution:**
Make sure you're running the notebook from the correct directory:
```bash
cd /path/to/PoPy/notebooks/dqn
jupyter notebook results.ipynb
```

Or in VS Code, make sure the notebook's working directory is set correctly.

---

## Quick Diagnosis Checklist

Run this cell to check your setup:

```python
# Diagnostic cell
import sys
print("Python version:", sys.version)

try:
    import torch
    print("✓ PyTorch installed:", torch.__version__)
except ImportError:
    print("✗ PyTorch NOT installed - run: pip install torch")

try:
    from rnn_agent import RNNAgent
    print("✓ rnn_agent.py found")
except ImportError:
    print("✗ rnn_agent.py NOT found - check working directory")

try:
    import gymnasium as gym
    print("✓ gymnasium installed")
except ImportError:
    print("✗ gymnasium NOT installed - run: pip install gymnasium")

try:
    from popy.simulation_tools import MonkeyBanditTask
    print("✓ popy package found")
except ImportError:
    print("✗ popy package NOT found - check PYTHONPATH")

print("\nIf all items show ✓, you're ready to go!")
```

---

## Best Practices to Avoid Issues

1. **Always restart kernel before a full training run**
   - This clears all variables and state

2. **Run cells in order**
   - Don't skip cells or run them out of sequence

3. **Monitor training progress**
   - Check the printed output during training
   - Look for the "Episode X/500" messages

4. **Start with default parameters**
   - Only change parameters after you have a working baseline

5. **Save your work frequently**
   - Save the notebook regularly
   - Consider saving trained models

6. **Use version control**
   - Commit working versions
   - Can revert if something breaks

---

## Still Having Issues?

If none of the above solutions work:

1. **Check the error message carefully**
   - Read the full traceback
   - Look for the line number where it fails

2. **Verify file integrity**
   - Make sure `rnn_agent.py` hasn't been corrupted
   - Re-download if needed

3. **Check Python environment**
   - Are you using the correct conda/virtual environment?
   - Try creating a fresh environment

4. **Simplify the problem**
   - Run `test_rnn_agent.py` first to verify basic functionality
   - Then try the notebook

5. **Look at the example output**
   - Check that your output matches the expected format
   - Compare with the documentation

---

## Emergency Reset

If everything is broken, start fresh:

```bash
# 1. Save any important results
# 2. Close Jupyter/VS Code
# 3. Reinstall dependencies
pip install --force-reinstall torch numpy pandas matplotlib gymnasium

# 4. Restart your editor
# 5. Open notebook
# 6. Restart kernel
# 7. Run all cells from top
```

---

## Getting Help

When reporting an issue, include:

1. **Full error message** (entire traceback)
2. **Code that caused the error**
3. **Python version** and **PyTorch version**
4. **What you've already tried**
5. **When the error occurs** (which cell)

This will help diagnose the problem quickly!
