# Getting Started with RNN Agent

## Quick Checklist

- [ ] Install PyTorch: `pip install torch`
- [ ] Verify installation: Run `test_rnn_agent.py`
- [ ] Open notebook: `results.ipynb`
- [ ] Run training cells
- [ ] Analyze results
- [ ] Experiment with parameters

## Detailed Setup

### Step 1: Install Dependencies

```bash
# Option A: Install PyTorch only
pip install torch

# Option B: Install all requirements
pip install -r requirements_rnn.txt

# Option C: Use conda
conda install pytorch -c pytorch
```

### Step 2: Verify Installation

Run the test script:
```bash
cd notebooks/dqn
python test_rnn_agent.py
```

You should see:
```
✓ PyTorch 2.x.x installed
✓ Environment created
✓ Agent initialized
✓ Training complete
✓ Plot saved
```

### Step 3: Open the Notebook

```bash
jupyter notebook results.ipynb
```

Or in VS Code:
1. Open `results.ipynb`
2. Select Python kernel
3. Run cells sequentially

### Step 4: Run Training

Execute the cells in this order:

1. **Imports** (cell with RNNAgent import)
2. **Initialize Agent** (create agent with parameters)
3. **Training Loop** (run 50k steps)
4. **View Results** (analyze performance)
5. **Plot Comparisons** (compare with ForagingAgent)

### Step 5: Customize (Optional)

Try different parameters:

```python
# More aggressive learning
rnn_agent = RNNAgent(
    hidden_size=64,        # Larger network
    learning_rate=0.003,   # Faster learning
    exploration_noise=0.1  # More exploration
)

# More realistic temporal dynamics
rnn_agent = RNNAgent(
    feedback_duration=5,   # Longer feedback
    decision_delay=2       # More delay
)
```

## File Guide

### Essential Files
- **`rnn_agent.py`**: Main implementation (start here to understand code)
- **`results.ipynb`**: Training notebook (start here to run experiments)
- **`test_rnn_agent.py`**: Quick test (start here to verify installation)

### Documentation
- **`RNN_AGENT_README.md`**: Complete documentation
- **`IMPLEMENTATION_SUMMARY.md`**: Overview and design decisions
- **`GETTING_STARTED.md`**: This file

### Utilities
- **`rnn_analysis_tools.py`**: Advanced analysis functions
- **`draw_architecture.py`**: Generate architecture diagrams
- **`requirements_rnn.txt`**: Dependency list

## Learning Path

### Beginner
1. Run `test_rnn_agent.py` to see the agent in action
2. Open `results.ipynb` and run all cells
3. Compare RNN agent vs ForagingAgent performance
4. Read `RNN_AGENT_README.md` for concepts

### Intermediate
1. Modify hyperparameters in notebook
2. Try different training durations
3. Use `rnn_analysis_tools.py` to analyze hidden states
4. Generate architecture diagrams with `draw_architecture.py`

### Advanced
1. Modify network architecture in `rnn_agent.py`
2. Implement actor-critic variant
3. Add memory replay buffer
4. Test on different task variants
5. Compare to neural recordings

## Troubleshooting

### Problem: "Import torch could not be resolved"
**Solution**: Install PyTorch
```bash
pip install torch
```

### Problem: Poor performance (mean reward < 0.5)
**Solution**: Train longer or adjust parameters
```python
# Increase training duration
n_runs = 100_000  # instead of 50_000

# Or adjust learning rate
learning_rate=0.003  # instead of 0.001
```

### Problem: Training is unstable
**Solution**: Reduce learning rate or increase update interval
```python
rnn_agent = RNNAgent(
    learning_rate=0.0005,  # Lower learning rate
)
# And in training loop:
update_interval = 200  # Update less frequently
```

### Problem: Agent not learning
**Solution**: Check reward collection
```python
# Make sure this is in your training loop:
agent.store_reward(reward)  # Store every reward
agent.update()  # Update periodically
```

### Problem: Notebook kernel crashes
**Solution**: Reduce batch size or network size
```python
rnn_agent = RNNAgent(
    hidden_size=16,  # Smaller network
)
update_interval = 50  # Update more frequently with smaller batches
```

## Expected Timeline

- **Installation**: 5 minutes
- **Test run**: 2-3 minutes
- **Full training (50k steps)**: 2-5 minutes
- **Analysis**: 10-15 minutes
- **Customization**: Variable

## Key Results to Check

After training, verify:

1. **Learning curve**: Should show improvement over time
2. **Mean reward**: Should be 0.55-0.65
3. **Switch behavior**: Should be outcome-dependent
   - Low switch prob after reward (~0.2)
   - High switch prob after no reward (~0.7)
4. **Block adaptation**: Performance should improve within blocks

## Next Experiments

Once you have the basic agent working:

### 1. Parameter Sensitivity
```python
# Test different learning rates
for lr in [0.0001, 0.001, 0.01]:
    agent = RNNAgent(learning_rate=lr)
    # ... train and compare
```

### 2. Architecture Variants
```python
# Test different hidden sizes
for h in [16, 32, 64, 128]:
    agent = RNNAgent(hidden_size=h)
    # ... train and compare
```

### 3. Temporal Dynamics
```python
# Test different feedback durations
for fd in [1, 3, 5, 10]:
    agent = RNNAgent(feedback_duration=fd)
    # ... train and compare
```

### 4. Task Difficulty
```python
# Test on different task variants
env_easy = gym.make("zsombi/monkey-bandit-task-v0", n_arms=2)
env_hard = gym.make("zsombi/monkey-bandit-task-v0", n_arms=4)
```

## Resources

### Code Documentation
- All functions have detailed docstrings
- Read the code comments for implementation details
- Check type hints for input/output specifications

### Neuroscience Background
- Song et al. (2017): RNNs for cognitive tasks
- Wang et al. (2018): Meta-RL with RNNs
- Mante et al. (2013): Context-dependent computation

### RL Theory
- Sutton & Barto (2018): RL textbook
- Williams (1992): REINFORCE algorithm
- Schulman et al. (2017): PPO (advanced)

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review the troubleshooting section above
3. Look at example code in `test_rnn_agent.py`
4. Check parameters are in reasonable ranges
5. Verify your environment has all dependencies

## Common Questions

**Q: Why use an RNN for this task?**
A: RNNs can learn temporal dependencies and maintain memory of past events, similar to neural circuits.

**Q: How long should I train?**
A: 50k steps is usually sufficient. Monitor the learning curve to decide.

**Q: Can I use a GPU?**
A: Yes, but it's not necessary for this task size. PyTorch will use GPU if available.

**Q: How do I save my trained agent?**
A: Add this code after training:
```python
torch.save(rnn_agent.network.state_dict(), 'trained_agent.pth')
```

**Q: How do I load a saved agent?**
A: 
```python
rnn_agent.network.load_state_dict(torch.load('trained_agent.pth'))
```

**Q: Why are results slightly different each time?**
A: Due to random initialization and stochastic training. This is normal.

**Q: Can I compare to real neural data?**
A: Yes! Extract hidden states with `rnn_analysis_tools.py` and compare to recordings.

## Success Criteria

You'll know it's working when:

- ✓ Agent learns to get more rewards over time
- ✓ Switch probability is higher after no-reward
- ✓ Switch probability is lower after reward
- ✓ Performance matches or exceeds rule-based agent
- ✓ Behavior adapts within blocks

## Final Tips

1. **Start simple**: Use default parameters first
2. **Monitor learning**: Check plots frequently
3. **Be patient**: Learning takes time (50k steps)
4. **Experiment**: Try different parameters
5. **Compare**: Always compare to baselines
6. **Document**: Keep notes on what works
7. **Visualize**: Use the analysis tools

---

**Ready to start? Run:**
```bash
python test_rnn_agent.py
```

Then open `results.ipynb` and start experimenting!

Good luck! 🚀
