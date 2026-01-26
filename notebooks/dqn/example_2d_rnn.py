"""
Quick example for using a 2D hidden state RNN (maximum interpretability).

With only 2 hidden units, you can:
1. Visualize the full phase portrait (vector field)
2. See trajectories in 2D space (no PCA needed)
3. Identify separatrices (decision boundaries)
4. Interpret fixed points as behavioral modes

This is the most interpretable configuration!
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from notebooks.dqn.rnn_agent import (
    RNNAgent, RNNAgentRecorder, RNNAnalyzer, set_seed
)


class SimpleBandit:
    """Simple 3-armed bandit for testing."""
    def __init__(self, probs=[0.8, 0.3, 0.3], seed=42):
        np.random.seed(seed)
        self.probs = np.array(probs)
        self.trial_id = 0
        
    def reset(self):
        self.trial_id = 0
        return {'trial_id': 0, 'best_arm': 0}
    
    def step(self, action):
        reward = int(np.random.random() < self.probs[action])
        self.trial_id += 1
        return reward, {'trial_id': self.trial_id, 'best_arm': 0}


def main():
    print("=" * 70)
    print("Ultra-Tiny RNN: 2D Hidden State for Maximum Interpretability")
    print("=" * 70)
    
    set_seed(42)
    
    # Create environment
    env = SimpleBandit(probs=[0.8, 0.3, 0.3], seed=42)
    
    # Create agent with 2D hidden state
    print("\n1. Creating 2D RNN agent...")
    agent = RNNAgent(
        n_arms=3,
        hidden_size=2,          # ⭐ 2D for full visualization!
        rnn_type='RNN',         # Vanilla RNN (simpler dynamics than GRU)
        use_action_input=True,
        learning_rate=0.001,
        gamma=1.0,
        entropy_coef=0.02,      # Higher entropy for exploration
        value_coef=0.5,
        bptt_truncation=20,
        device='cpu',
        seed=42
    )
    print(f"   Hidden size: {agent.network.hidden_size}")
    print(f"   RNN type: {agent.network.rnn_type}")
    
    # Train
    print("\n2. Training agent...")
    recorder = RNNAgentRecorder()
    agent.reset()
    info = env.reset()
    action = agent.last_action
    
    for trial in range(2000):
        reward, info = env.step(action)
        action, switched = agent.act(reward)
        agent.store_reward(reward)
        
        switch_prob = agent.get_switch_probability(reward)
        recorder.record(action, reward, info, agent, switched, switch_prob)
        
        if agent.should_update():
            metrics = agent.update()
            if trial % 200 == 0:
                print(f"   Trial {trial}: Loss={metrics['loss']:.3f}, "
                      f"Entropy={metrics['entropy']:.4f}")
    
    agent.update(force=True)
    recording_df = recorder.get_recording()
    
    # Compute stats
    print("\n3. Behavioral statistics:")
    analyzer = RNNAnalyzer(agent)
    stats = analyzer.compute_behavioral_stats(recording_df)
    print(f"   - P(stay | reward):        {stats['p_stay_given_reward']:.2%}")
    print(f"   - P(switch | no reward):   {stats['p_switch_given_no_reward']:.2%}")
    print(f"   - Overall reward rate:     {stats['reward_rate']:.2%}")
    
    # Find fixed points
    print("\n4. Fixed-point analysis:")
    for reward_val in [0.0, 1.0]:
        fps = analyzer.find_fixed_points(reward=reward_val, action=0, n_inits=10)
        print(f"   Reward={reward_val}, Action=0 → {len(fps)} fixed point(s)")
        for i, fp in enumerate(fps):
            jac = analyzer.compute_jacobian_at_fixed_point(fp, reward=reward_val, action=0)
            eigs = np.linalg.eigvals(jac)
            max_abs = np.max(np.abs(eigs))
            stability = "stable" if max_abs < 1.0 else "unstable"
            print(f"     FP{i+1}: [{fp[0]:+.3f}, {fp[1]:+.3f}], "
                  f"max|λ|={max_abs:.3f} ({stability})")
    
    # Create comprehensive visualization
    print("\n5. Creating visualizations...")
    fig = plt.figure(figsize=(18, 12))
    
    # Vector fields (2x2 grid for different input conditions)
    for i, (reward_val, title) in enumerate([(0.0, 'No Reward'), (1.0, 'Reward')]):
        for j, action_val in enumerate([0, 1]):
            ax = plt.subplot(3, 3, i*3 + j + 1)
            analyzer.plot_vector_field_2d(
                reward=reward_val, 
                action=action_val,
                xlim=(-1.5, 1.5), 
                ylim=(-1.5, 1.5),
                n_grid=15,
                ax=ax
            )
            ax.set_title(f'{title}, Arm {action_val}')
            
            # Add fixed points
            fps = analyzer.find_fixed_points(reward=reward_val, action=action_val, n_inits=10)
            for fp in fps:
                jac = analyzer.compute_jacobian_at_fixed_point(fp, reward=reward_val, action=action_val)
                eigs = np.linalg.eigvals(jac)
                max_abs = np.max(np.abs(eigs))
                color = 'green' if max_abs < 1.0 else 'red'
                ax.scatter(fp[0], fp[1], s=200, c=color, marker='*', 
                          edgecolors='black', linewidths=2, zorder=10)
    
    # Sample trajectory
    ax_traj = plt.subplot(3, 3, 3)
    agent.reset()
    env.reset()
    action = agent.last_action
    all_hidden = []
    
    for trial in range(100):
        reward, _ = env.step(action)
        action, switched, h = agent.act(reward, return_hidden_states=True)
        all_hidden.append(h)
    
    all_hidden = np.concatenate(all_hidden, axis=0)
    analyzer.plot_trajectory(all_hidden, ax=ax_traj, alpha=0.7, linewidth=2)
    ax_traj.set_title('Sample Trajectory (100 trials)')
    ax_traj.set_xlim(-1.5, 1.5)
    ax_traj.set_ylim(-1.5, 1.5)
    
    # Behavioral statistics
    ax_behav = plt.subplot(3, 3, 4)
    df = recording_df.copy()
    df['prev_reward'] = df['reward'].shift(1)
    df = df.dropna()
    switch_by_reward = df.groupby('prev_reward')['switched'].mean()
    ax_behav.bar([0, 1], switch_by_reward, color=['red', 'green'], alpha=0.7)
    ax_behav.set_xticks([0, 1])
    ax_behav.set_xticklabels(['No reward', 'Reward'])
    ax_behav.set_ylabel('P(switch)')
    ax_behav.set_title('Strategy: Switch Probability')
    ax_behav.set_ylim([0, 1])
    ax_behav.grid(True, alpha=0.3, axis='y')
    
    # Reward rate over time
    ax_reward = plt.subplot(3, 3, 5)
    df['trial_bin'] = (df.index // 100)
    reward_over_time = df.groupby('trial_bin')['reward'].mean()
    ax_reward.plot(reward_over_time.index * 100, reward_over_time.values, 
                   marker='o', linewidth=2)
    ax_reward.axhline(0.8, color='gray', linestyle='--', label='Best arm')
    ax_reward.set_xlabel('Trial')
    ax_reward.set_ylabel('Reward rate')
    ax_reward.set_title('Learning Progress')
    ax_reward.set_ylim([0, 1])
    ax_reward.legend()
    ax_reward.grid(True, alpha=0.3)
    
    # Switch rate over time
    ax_switch = plt.subplot(3, 3, 6)
    switch_over_time = df.groupby('trial_bin')['switched'].mean()
    ax_switch.plot(switch_over_time.index * 100, switch_over_time.values,
                   marker='o', linewidth=2, color='orange')
    ax_switch.set_xlabel('Trial')
    ax_switch.set_ylabel('Switch rate')
    ax_switch.set_title('Exploration Over Time')
    ax_switch.set_ylim([0, 1])
    ax_switch.grid(True, alpha=0.3)
    
    # Hidden states colored by reward
    ax_hidden = plt.subplot(3, 3, 7)
    rewarded = all_hidden[1:][df['reward'].values[:len(all_hidden)-1] == 1]
    unrewarded = all_hidden[1:][df['reward'].values[:len(all_hidden)-1] == 0]
    
    if len(rewarded) > 0:
        ax_hidden.scatter(rewarded[:, 0], rewarded[:, 1], 
                         c='green', alpha=0.3, s=20, label='After reward')
    if len(unrewarded) > 0:
        ax_hidden.scatter(unrewarded[:, 0], unrewarded[:, 1], 
                         c='red', alpha=0.3, s=20, label='After no reward')
    ax_hidden.set_xlabel('Hidden unit 1')
    ax_hidden.set_ylabel('Hidden unit 2')
    ax_hidden.set_title('Hidden States by Outcome')
    ax_hidden.legend()
    ax_hidden.grid(True, alpha=0.3)
    ax_hidden.set_xlim(-1.5, 1.5)
    ax_hidden.set_ylim(-1.5, 1.5)
    
    # Hidden states colored by action
    ax_action = plt.subplot(3, 3, 8)
    for arm in range(3):
        arm_states = all_hidden[1:][df['action'].values[:len(all_hidden)-1] == arm]
        if len(arm_states) > 0:
            ax_action.scatter(arm_states[:, 0], arm_states[:, 1], 
                            alpha=0.4, s=20, label=f'Arm {arm}')
    ax_action.set_xlabel('Hidden unit 1')
    ax_action.set_ylabel('Hidden unit 2')
    ax_action.set_title('Hidden States by Action')
    ax_action.legend()
    ax_action.grid(True, alpha=0.3)
    ax_action.set_xlim(-1.5, 1.5)
    ax_action.set_ylim(-1.5, 1.5)
    
    # Action preference
    ax_pref = plt.subplot(3, 3, 9)
    action_counts = df['action'].value_counts().sort_index()
    ax_pref.bar(action_counts.index, action_counts.values, 
               color=['green', 'orange', 'orange'], alpha=0.7)
    ax_pref.set_xlabel('Arm')
    ax_pref.set_ylabel('Count')
    ax_pref.set_title('Action Distribution')
    ax_pref.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/tmp/tiny_rnn_2d_analysis.png', dpi=150, bbox_inches='tight')
    print("   Saved comprehensive analysis to: /tmp/tiny_rnn_2d_analysis.png")
    
    print("\n" + "=" * 70)
    print("Analysis complete! Check /tmp/tiny_rnn_2d_analysis.png")
    print("=" * 70)
    
    print("\n💡 Interpretation tips:")
    print("   - Green stars = stable fixed points (attractors)")
    print("   - Red stars = unstable fixed points (repellers)")
    print("   - Vector field shows flow of hidden state dynamics")
    print("   - Trajectories show how hidden state evolves over trials")
    print("   - Separatrices (boundaries) indicate decision regions")
    print("   - Compare 'No Reward' vs 'Reward' vector fields to see")
    print("     how feedback changes the dynamical landscape")


if __name__ == '__main__':
    main()
