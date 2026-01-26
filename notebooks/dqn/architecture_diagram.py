"""
Visual Architecture Diagram for Tiny RNN Agent
"""

ARCHITECTURE_DIAGRAM = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                     TINY RNN AGENT ARCHITECTURE (v2.0)                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│                              INPUT CONSTRUCTION                          │
└─────────────────────────────────────────────────────────────────────────┘

   Environment                    Agent State
   ┌──────────┐                  ┌──────────────┐
   │ Reward   │                  │ Previous     │
   │ (0 or 1) │                  │ Action (0-2) │
   └────┬─────┘                  └──────┬───────┘
        │                               │
        │                               │ One-hot encode
        │                               │
        ▼                               ▼
   ┌────────┐                  ┌──────────────┐
   │  [r]   │                  │ [0, 1, 0]    │
   └────┬───┘                  └──────┬───────┘
        │                              │
        └──────────┬───────────────────┘
                   │
                   │ Concatenate
                   ▼
          ┌─────────────────┐
          │ [r, a0, a1, a2] │  ← Input tensor (1 + n_arms dims)
          └────────┬────────┘
                   │
                   │ Repeat for feedback_duration steps
                   │
                   ▼
          ┌─────────────────┐
          │ Sequence input  │
          │ (T × input_dim) │
          └────────┬────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           NETWORK FORWARD PASS                           │
└─────────────────────────────────────────────────────────────────────────┘

                   │
                   │ Initial hidden state h₀
                   ▼
          ┌─────────────────┐
          │  RNN/GRU Cell   │  ← Tiny! (2-4 hidden units)
          │                 │
          │  h_t = f(x_t,   │
          │         h_{t-1})│
          └────────┬────────┘
                   │
                   │ Hidden state h_T
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
   ┌────────────┐    ┌────────────┐
   │  Actor     │    │  Critic    │
   │  Head      │    │  Head      │
   │            │    │            │
   │ Linear     │    │ Linear     │
   │ + Sigmoid  │    │            │
   └─────┬──────┘    └─────┬──────┘
         │                 │
         ▼                 ▼
   ┌──────────┐      ┌──────────┐
   │ P(switch)│      │  V(s)    │  ← Value estimate
   └────┬─────┘      └─────┬────┘
        │                  │
        │                  │
        ▼                  │

┌─────────────────────────────────────────────────────────────────────────┐
│                        ACTION SAMPLING & TRAINING                        │
└─────────────────────────────────────────────────────────────────────────┘

   P(switch)              
   ┌────────┐             
   │ Bern.  │ ← Sample from Bernoulli distribution            
   │ (p)    │   (no external noise!)                         
   └───┬────┘                                                 
       │                                                      
       ▼                                                      
   ┌────────────┐                                            
   │ should_    │                                            
   │ switch?    │                                            
   └─────┬──────┘                                            
         │                                                    
    ┌────┴─────┐                                            
    │          │                                             
 switch=1   switch=0                                         
    │          │                                             
    ▼          ▼                                             
┌────────┐ ┌────────┐                                        
│ Random │ │  Stay  │                                        
│ alt.   │ │  prev  │                                        
│ arm    │ │  arm   │                                        
└────┬───┘ └───┬────┘                                        
     │         │                                             
     └────┬────┘                                             
          │                                                   
          ▼                                                   
      ┌────────┐                                             
      │ Action │                                             
      └────┬───┘                                             
           │                                                  
           │ Execute in environment                          
           ▼                                                  
      ┌────────┐                                             
      │ Reward │                                             
      └────┬───┘                                             
           │                                                  
           │ Buffer K trials                                 
           ▼                                                  

┌─────────────────────────────────────────────────────────────────────────┐
│                     TRUNCATED BPTT UPDATE (every K trials)              │
└─────────────────────────────────────────────────────────────────────────┘

   Buffered Trials (K=20 by default)
   ┌─────────────────────────────────────┐
   │ [x₁, a₁, r₁, V₁, log π₁, H₁]       │
   │ [x₂, a₂, r₂, V₂, log π₂, H₂]       │
   │ ...                                  │
   │ [x_K, a_K, r_K, V_K, log π_K, H_K]  │
   └──────────────┬──────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ Compute Returns:             │
   │ R_t = Σ_{k=0}^{T-t} γᵏ r_{t+k}│
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ Compute Advantages:          │
   │ A_t = R_t - V_t              │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ Compute Losses:              │
   │                              │
   │ L_actor = -log π(a|s) · A   │
   │ L_critic = (V - R)²          │
   │ L_entropy = -H(π)            │
   │                              │
   │ L_total = L_actor            │
   │         + λ_V · L_critic     │
   │         - λ_H · L_entropy    │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ Backpropagate through        │
   │ K-step sequence              │
   │                              │
   │ Clip gradients (max_norm=1)  │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ Update network parameters    │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ Detach hidden state          │
   │ (truncation boundary)        │
   └──────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         ANALYSIS & INTERPRETATION                        │
└─────────────────────────────────────────────────────────────────────────┘

   1. FIXED-POINT ANALYSIS
   ┌─────────────────────────────┐
   │ For each input condition:   │
   │ - Find h* s.t. h* = f(h*, u)│
   │ - Compute Jacobian J        │
   │ - Analyze eigenvalues       │
   └─────────────────────────────┘
        ↓
   Stable attractor (|λ| < 1) or unstable (|λ| ≥ 1)?

   2. TRAJECTORY VISUALIZATION
   ┌─────────────────────────────┐
   │ Record h_t over trials      │
   │ - Plot in 2D (if dim=2)     │
   │ - Or use PCA projection     │
   └─────────────────────────────┘
        ↓
   See decision-making dynamics over time

   3. VECTOR FIELD (2D only)
   ┌─────────────────────────────┐
   │ For each point in (h₁, h₂): │
   │ - Compute Δh = f(h,u) - h   │
   │ - Plot as vector field      │
   └─────────────────────────────┘
        ↓
   Visualize attractor basins & separatrices

   4. BEHAVIORAL CHARACTERIZATION
   ┌─────────────────────────────┐
   │ - P(stay | reward)          │
   │ - P(switch | no reward)     │
   │ - Reward rate by arm        │
   │ - Switch rate over time     │
   └─────────────────────────────┘
        ↓
   WSLS? Perseveration? Inference?

╔═══════════════════════════════════════════════════════════════════════════╗
║                            KEY DIFFERENCES                                 ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  BEFORE (v1.0)                  │  AFTER (v2.0)                           ║
╠═════════════════════════════════╪═════════════════════════════════════════╣
║ Input: [reward, prev_switch]    │ Input: [reward, prev_action_onehot]    ║
║ Hidden: 32 units                 │ Hidden: 4 units (tiny!)                ║
║ Output: P(switch) only           │ Output: P(switch) + V(state)           ║
║ Training: REINFORCE              │ Training: Actor-Critic                 ║
║ BPTT: Broken (detach every step) │ BPTT: Truncated (every K steps)       ║
║ Sampling: External noise         │ Sampling: Bernoulli distribution      ║
║ Exploration: Noise injection     │ Exploration: Entropy bonus            ║
║ Analysis: None                   │ Analysis: Full toolbox                ║
║ Reproducibility: None            │ Reproducibility: Seeding + save/load  ║
╚═════════════════════════════════╧═════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════╗
║                              HYPERPARAMETERS                               ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Parameter          │ Default │ Purpose                                    ║
╠════════════════════╪═════════╪════════════════════════════════════════════╣
║ hidden_size        │    4    │ Number of RNN hidden units (2-4 for viz)  ║
║ rnn_type           │  'GRU'  │ 'RNN' (simple) or 'GRU' (better)          ║
║ learning_rate      │  0.001  │ Adam optimizer learning rate              ║
║ gamma              │   1.0   │ Discount factor (1.0 for bandits)         ║
║ entropy_coef       │  0.01   │ Entropy bonus weight (exploration)        ║
║ value_coef         │   0.5   │ Value loss weight                         ║
║ bptt_truncation    │   20    │ Update every K trials                     ║
║ feedback_duration  │    1    │ Timesteps to present feedback             ║
║ decision_delay     │    0    │ Delay between feedback and decision       ║
╚════════════════════╧═════════╧════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════╗
║                            TYPICAL USAGE                                   ║
╚═══════════════════════════════════════════════════════════════════════════╝

1. TRAINING
   ┌─────────────────────────────────────────────────────────────┐
   │ agent = RNNAgent(n_arms=3, hidden_size=4, seed=42)         │
   │ for trial in range(1000):                                   │
   │     action, _ = agent.act(reward)                           │
   │     agent.store_reward(reward)                              │
   │     if agent.should_update():                               │
   │         agent.update()                                      │
   │ agent.update(force=True)                                    │
   └─────────────────────────────────────────────────────────────┘

2. ANALYSIS
   ┌─────────────────────────────────────────────────────────────┐
   │ analyzer = RNNAnalyzer(agent)                               │
   │ fps = analyzer.find_fixed_points(reward=1.0, action=0)      │
   │ analyzer.plot_vector_field_2d(reward=1.0, action=0)         │
   │ stats = analyzer.compute_behavioral_stats(recording_df)     │
   └─────────────────────────────────────────────────────────────┘

3. SAVE/LOAD
   ┌─────────────────────────────────────────────────────────────┐
   │ agent.save('checkpoint.pth')                                │
   │ agent.load('checkpoint.pth')                                │
   └─────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════╗
║                              FILES CREATED                                 ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ File                          │ Description                                ║
╠═══════════════════════════════╪════════════════════════════════════════════╣
║ rnn_agent.py                  │ Core implementation (refactored)           ║
║ README_tiny_rnn.md            │ Complete documentation                     ║
║ IMPROVEMENTS.md               │ Before/after comparison                    ║
║ QUICK_REFERENCE.md            │ One-page cheat sheet                       ║
║ SUMMARY.md                    │ Implementation summary                     ║
║ example_tiny_rnn_usage.py     │ 4D agent example                           ║
║ example_2d_rnn.py             │ 2D agent with vector fields                ║
╚═══════════════════════════════╧════════════════════════════════════════════╝

"""

if __name__ == "__main__":
    print(ARCHITECTURE_DIAGRAM)
