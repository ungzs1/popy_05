"""
Generate architecture diagram for the RNN agent.

This script creates a visual representation of the RNN agent architecture.
Requires matplotlib and may require additional packages for fancy diagrams.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_rnn_architecture():
    """Draw the RNN agent architecture diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'RNN Agent Architecture for Stay/Switch Task', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # === INPUT LAYER ===
    # Environment
    env_box = FancyBboxPatch((0.5, 9), 1.5, 1, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(env_box)
    ax.text(1.25, 9.5, 'Environment\n(3-Armed\nBandit)', 
            ha='center', va='center', fontsize=9)
    
    # Reward signal
    reward_box = FancyBboxPatch((3, 9.5), 1.2, 0.6,
                               boxstyle="round,pad=0.05",
                               edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(reward_box)
    ax.text(3.6, 9.8, 'Reward\n(0 or 1)', ha='center', va='center', fontsize=8)
    
    # Previous switch
    switch_box = FancyBboxPatch((3, 8.5), 1.2, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='purple', facecolor='plum', linewidth=2)
    ax.add_patch(switch_box)
    ax.text(3.6, 8.8, 'Prev Switch\n(0 or 1)', ha='center', va='center', fontsize=8)
    
    # Arrows from env
    arrow1 = FancyArrowPatch((2, 9.7), (3, 9.8), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((2, 9.3), (3, 8.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='purple')
    ax.add_patch(arrow2)
    
    # === INPUT PROCESSING ===
    # Temporal expansion
    temporal_box = FancyBboxPatch((5, 8.5), 1.8, 1.6,
                                  boxstyle="round,pad=0.1",
                                  edgecolor='orange', facecolor='lightyellow', 
                                  linewidth=2, linestyle='--')
    ax.add_patch(temporal_box)
    ax.text(5.9, 9.8, 'Temporal Processing', ha='center', va='top', 
            fontsize=9, fontweight='bold')
    ax.text(5.9, 9.3, 'Feedback Duration:\n3 timesteps\n(sustained input)', 
            ha='center', va='center', fontsize=7)
    
    # Arrows to temporal
    arrow3 = FancyArrowPatch((4.2, 9.8), (5, 9.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)
    arrow4 = FancyArrowPatch((4.2, 8.8), (5, 9.0),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    # === RNN LAYER ===
    rnn_box = FancyBboxPatch((4.5, 6), 2.8, 2,
                            boxstyle="round,pad=0.1",
                            edgecolor='darkblue', facecolor='lightsteelblue', linewidth=3)
    ax.add_patch(rnn_box)
    ax.text(5.9, 7.7, 'RNN (GRU)', ha='center', va='top', 
            fontsize=11, fontweight='bold')
    ax.text(5.9, 7.2, 'Hidden Size: 32\nNum Layers: 1', 
            ha='center', va='center', fontsize=8)
    
    # Hidden state loop
    hidden_loop = FancyArrowPatch((7.3, 7), (7.8, 7.5),
                                 arrowstyle='->', mutation_scale=15, 
                                 linewidth=2, color='darkblue',
                                 connectionstyle="arc3,rad=.5")
    ax.add_patch(hidden_loop)
    ax.text(8.1, 7.5, 'Hidden\nState', ha='left', va='center', 
            fontsize=7, style='italic')
    
    # Arrow from temporal to RNN
    arrow5 = FancyArrowPatch((5.9, 8.5), (5.9, 8),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5, color='black')
    ax.add_patch(arrow5)
    
    # === DECISION DELAY (Optional) ===
    delay_box = FancyBboxPatch((4.5, 4.8), 2.8, 0.8,
                              boxstyle="round,pad=0.05",
                              edgecolor='gray', facecolor='lightgray', 
                              linewidth=1, linestyle=':')
    ax.add_patch(delay_box)
    ax.text(5.9, 5.2, 'Decision Delay (1 step)', 
            ha='center', va='center', fontsize=7, style='italic')
    
    # Arrow through delay
    arrow6 = FancyArrowPatch((5.9, 6), (5.9, 5.6),
                            arrowstyle='->', mutation_scale=15, linewidth=1.5, 
                            color='gray', linestyle=':')
    ax.add_patch(arrow6)
    
    # === OUTPUT LAYER ===
    fc_box = FancyBboxPatch((5, 3.5), 1.8, 1,
                           boxstyle="round,pad=0.1",
                           edgecolor='darkred', facecolor='lightcoral', linewidth=2)
    ax.add_patch(fc_box)
    ax.text(5.9, 4.2, 'Fully Connected', ha='center', va='top', fontsize=9, fontweight='bold')
    ax.text(5.9, 3.8, '→ Sigmoid', ha='center', va='center', fontsize=8)
    
    # Arrow to output
    arrow7 = FancyArrowPatch((5.9, 4.8), (5.9, 4.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow7)
    
    # === SWITCH PROBABILITY ===
    prob_box = FancyBboxPatch((4.8, 2.2), 2.2, 0.8,
                             boxstyle="round,pad=0.1",
                             edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(prob_box)
    ax.text(5.9, 2.6, 'Switch Probability\nP(switch) ∈ [0, 1]', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow to probability
    arrow8 = FancyArrowPatch((5.9, 3.5), (5.9, 3.0),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='darkred')
    ax.add_patch(arrow8)
    
    # === DECISION MAKING ===
    # Add exploration noise
    noise_box = FancyBboxPatch((7.5, 2.2), 1.3, 0.8,
                              boxstyle="round,pad=0.05",
                              edgecolor='orange', facecolor='moccasin', linewidth=1)
    ax.add_patch(noise_box)
    ax.text(8.15, 2.6, '+ Noise\nσ=0.05', ha='center', va='center', fontsize=7)
    
    # Decision node
    decision_box = FancyBboxPatch((4.5, 0.8), 2.8, 1,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='purple', facecolor='thistle', linewidth=2)
    ax.add_patch(decision_box)
    ax.text(5.9, 1.5, 'Decision', ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(5.9, 1.1, 'Sample: stay or switch', ha='center', va='center', fontsize=8)
    
    # Arrow to decision
    arrow9 = FancyArrowPatch((5.9, 2.2), (5.9, 1.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow9)
    
    # === ACTIONS ===
    stay_box = FancyBboxPatch((2.5, 0.2), 1.3, 0.5,
                             boxstyle="round,pad=0.05",
                             edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(stay_box)
    ax.text(3.15, 0.45, 'STAY\n(repeat action)', ha='center', va='center', fontsize=8)
    
    switch_box_out = FancyBboxPatch((6.5, 0.2), 1.3, 0.5,
                                    boxstyle="round,pad=0.05",
                                    edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax.add_patch(switch_box_out)
    ax.text(7.15, 0.45, 'SWITCH\n(random alt)', ha='center', va='center', fontsize=8)
    
    # Arrows to actions
    arrow10 = FancyArrowPatch((5.2, 0.8), (3.8, 0.5),
                             arrowstyle='->', mutation_scale=15, linewidth=2, color='blue')
    ax.add_patch(arrow10)
    ax.text(4.3, 0.65, 'P(stay)', ha='center', fontsize=7)
    
    arrow11 = FancyArrowPatch((6.6, 0.8), (6.8, 0.7),
                             arrowstyle='->', mutation_scale=15, linewidth=2, color='red')
    ax.add_patch(arrow11)
    ax.text(6.9, 0.9, 'P(switch)', ha='left', fontsize=7)
    
    # === LEARNING (REINFORCE) ===
    learning_box = FancyBboxPatch((0.5, 4), 2.5, 2.5,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='darkgreen', facecolor='honeydew', 
                                 linewidth=2, linestyle='-.')
    ax.add_patch(learning_box)
    ax.text(1.75, 6.2, 'REINFORCE Learning', ha='center', va='top', 
            fontsize=10, fontweight='bold', color='darkgreen')
    ax.text(1.75, 5.5, '1. Store log π(a)\n2. Collect rewards\n3. Compute returns\n' + 
                       '4. Policy gradient:\n   ∇J = Σ log π(a) R\n5. Update weights',
            ha='center', va='center', fontsize=7, family='monospace')
    
    # Feedback arrow
    feedback_arrow = FancyArrowPatch((1.75, 4), (1.75, 0.8),
                                    arrowstyle='->', mutation_scale=20, 
                                    linewidth=2, color='darkgreen',
                                    linestyle='--')
    ax.add_patch(feedback_arrow)
    ax.text(1, 2.4, 'Rewards', ha='center', va='center', 
            fontsize=8, color='darkgreen', rotation=90)
    
    gradient_arrow = FancyArrowPatch((3, 5.2), (4.5, 7),
                                    arrowstyle='->', mutation_scale=20,
                                    linewidth=2, color='darkgreen',
                                    linestyle='--')
    ax.add_patch(gradient_arrow)
    ax.text(3.5, 6, 'Gradients', ha='center', va='center',
            fontsize=8, color='darkgreen', rotation=45)
    
    # === INFO BOXES ===
    # Parameters
    params_text = ax.text(9.2, 10.5, 
                         'Key Parameters:\n' +
                         '• Hidden size: 32\n' +
                         '• Learning rate: 0.001\n' +
                         '• Discount γ: 0.95\n' +
                         '• Exploration: 0.05\n' +
                         '• Update interval: 100',
                         ha='left', va='top', fontsize=7,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Training info
    training_text = ax.text(9.2, 7,
                           'Training:\n' +
                           '• Algorithm: REINFORCE\n' +
                           '• Optimizer: Adam\n' +
                           '• Gradient clipping: 1.0\n' +
                           '• Return normalization\n' +
                           '• ~50k trials',
                           ha='left', va='top', fontsize=7,
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Temporal dynamics
    temporal_text = ax.text(9.2, 3.5,
                           'Temporal:\n' +
                           '• Feedback: 3 steps\n' +
                           '• Delay: 1 step\n' +
                           '• Continuous hidden\n  state updates',
                           ha='left', va='top', fontsize=7,
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    return fig


def draw_temporal_timeline():
    """Draw the temporal dynamics timeline."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Title
    ax.text(7, 3.7, 'Temporal Dynamics: Single Trial', 
            ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Timeline
    ax.plot([1, 13], [2, 2], 'k-', linewidth=2)
    
    # Feedback period
    feedback_rect = mpatches.Rectangle((1, 1.5), 3, 1, 
                                       edgecolor='green', facecolor='lightgreen', 
                                       linewidth=2, alpha=0.7)
    ax.add_patch(feedback_rect)
    ax.text(2.5, 2, 'Feedback Period\n(3 timesteps)', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(2.5, 1.2, 'Input: [reward, prev_switch]', 
            ha='center', va='center', fontsize=7, style='italic')
    
    # Decision delay
    delay_rect = mpatches.Rectangle((4, 1.5), 1, 1,
                                    edgecolor='orange', facecolor='moccasin',
                                    linewidth=2, alpha=0.7, linestyle='--')
    ax.add_patch(delay_rect)
    ax.text(4.5, 2, 'Delay\n(1 step)', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    ax.text(4.5, 1.2, 'Input: zeros', ha='center', va='center', 
            fontsize=7, style='italic')
    
    # Decision
    decision_rect = mpatches.Rectangle((5, 1.5), 1.5, 1,
                                      edgecolor='blue', facecolor='lightblue',
                                      linewidth=2, alpha=0.7)
    ax.add_patch(decision_rect)
    ax.text(5.75, 2, 'Decision\nOutput', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    ax.text(5.75, 1.2, 'P(switch)', ha='center', va='center',
            fontsize=7, style='italic')
    
    # Action execution
    action_rect = mpatches.Rectangle((6.5, 1.5), 1.5, 1,
                                    edgecolor='purple', facecolor='plum',
                                    linewidth=2, alpha=0.7)
    ax.add_patch(action_rect)
    ax.text(7.25, 2, 'Action', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    ax.text(7.25, 1.2, 'Stay/Switch', ha='center', va='center',
            fontsize=7, style='italic')
    
    # Next trial
    next_rect = mpatches.Rectangle((8, 1.5), 3, 1,
                                  edgecolor='green', facecolor='lightgreen',
                                  linewidth=2, alpha=0.3, linestyle=':')
    ax.add_patch(next_rect)
    ax.text(9.5, 2, 'Next Trial\nFeedback...', ha='center', va='center',
            fontsize=9, fontweight='bold', alpha=0.5)
    
    # RNN processing
    ax.text(7, 3.2, 'RNN Hidden State (continuous updates)', 
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.5))
    ax.annotate('', xy=(11, 3), xytext=(1, 3),
               arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Time markers
    for i, t in enumerate([1, 2, 3, 4, 5, 6.5, 8]):
        ax.plot([t, t], [2, 1.8], 'k-', linewidth=1)
        ax.text(t, 1.6, f't={i}', ha='center', va='top', fontsize=7)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Generating RNN Agent Architecture Diagrams...")
    
    # Generate architecture diagram
    fig1 = draw_rnn_architecture()
    fig1.savefig('rnn_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: rnn_architecture.png")
    
    # Generate temporal timeline
    fig2 = draw_temporal_timeline()
    fig2.savefig('temporal_timeline.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: temporal_timeline.png")
    
    plt.show()
    print("\nDiagrams generated successfully!")
