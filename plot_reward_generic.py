"""plot_reward_generic.py : reward convergence plot for a train_generic.py run."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = np.load('logs/reward_history_generic.npy')
ep = np.arange(1, len(r) + 1)
ma = np.convolve(r, np.ones(min(10, len(r))) / min(10, len(r)), mode='valid')

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ep, r, color='#a5c8ff', linewidth=1, alpha=0.6, label='Episode reward')
ax.plot(ep[len(ep)-len(ma):], ma, color='#1f5fd1', linewidth=2.2, label='10-episode moving average')
ax.set_xlabel('Episode')
ax.set_ylabel('Cumulative episode reward')
ax.set_title('DQN Training Reward Convergence', fontsize=12, weight='bold')
ax.grid(alpha=0.25)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('screens/osm_reward_curve.png', dpi=150)
print('saved screens/osm_reward_curve.png | first10 avg:', r[:10].mean(), '| last10 avg:', r[-10:].mean())
