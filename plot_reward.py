import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = np.load('logs/reward_history.npy')
ep = np.arange(1, len(r) + 1)

def moving_avg(x, w=10):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')

ma = moving_avg(r, 10)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ep, r, color='#a5c8ff', linewidth=1, alpha=0.6, label='Episode reward')
ax.plot(ep[9:], ma, color='#1f5fd1', linewidth=2.2, label='10-episode moving average')
ax.set_xlabel('Episode')
ax.set_ylabel('Cumulative episode reward')
ax.set_title('DQN Training Reward Convergence (SUMO + CV environment)', fontsize=12, weight='bold')
ax.grid(alpha=0.25)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('screens/fig_11_5_reward_curve.png', dpi=150)
print('saved reward curve, episodes=', len(r), 'final avg10=', ma[-1] if len(ma) else None)
