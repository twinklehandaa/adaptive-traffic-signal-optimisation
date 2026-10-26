import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from baselines import run_fixed_time, run_actuated
from traffic_env import TrafficEnv
from dqn_agent import DQNAgent

MAX_STEPS = 1200
SEEDS = [1, 2, 3]

def run_dqn(seed):
    env = TrafficEnv(sumo_cfg='network/intersection.sumocfg', gui=False,
                      max_steps=MAX_STEPS, seed=seed)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    agent.load('checkpoints/dqn_latest.npz')
    agent.eps = 0.0  # greedy / evaluation mode
    state = env.reset()
    done = False
    while not done:
        a = agent.act(state)
        state, r, done, info = env.step(a)
    result = {
        'avg_wait': float(np.mean(env.wait_log)),
        'avg_queue': float(np.mean(env.queue_log)),
        'max_queue': int(np.max(env.queue_log)),
        'arrived_throughput': env.arrived_total,
    }
    env.close()
    return result


results = {'Fixed-Time': [], 'Actuated': [], 'DQN (ours)': []}
for seed in SEEDS:
    results['Fixed-Time'].append(run_fixed_time('network/intersection.sumocfg', max_steps=MAX_STEPS, seed=seed))
    results['Actuated'].append(run_actuated('network/intersection.sumocfg', max_steps=MAX_STEPS, seed=seed))
    results['DQN (ours)'].append(run_dqn(seed))

summary = {}
for k, runs in results.items():
    waits = [r['avg_wait'] for r in runs]
    queues = [r['avg_queue'] for r in runs]
    thrus = [r['arrived_throughput'] for r in runs]
    summary[k] = {
        'avg_wait': float(np.mean(waits)),
        'avg_wait_std': float(np.std(waits)),
        'avg_queue': float(np.mean(queues)),
        'avg_arrived_throughput': float(np.mean(thrus)),
    }

with open('logs/comparison_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

labels = list(summary.keys())
waits = [summary[k]['avg_wait'] for k in labels]
errs = [summary[k]['avg_wait_std'] for k in labels]
colors = ['#c0392b', '#e67e22', '#1f5fd1']

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(labels, waits, yerr=errs, capsize=6, color=colors)
ax.set_ylabel('Average total network waiting time (s)')
ax.set_title('Fixed-Time vs Actuated vs DQN\n(mean over 3 seeded runs, 20-min scenario)', fontsize=11, weight='bold')
for b, v in zip(bars, waits):
    ax.text(b.get_x() + b.get_width() / 2, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=10)
ax.grid(alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig('screens/fig_11_6_comparison.png', dpi=150)
print('saved comparison chart')
