"""compare_baselines_generic.py : fixed-time vs native-actuated vs DQN,
for ANY network (point it at your OSM osm.sumocfg). Produces
logs/osm_comparison_summary.json and screens/osm_comparison.png.

Usage:
    python3 compare_baselines_generic.py path/to/osm.sumocfg path/to/checkpoint.npz
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from baselines_generic import run_fixed_time, run_native_actuated
from traffic_env_generic import TrafficEnvGeneric
from dqn_agent import DQNAgent

SUMO_CFG = sys.argv[1] if len(sys.argv) > 1 else 'osm_real/osm.sumocfg'
CHECKPOINT = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/dqn_generic_latest.npz'
MAX_STEPS = 1200
SEEDS = [1, 2, 3]

print(f'Comparing on: {SUMO_CFG}  |  checkpoint: {CHECKPOINT}')


def run_dqn(seed):
    env = TrafficEnvGeneric(sumo_cfg=SUMO_CFG, gui=False, max_steps=MAX_STEPS, seed=seed)
    state = env.reset()  # triggers auto-discovery, sets env.state_dim/action_dim
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    agent.load(CHECKPOINT)
    agent.eps = 0.0  # greedy evaluation
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


results = {'Fixed-Time': [], 'Native Actuated': [], 'DQN (ours)': []}
for seed in SEEDS:
    print(f'-- seed {seed} --')
    results['Fixed-Time'].append(run_fixed_time(SUMO_CFG, max_steps=MAX_STEPS, seed=seed))
    results['Native Actuated'].append(run_native_actuated(SUMO_CFG, max_steps=MAX_STEPS, seed=seed))
    results['DQN (ours)'].append(run_dqn(seed))

summary = {}
for k, runs in results.items():
    waits = [r['avg_wait'] for r in runs]
    queues = [r['avg_queue'] for r in runs]
    thrus = [r['arrived_throughput'] for r in runs]
    summary[k] = {
        'avg_wait': float(np.mean(waits)), 'avg_wait_std': float(np.std(waits)),
        'avg_queue': float(np.mean(queues)),
        'avg_arrived_throughput': float(np.mean(thrus)),
    }

with open('logs/osm_comparison_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

labels = list(summary.keys())
waits = [summary[k]['avg_wait'] for k in labels]
errs = [summary[k]['avg_wait_std'] for k in labels]
colors = ['#c0392b', '#27ae60', '#1f5fd1']

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(labels, waits, yerr=errs, capsize=6, color=colors)
ax.set_ylabel('Average waiting time (s)')
ax.set_title('Fixed-time vs native actuated vs DQN\n(3 seeded runs)', fontsize=11, weight='bold')
for b, v in zip(bars, waits):
    ax.text(b.get_x() + b.get_width() / 2, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=10)
ax.grid(alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig('screens/osm_comparison.png', dpi=150)
print('saved screens/osm_comparison.png')
