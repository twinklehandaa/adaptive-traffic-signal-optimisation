"""train_generic.py : training loop for ANY auto-discovered intersection
(a real OSM import, or the bundled synthetic one). Same logic as
train.py, just pointed at TrafficEnvGeneric.

Usage:
    python3 train_generic.py path/to/your_real_network.sumocfg
"""
import sys
import time
import numpy as np

from traffic_env_generic import TrafficEnvGeneric
from dqn_agent import DQNAgent

EPISODES = 120
MAX_STEPS = 1200

sumo_cfg = sys.argv[1] if len(sys.argv) > 1 else 'network/intersection.sumocfg'
print(f'Training on: {sumo_cfg}')

env = TrafficEnvGeneric(sumo_cfg=sumo_cfg, gui=False, max_steps=MAX_STEPS)
state = env.reset()  # triggers auto-discovery + prints the detected TLS/lanes
agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)

history, wait_history = [], []
t_start = time.time()

for ep in range(1, EPISODES + 1):
    state, total_r, done = env.reset(), 0.0, False
    while not done:
        action = agent.act(state)
        nxt, reward, done, info = env.step(action)
        agent.remember(state, action, reward, nxt, done)
        agent.replay()
        state, total_r = nxt, total_r + reward
    agent.decay_epsilon()
    history.append(total_r)

    if ep % 5 == 0 or ep == 1:
        elapsed = time.time() - t_start
        print(f'Episode {ep:3d} | reward {total_r:9.1f} | '
              f'avg10 {np.mean(history[-10:]):9.1f} | eps {agent.eps:.3f} | '
              f'elapsed {elapsed:6.1f}s', flush=True)

agent.save('checkpoints/dqn_generic_latest.npz')
env.close()
np.save('logs/reward_history_generic.npy', np.array(history))
print('Training complete. Checkpoint saved to checkpoints/dqn_generic_latest.npz')
