"""train.py : training loop."""
import time
import numpy as np

from traffic_env import TrafficEnv
from dqn_agent import DQNAgent

EPISODES = 120
MAX_STEPS = 1200

env = TrafficEnv(sumo_cfg='network/intersection.sumocfg', gui=False, max_steps=MAX_STEPS)
agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)

history = []
wait_history = []
t_start = time.time()

for ep in range(1, EPISODES + 1):
    state, total_r, done = env.reset(), 0.0, False
    ep_wait = 0.0
    while not done:
        action = agent.act(state)
        nxt, reward, done, info = env.step(action)
        agent.remember(state, action, reward, nxt, done)
        agent.replay()
        state, total_r = nxt, total_r + reward
        ep_wait = info['cum_wait']
    agent.decay_epsilon()
    history.append(total_r)
    wait_history.append(ep_wait)

    if ep % 5 == 0 or ep == 1:
        elapsed = time.time() - t_start
        print(f'Episode {ep:3d} | reward {total_r:9.1f} | '
              f'avg10 {np.mean(history[-10:]):9.1f} | eps {agent.eps:.3f} | '
              f'wait {ep_wait:8.1f} | elapsed {elapsed:6.1f}s', flush=True)

agent.save('checkpoints/dqn_latest.npz')
env.close()
np.save('logs/reward_history.npy', np.array(history))
np.save('logs/wait_history.npy', np.array(wait_history))
print('Training complete. Checkpoint saved to checkpoints/dqn_latest.npz')
