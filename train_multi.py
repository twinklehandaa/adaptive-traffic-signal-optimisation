"""train_multi.py : trains one DQN agent per intersection (independent
DQN, "IDQN") on a multi-intersection network, all inside ONE shared
SUMO simulation so the agents learn while affecting each other.

Each agent is the exact same DQNAgent class used for the single
intersection; only the environment and the loop around it change.

Usage:
    python3 make_multi_network.py                      # build the 2x2 grid once
    python3 train_multi.py                             # train on it
    python3 train_multi.py network_multi/grid.sumocfg --episodes 120
    python3 train_multi.py path/to/osm.sumocfg --neighbour-weight 0.2
    python3 train_multi.py --fast                      # skip frame rendering (≈5x faster,
                                                       # identical state values)
"""
import argparse
import json
import os
import time

import numpy as np

from dqn_agent import DQNAgent
from traffic_env_multi import MultiTrafficEnv, safe_name


def checkpoint_path(ckpt_dir, tls_id):
    return os.path.join(ckpt_dir, f'dqn_{safe_name(tls_id)}.npz')


def save_all(agents, env, ckpt_dir, args):
    os.makedirs(ckpt_dir, exist_ok=True)
    for tid, ag in agents.items():
        ag.save(checkpoint_path(ckpt_dir, tid))
    meta = {'sumo_cfg': args.sumo_cfg, 'tls_ids': env.tls_ids,
            'state_dims': env.state_dims,
            'neighbours': {t: env.ctrls[t].neighbours for t in env.tls_ids},
            'neighbour_weight': args.neighbour_weight, 'episodes': args.episodes}
    with open(os.path.join(ckpt_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def load_agents(env, ckpt_dir):
    """Rebuild one greedy agent per intersection from a train_multi.py run."""
    agents = {}
    for tid in env.tls_ids:
        ag = DQNAgent(state_dim=env.state_dims[tid], action_dim=env.action_dim)
        ag.load(checkpoint_path(ckpt_dir, tid))
        ag.eps = 0.0
        agents[tid] = ag
    return agents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sumo_cfg', nargs='?', default='network_multi/grid.sumocfg')
    ap.add_argument('--episodes', type=int, default=120)
    ap.add_argument('--max-steps', type=int, default=1200)
    ap.add_argument('--neighbour-weight', type=float, default=0.1,
                    help='share of neighbours\' reward added to each agent\'s reward')
    ap.add_argument('--fast', action='store_true',
                    help='skip rendering CV frames during training (same state values)')
    ap.add_argument('--ckpt-dir', default='checkpoints/multi')
    args = ap.parse_args()

    print(f'Training on: {args.sumo_cfg}')
    env = MultiTrafficEnv(args.sumo_cfg, max_steps=args.max_steps,
                          neighbour_weight=args.neighbour_weight,
                          use_perception=not args.fast)
    env.reset()  # auto-discovers every signal + neighbours
    agents = {tid: DQNAgent(state_dim=env.state_dims[tid], action_dim=env.action_dim,
                            seed=i)
              for i, tid in enumerate(env.tls_ids)}

    os.makedirs('logs', exist_ok=True)
    reward_hist, wait_hist = [], []
    best_wait = float('inf')
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        states = env.reset()
        totals = {t: 0.0 for t in env.tls_ids}
        done = False
        while not done:
            actions = {t: agents[t].act(states[t]) for t in env.tls_ids}
            nxt, rewards, done, _ = env.step(actions)
            for t in env.tls_ids:
                agents[t].remember(states[t], actions[t], rewards[t], nxt[t], done)
                agents[t].replay()
                totals[t] += rewards[t]
            states = nxt
        for ag in agents.values():
            ag.decay_epsilon()

        avg_wait = float(np.mean(env.wait_log))
        reward_hist.append([totals[t] for t in env.tls_ids])
        wait_hist.append(avg_wait)

        # keep the best greedy-ish policy seen so far as well as the latest
        if ep > args.episodes // 2 and avg_wait < best_wait:
            best_wait = avg_wait
            save_all(agents, env, args.ckpt_dir + '_best', args)

        if ep == 1 or ep % 5 == 0:
            per = '  '.join(f'{t}:{totals[t]:7.0f}' for t in env.tls_ids)
            print(f'Episode {ep:3d} | network wait {avg_wait:8.1f} | '
                  f'arrived {env.arrived_total:4d} | eps {agents[env.tls_ids[0]].eps:.3f} | '
                  f'{per} | {time.time() - t0:6.0f}s', flush=True)

    save_all(agents, env, args.ckpt_dir, args)
    env.close()
    np.save('logs/multi_reward_history.npy', np.array(reward_hist))
    np.save('logs/multi_wait_history.npy', np.array(wait_hist))
    print(f'Training complete. Checkpoints in {args.ckpt_dir}/ '
          f'(best-episode copy in {args.ckpt_dir}_best/)')


if __name__ == '__main__':
    main()
