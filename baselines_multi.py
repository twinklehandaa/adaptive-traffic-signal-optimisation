"""baselines_multi.py : fixed-time, actuated and DQN controllers on a
multi-intersection network, measured the same way (sum of waiting time
and halting vehicles over every controlled lane of every signal).

Usage:
    python3 baselines_multi.py                                  # 3 seeds, grid network
    python3 baselines_multi.py network_multi/grid.sumocfg --seeds 0 1 2 --ckpt-dir checkpoints/multi
Baselines: fixed_time (network's own 42 s greens), fixed_short (same
program with a shorter green, tuned on a held-out seed), actuated, dqn.
Writes logs/multi_comparison_summary.json and screens/multi_comparison.png
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.environ.get('SUMO_HOME', '/usr/share/sumo') + '/tools')
import traci  # noqa: E402

from traffic_env_generic import discover_intersection  # noqa: E402
from baselines_generic import force_static_program  # noqa: E402
from traffic_env_multi import MultiTrafficEnv  # noqa: E402


def _summary(wait_log, queue_log, arrived, per_tls_wait=None):
    out = {'avg_wait': float(np.mean(wait_log)), 'avg_queue': float(np.mean(queue_log)),
           'max_queue': int(np.max(queue_log)), 'arrived_throughput': int(arrived)}
    if per_tls_wait:
        out['per_tls_avg_wait'] = {t: float(np.mean(v)) for t, v in per_tls_wait.items()}
    return out


def force_static_green(tls_id, green_time):
    """Same phase order as the network's program, but every green phase
    lasts green_time seconds (yellows unchanged). A shorter, well-tuned
    cycle is a much fairer fixed-time reference than the 42 s default."""
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phases = []
    for p in logic.phases:
        d = p.duration if 'y' in p.state or not any(ch in 'Gg' for ch in p.state) else green_time
        phases.append(traci.trafficlight.Phase(duration=d, minDur=d, maxDur=d, state=p.state))
    traci.trafficlight.setProgramLogic(tls_id, traci.trafficlight.Logic('static_short', 0, 0, phases=phases))
    traci.trafficlight.setProgram(tls_id, 'static_short')


def run_fixed_time_multi(sumo_cfg, max_steps=1200, seed=0, tls_ids=None, green_time=None):
    """Every signal runs a static program (actuation disabled). green_time=None
    keeps the network's own green durations; a number overrides them."""
    traci.start(['sumo', '-c', sumo_cfg, '--no-step-log', 'true',
                 '--waiting-time-memory', '10000', '--time-to-teleport', '300',
                 '--seed', str(seed)])
    traci.simulationStep()
    ids = tls_ids or list(traci.trafficlight.getIDList())
    lanes = {}
    for tid in ids:
        try:
            _, lanes[tid], _ = discover_intersection(tid)
            if green_time is None:
                force_static_program(tid)
            else:
                force_static_green(tid, green_time)
        except RuntimeError:
            pass
    wait_log, queue_log, arrived = [], [], 0
    per_tls = {t: [] for t in lanes}
    for _ in range(max_steps):
        traci.simulationStep()
        tw = tq = 0.0
        for t, ls in lanes.items():
            w = sum(traci.lane.getWaitingTime(l) for l in ls)
            per_tls[t].append(w)
            tw += w
            tq += sum(traci.lane.getLastStepHaltingNumber(l) for l in ls)
        wait_log.append(tw)
        queue_log.append(tq)
        arrived += traci.simulation.getArrivedNumber()
        if traci.simulation.getMinExpectedNumber() <= 0:
            break
    traci.close()
    return _summary(wait_log, queue_log, arrived, per_tls)


def run_actuated_multi(sumo_cfg, max_steps=1200, seed=0, max_green=50, tls_ids=None):
    """Gap-out actuated control at every signal (switch when the served
    approach empties after min-green, or at max-green)."""
    env = MultiTrafficEnv(sumo_cfg, max_steps=max_steps, seed=seed,
                          tls_ids=tls_ids, use_perception=False)
    env.reset()
    done = False
    while not done:
        actions = {}
        for tid, c in env.ctrls.items():
            served = c.phase_lanes[c.green_phases[c.cur_phase_idx]]
            active = sum(traci.lane.getLastStepVehicleNumber(l) for l in served)
            actions[tid] = int((c.phase_timer >= c.min_green and active == 0)
                               or c.phase_timer >= max_green)
        _, _, done, _ = env.step(actions)
    out = _summary(env.wait_log, env.queue_log, env.arrived_total, env.per_tls_wait)
    env.close()
    return out


def run_dqn_multi(sumo_cfg, ckpt_dir, max_steps=1200, seed=0, tls_ids=None):
    from train_multi import load_agents
    env = MultiTrafficEnv(sumo_cfg, max_steps=max_steps, seed=seed,
                          tls_ids=tls_ids, use_perception=False)
    states = env.reset()
    agents = load_agents(env, ckpt_dir)
    done = False
    while not done:
        actions = {t: agents[t].act(states[t]) for t in env.tls_ids}
        states, _, done, _ = env.step(actions)
    out = _summary(env.wait_log, env.queue_log, env.arrived_total, env.per_tls_wait)
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sumo_cfg', nargs='?', default='network_multi/grid.sumocfg')
    ap.add_argument('--ckpt-dir', default='checkpoints/multi')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--max-steps', type=int, default=1200)
    ap.add_argument('--short-green', type=int, default=None,
                    help='green time for the short-cycle fixed baseline '
                         '(default: tuned automatically on seed 99)')
    args = ap.parse_args()

    if args.short_green is None:
        # tune on a seed NOT used for evaluation, so the baseline is fair
        best = None
        for g in (10, 15, 20, 25, 30):
            w = run_fixed_time_multi(args.sumo_cfg, args.max_steps, 99, green_time=g)['avg_wait']
            if best is None or w < best[1]:
                best = (g, w)
        args.short_green = best[0]
        print(f'tuned fixed_short green time: {args.short_green} s')

    controllers = {
        'fixed_time': lambda s: run_fixed_time_multi(args.sumo_cfg, args.max_steps, s),
        'fixed_short': lambda s: run_fixed_time_multi(args.sumo_cfg, args.max_steps, s,
                                                      green_time=args.short_green),
        'actuated': lambda s: run_actuated_multi(args.sumo_cfg, args.max_steps, s),
    }
    if os.path.exists(os.path.join(args.ckpt_dir, 'meta.json')):
        controllers['dqn'] = lambda s: run_dqn_multi(args.sumo_cfg, args.ckpt_dir,
                                                     args.max_steps, s)
    else:
        print(f'(no trained agents in {args.ckpt_dir}/ -- run train_multi.py first; '
              f'comparing baselines only)')

    results = {}
    for name, fn in controllers.items():
        runs = [fn(s) for s in args.seeds]
        results[name] = {
            'avg_wait': float(np.mean([r['avg_wait'] for r in runs])),
            'avg_queue': float(np.mean([r['avg_queue'] for r in runs])),
            'arrived_throughput': float(np.mean([r['arrived_throughput'] for r in runs])),
            'per_tls_avg_wait': {t: float(np.mean([r['per_tls_avg_wait'][t] for r in runs]))
                                 for t in runs[0]['per_tls_avg_wait']},
            'runs': runs,
        }
        r = results[name]
        print(f'{name:<11} wait {r["avg_wait"]:8.1f} s | queue {r["avg_queue"]:6.2f} | '
              f'arrived {r["arrived_throughput"]:6.0f} | per signal: ' +
              '  '.join(f'{t}={w:.1f}' for t, w in r['per_tls_avg_wait'].items()))

    os.makedirs('logs', exist_ok=True)
    with open('logs/multi_comparison_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('saved logs/multi_comparison_summary.json')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names = list(results)
        tls = list(results[names[0]]['per_tls_avg_wait'])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(names, [results[n]['avg_wait'] for n in names],
                    color=['#999999', '#666666', '#e0a040', '#3a7bd5'][:len(names)])
        axes[0].set_title('Network average waiting time')
        axes[0].set_ylabel('summed lane waiting time (s)')
        w = 0.8 / len(names)
        x = np.arange(len(tls))
        for i, n in enumerate(names):
            axes[1].bar(x + i * w, [results[n]['per_tls_avg_wait'][t] for t in tls], w, label=n)
        axes[1].set_xticks(x + w * (len(names) - 1) / 2)
        axes[1].set_xticklabels(tls)
        axes[1].set_title('Average waiting time per intersection')
        axes[1].legend()
        fig.tight_layout()
        os.makedirs('screens', exist_ok=True)
        fig.savefig('screens/multi_comparison.png', dpi=130)
        print('saved screens/multi_comparison.png')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
