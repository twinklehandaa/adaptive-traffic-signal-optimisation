"""console_multi_demo.py : console presentation of the CV -> DQN ->
actuation pipeline running at SEVERAL intersections at once.

Every simulated second prints one row per intersection (what its camera
saw, what its DQN decided, what was actuated, and its reward). Use
--focus <TLS_ID> to also print the full single-intersection-style
breakdown (per-lane CV output + reward formula) for one junction.

Usage:
    python3 console_multi_demo.py
    python3 console_multi_demo.py network_multi/grid.sumocfg checkpoints/multi --steps 300
    python3 console_multi_demo.py --focus A0 --delay 0.2
"""
import argparse
import time

import numpy as np

from baselines_multi import run_fixed_time_multi
from traffic_env_multi import MultiTrafficEnv
from train_multi import load_agents

W = 104


def actuation_label(info):
    if info['ignored_in_yellow']:
        return 'in yellow'
    if info['applied_switch']:
        return 'SWITCH' + (' (yellow)' if info['yellow_used'] else '')
    if info['blocked_by_min_green']:
        return 'BLOCKED min-green'
    if info['in_yellow']:
        return 'in yellow'
    return 'hold'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sumo_cfg', nargs='?', default='network_multi/grid.sumocfg')
    ap.add_argument('ckpt_dir', nargs='?', default='checkpoints/multi')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--delay', type=float, default=0.05)
    ap.add_argument('--focus', default=None, help='TLS id to show in full detail')
    ap.add_argument('--every', type=int, default=1, help='print every N steps')
    args = ap.parse_args()

    print('\n' + '=' * W)
    print(' SMART TRAFFIC SIGNALS | MULTI-INTERSECTION | CV -> DQN -> ACTUATION')
    print('=' * W)

    # BEFORE: fixed-time at every signal (own TraCI connection, closed before DQN starts)
    print('\n BEFORE: FIXED-TIME SIGNALS AT EVERY INTERSECTION')
    print('-' * W)
    base = run_fixed_time_multi(args.sumo_cfg, max_steps=args.steps, seed=1)
    print(f"Network average waiting time : {base['avg_wait']:.2f} s")
    print(f"Network average queue length : {base['avg_queue']:.2f} vehicles")
    print(f"Vehicles arrived             : {base['arrived_throughput']}")

    env = MultiTrafficEnv(args.sumo_cfg, max_steps=args.steps, seed=1, use_perception=True)
    states = env.reset()
    agents = load_agents(env, args.ckpt_dir)
    totals = {t: 0.0 for t in env.tls_ids}
    switches = {t: 0 for t in env.tls_ids}

    try:
        done = False
        while not done:
            step = env.step_count + 1
            q_vals, actions = {}, {}
            for t in env.tls_ids:
                q, _ = agents[t].q.forward(states[t][None, :])
                q_vals[t] = q[0]
                actions[t] = int(np.argmax(q[0]))
            cv_before = {t: env.ctrls[t].last_cv_output for t in env.tls_ids}
            phase_before = {t: env.ctrls[t].green_phases[env.ctrls[t].cur_phase_idx]
                            for t in env.tls_ids}

            states, rewards, done, infos = env.step(actions)
            for t in env.tls_ids:
                totals[t] += rewards[t]
                switches[t] += int(infos[t]['applied_switch'])

            if step % args.every:
                continue
            print('\n' + '─' * W)
            print(f'STEP {step:04d}   network queue={env.queue_log[-1]:.0f}   '
                  f'network wait={env.wait_log[-1]:.0f}s')
            print(f"{'TLS':<8}{'veh':>5}{'queued':>8}{'blobs':>7}{'nbr q':>7}"
                  f"{'phase':>7}{'Q(HOLD)':>10}{'Q(SWITCH)':>11}  {'DQN':<7}"
                  f"{'ACTUATION':<19}{'REWARD':>9}")
            for t in env.tls_ids:
                c = env.ctrls[t]
                cv = cv_before[t]
                veh = sum(cv[l]['count'] for l in c.lanes)
                qd = sum(cv[l]['queue'] for l in c.lanes)
                nq = sum(sum(cv_before[n][l]['queue'] for l in env.ctrls[n].lanes)
                         for n in c.neighbours)
                print(f"{t:<8}{veh:>5}{qd:>8}{cv.get('_detected_blobs_this_frame', 0):>7}"
                      f"{nq:>7}{phase_before[t]:>7}{q_vals[t][0]:>10.2f}{q_vals[t][1]:>11.2f}  "
                      f"{'HOLD' if actions[t] == 0 else 'SWITCH':<7}"
                      f"{actuation_label(infos[t]):<19}{rewards[t]:>9.2f}")

            if args.focus and args.focus in env.ctrls:
                c = env.ctrls[args.focus]
                rc = infos[args.focus]['reward_components']
                print(f'\n  [{args.focus}] per-lane CV output:')
                for l in c.lanes:
                    item = cv_before[args.focus][l]
                    print(f"    {l:<20} vehicles={item['count']:>2}  queued={item['queue']:>2}")
                print(f"  [{args.focus}] neighbours: {c.neighbours}")
                print(f"  [{args.focus}] R = -(Δwait) - 0.25(queue) + 0.10(throughput)"
                      f" = {rc['wait_term']:.2f} {rc['queue_term']:+.2f} "
                      f"{rc['throughput_term']:+.2f} = {rc['local']:.2f}"
                      + (f"  (+ neighbour share -> {rc['total']:.2f})"
                         if abs(rc['total'] - rc['local']) > 1e-9 else ''))
            if args.delay:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print('\nDemo stopped.')
    finally:
        after_wait = float(np.mean(env.wait_log)) if env.wait_log else 0.0
        after_queue = float(np.mean(env.queue_log)) if env.queue_log else 0.0
        pct = lambda b, a: (b - a) / b * 100.0 if b else 0.0

        print('\n' + '=' * W)
        print(' AFTER: DQN AT EVERY INTERSECTION  |  BEFORE vs AFTER')
        print('=' * W)
        print(f"{'METRIC':<30}{'BEFORE':>14}{'AFTER':>14}{'CHANGE':>14}{'IMPROVEMENT':>16}")
        print('-' * 88)
        print(f"{'Network avg waiting time':<30}{base['avg_wait']:>12.2f} s{after_wait:>12.2f} s"
              f"{after_wait - base['avg_wait']:>12.2f} s{pct(base['avg_wait'], after_wait):>15.1f}%")
        print(f"{'Network avg queue length':<30}{base['avg_queue']:>14.2f}{after_queue:>14.2f}"
              f"{after_queue - base['avg_queue']:>14.2f}{pct(base['avg_queue'], after_queue):>15.1f}%")
        print(f"{'Vehicles arrived':<30}{base['arrived_throughput']:>14}{env.arrived_total:>14}"
              f"{env.arrived_total - base['arrived_throughput']:>+14}")
        print('-' * 88)
        print(f"{'PER INTERSECTION (avg wait)':<30}{'BEFORE':>14}{'AFTER':>14}"
              f"{'IMPROVEMENT':>16}{'SWITCHES':>12}")
        for t in env.tls_ids:
            b = base['per_tls_avg_wait'].get(t, 0.0)
            a = float(np.mean(env.per_tls_wait[t])) if env.per_tls_wait[t] else 0.0
            print(f"{t:<30}{b:>12.2f} s{a:>12.2f} s{pct(b, a):>15.1f}%{switches[t]:>12}")
        print('=' * W)
        print('PIPELINE DEMONSTRATION COMPLETE')
        env.close()


if __name__ == '__main__':
    main()
