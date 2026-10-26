"""Console presentation of the CV -> DQN -> Actuation pipeline."""
import argparse
import os
import sys
import time
import numpy as np

sys.path.append(os.environ.get('SUMO_HOME', '/usr/share/sumo') + '/tools')
import traci

from traffic_env_generic import TrafficEnvGeneric
from dqn_agent import DQNAgent
from baselines_generic import run_fixed_time


def fmt_state(state, n):
    counts = state[:n] * 20.0
    queues = state[n:2*n] * 20.0
    return counts, queues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sumo_cfg', nargs='?', default='osm_real/osm.sumocfg')
    ap.add_argument('checkpoint', nargs='?', default='checkpoints/dqn_generic_latest.npz')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--delay', type=float, default=0.08)
    args = ap.parse_args()

    print('\n' + '=' * 96)
    print(' SMART TRAFFIC SIGNAL | CV -> DQN -> ACTUATION')
    print('=' * 96)
    print('Each cycle below is one complete AI decision pipeline.\n')

    # ---------------------------------------------------------------
    # BEFORE: genuine fixed-time baseline
    # ---------------------------------------------------------------
    # IMPORTANT: run the baseline BEFORE creating the DQN environment.
    # Both use the same SUMO seed, but each gets its own TraCI connection.
    # Starting two simulations on TraCI's default label causes:
    #   TraCIException: Connection 'default' is already active.
    print('\n' + '=' * 96)
    print(' BEFORE: FIXED-TIME TRAFFIC SIGNAL')
    print('=' * 96)
    print('Running the same SUMO scenario with the learned controller disabled...')
    baseline = run_fixed_time(args.sumo_cfg, max_steps=args.steps, seed=1)
    print(f"Average waiting time : {baseline['avg_wait']:.2f} s")
    print(f"Average queue length : {baseline['avg_queue']:.2f} vehicles")
    print(f"Maximum queue        : {baseline['max_queue']} vehicles")
    print(f"Vehicles arrived     : {baseline['arrived_throughput']}")
    print('=' * 96)

    # Start the DQN/SUMO environment only after the baseline connection
    # has been closed by run_fixed_time().
    env = TrafficEnvGeneric(sumo_cfg=args.sumo_cfg, gui=False, max_steps=args.steps, seed=1)
    state = env.reset()
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    agent.load(args.checkpoint)
    agent.eps = 0.0

    total_reward = 0.0
    previous_phase = env.cur_phase_idx

    try:
        while env.step_count < args.steps:
            step = env.step_count + 1
            cv = env.last_cv_output or {}
            n = len(env.lanes)
            counts, queues = fmt_state(state, n)

            q, _ = agent.q.forward(state[None, :])
            q_hold, q_switch = map(float, q[0])
            action = int(np.argmax(q[0]))
            action_name = 'HOLD' if action == 0 else 'SWITCH'

            current_phase = env.green_phases[env.cur_phase_idx]
            timer = float(state[-1] * 60.0)

            # ---------------- STEP 1: CV ----------------
            print('\n' + '─' * 96)
            print(f'STEP {step:04d}  |  1. CV DETECTION')
            print('─' * 96)
            print(f"Raw detected blobs : {cv.get('_detected_blobs_this_frame', 0)}")
            for i, lane in enumerate(env.lanes):
                item = cv.get(lane, {'count': 0, 'queue': 0})
                print(f"  {lane:<22} vehicles={item['count']:>2}   queued={item['queue']:>2}")
            print(f"CV output -> vehicle counts={counts.round(1).tolist()}")
            print(f"CV output -> queue counts  ={queues.round(1).tolist()}")

            # ---------------- STEP 2: DQN ----------------
            print('\n' + '─' * 96)
            print('2. DQN AGENT')
            print('─' * 96)
            print(f"Input state : counts + queues + phase one-hot + timer")
            print(f"State size  : {env.state_dim}")
            print('Network     : 2 hidden layers x 64 neurons, ReLU -> Q-values')
            print(f"Current phase={current_phase} | phase timer={timer:.0f}s")
            print(f"Q(HOLD)={q_hold:8.3f}   Q(SWITCH)={q_switch:8.3f}")
            print(f"Decision   : argmax(Q) -> {action_name}")

            # ---------------- ACTUATION + REWARD ----------------
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            switched = env.cur_phase_idx != previous_phase
            next_phase = env.green_phases[env.cur_phase_idx]
            rc = info['reward_components']

            print('\n' + '─' * 96)
            print('3. ACTUATION + REWARD')
            print('─' * 96)
            print(f"Requested action : {action_name}")
            if info['applied_switch']:
                print('Actuation        : SWITCH APPLIED')
                print(f"Phase change     : {current_phase} -> {next_phase}" + ('  (yellow transition)' if info['yellow_used'] else ''))
            elif info['blocked_by_min_green']:
                print('Actuation        : SWITCH BLOCKED - minimum green time not reached')
                print(f"Phase            : {current_phase} -> {next_phase}")
            else:
                print('Actuation        : HOLD CURRENT PHASE')
                print(f"Phase            : {current_phase} -> {next_phase}")
            print('')
            print('Reward formula:')
            print('  R = -(Δ waiting time) - 0.25(queue) + 0.10(throughput)')
            print(f"  waiting-time term = {rc['wait_term']:8.3f}")
            print(f"  queue penalty     = {rc['queue_term']:8.3f}   [queue={rc['queue']}]")
            print(f"  throughput bonus  = {rc['throughput_term']:8.3f}   [vehicles={rc['throughput']}]")
            print(f"  TOTAL REWARD      = {reward:8.3f}")
            print(f"Cumulative reward   = {total_reward:8.3f}")

            previous_phase = env.cur_phase_idx
            state = next_state
            if args.delay:
                time.sleep(args.delay)
            if done:
                break

    except KeyboardInterrupt:
        print('\nDemo stopped.')
    finally:
        # ---------------------------------------------------------------
        # AFTER + BEFORE/AFTER COMPARISON
        # ---------------------------------------------------------------
        after_wait = float(np.mean(env.wait_log)) if env.wait_log else 0.0
        after_queue = float(np.mean(env.queue_log)) if env.queue_log else 0.0
        wait_change = after_wait - baseline['avg_wait']
        queue_change = after_queue - baseline['avg_queue']
        wait_improvement = ((baseline['avg_wait'] - after_wait) / baseline['avg_wait'] * 100.0) if baseline['avg_wait'] else 0.0
        queue_improvement = ((baseline['avg_queue'] - after_queue) / baseline['avg_queue'] * 100.0) if baseline['avg_queue'] else 0.0

        print('\n' + '=' * 96)
        print(' AFTER: DQN CONTROL + BEFORE vs AFTER')
        print('=' * 96)
        print(f"{'METRIC':<28}{'BEFORE':>16}{'AFTER':>16}{'CHANGE':>16}{'IMPROVEMENT':>18}")
        print('-' * 94)
        print(f"{'Average waiting time':<28}{baseline['avg_wait']:>13.2f} s{after_wait:>13.2f} s{wait_change:>13.2f} s{wait_improvement:>16.1f}%")
        print(f"{'Average queue length':<28}{baseline['avg_queue']:>16.2f}{after_queue:>16.2f}{queue_change:>16.2f}{queue_improvement:>17.1f}%")
        print('=' * 96)
        print(f"DQN cumulative reward : {total_reward:.3f}")
        print(f"DQN vehicles arrived : {env.arrived_total}")
        print('\nINTERPRETATION')
        if wait_improvement > 0:
            print(f"  ✓ Waiting time reduced by {abs(wait_change):.2f} s ({wait_improvement:.1f}%).")
        else:
            print(f"  • Waiting time changed by {wait_change:+.2f} s ({wait_improvement:.1f}%).")
        if queue_improvement > 0:
            print(f"  ✓ Queue length reduced by {abs(queue_change):.2f} vehicles ({queue_improvement:.1f}%).")
        else:
            print(f"  • Queue length changed by {queue_change:+.2f} vehicles ({queue_improvement:.1f}%).")
        print('\nPIPELINE DEMONSTRATION COMPLETE')
        print('=' * 96)
        env.close()


if __name__ == '__main__':
    main()
