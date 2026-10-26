"""baselines.py : fixed-time and actuated (gap-out) baseline controllers.

Run on the identical network + seeded demand as the DQN so the three
controllers (fixed-time, actuated, DQN) can be compared fairly on the
same traffic scenario, as Section 11.4 / Fig 11.6 of the report requires.
"""
import os
import sys

import numpy as np

sys.path.append(os.environ.get('SUMO_HOME', '/usr/share/sumo') + '/tools')
import traci  # noqa: E402

LANES = ['N1_0', 'N1_1', 'E1_0', 'E1_1', 'S1_0', 'S1_1', 'W1_0', 'W1_1']


def run_fixed_time(sumo_cfg, tls_id='J1', max_steps=1200, seed=0):
    """The network's built-in static tlLogic (42s green / 3s yellow, fixed cycle)."""
    args = ['sumo', '-c', sumo_cfg, '--no-step-log', 'true',
            '--waiting-time-memory', '10000', '--time-to-teleport', '300',
            '--seed', str(seed)]
    traci.start(args)
    wait_samples, queue_samples, arrived = [], [], 0
    for _ in range(max_steps):
        traci.simulationStep()
        w = sum(traci.lane.getWaitingTime(l) for l in LANES)
        q = sum(traci.lane.getLastStepHaltingNumber(l) for l in LANES)
        arrived += traci.simulation.getArrivedNumber()
        wait_samples.append(w)
        queue_samples.append(q)
    traci.close()
    return {
        'avg_wait': float(np.mean(wait_samples)),
        'avg_queue': float(np.mean(queue_samples)),
        'max_queue': int(np.max(queue_samples)),
        'arrived_throughput': arrived,
    }


def run_actuated(sumo_cfg, tls_id='J1', max_steps=1200, seed=0,
                  min_green=10, max_green=50, yellow=3, gap=3):
    """Simple gap-out actuated control: extend the green phase while a
    vehicle is still detected within `gap` seconds of clearing, else
    switch, subject to min/max green -- the classic rule-based baseline
    used alongside fixed-time in the literature (Section 6)."""
    args = ['sumo', '-c', sumo_cfg, '--no-step-log', 'true',
            '--waiting-time-memory', '10000', '--time-to-teleport', '300',
            '--seed', str(seed)]
    traci.start(args)
    phases = [0, 2]
    cur_phase, phase_timer = 0, 0
    traci.trafficlight.setPhase(tls_id, 0)
    approach_lanes = {0: ['N1_0', 'N1_1', 'S1_0', 'S1_1'], 2: ['E1_0', 'E1_1', 'W1_0', 'W1_1']}

    wait_samples, queue_samples, arrived = [], [], 0
    step = 0
    while step < max_steps:
        active = sum(traci.lane.getLastStepVehicleNumber(l) for l in approach_lanes[cur_phase])
        should_switch = (phase_timer >= min_green and active == 0) or phase_timer >= max_green
        if should_switch:
            traci.trafficlight.setPhase(tls_id, cur_phase + 1)
            for _ in range(yellow):
                traci.simulationStep()
                step += 1
                arrived += traci.simulation.getArrivedNumber()
                wait_samples.append(sum(traci.lane.getWaitingTime(l) for l in LANES))
                queue_samples.append(sum(traci.lane.getLastStepHaltingNumber(l) for l in LANES))
            nxt = phases[(phases.index(cur_phase) + 1) % len(phases)]
            traci.trafficlight.setPhase(tls_id, nxt)
            cur_phase, phase_timer = nxt, 0
        traci.simulationStep()
        step += 1
        phase_timer += 1
        arrived += traci.simulation.getArrivedNumber()
        wait_samples.append(sum(traci.lane.getWaitingTime(l) for l in LANES))
        queue_samples.append(sum(traci.lane.getLastStepHaltingNumber(l) for l in LANES))
    traci.close()
    return {
        'avg_wait': float(np.mean(wait_samples)),
        'avg_queue': float(np.mean(queue_samples)),
        'max_queue': int(np.max(queue_samples)),
        'arrived_throughput': arrived,
    }
