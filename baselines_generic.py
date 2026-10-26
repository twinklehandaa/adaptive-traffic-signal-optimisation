"""baselines_generic.py : fixed-time and actuated baselines for ANY
single-intersection network, using the same auto-discovery as
traffic_env_generic.py instead of hardcoded lane names.
"""
import os
import sys

import numpy as np

sys.path.append(os.environ.get('SUMO_HOME', '/usr/share/sumo') + '/tools')
import traci  # noqa: E402

from traffic_env_generic import discover_intersection  # noqa: E402


def _metrics(lanes):
    w = sum(traci.lane.getWaitingTime(l) for l in lanes)
    q = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
    return w, q


def force_static_program(tls_id):
    """Real OSM networks are often generated with an 'actuated' tlLogic
    (gap-out extension baked into the network itself). To get a genuine
    fixed-time baseline for comparison, rebuild the same phase list as a
    plain static program (dropping min/max duration) and switch to it."""
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    new_phases = [traci.trafficlight.Phase(duration=p.duration, minDur=p.duration,
                                            maxDur=p.duration, state=p.state)
                  for p in logic.phases]
    new_logic = traci.trafficlight.Logic('static_fixed', 0, 0, phases=new_phases)
    traci.trafficlight.setProgramLogic(tls_id, new_logic)
    traci.trafficlight.setProgram(tls_id, 'static_fixed')


def run_fixed_time(sumo_cfg, max_steps=1200, seed=0):
    """A genuine fixed-time baseline: same phase durations as the network's
    own program, but with actuation disabled (min=max=nominal duration)."""
    args = ['sumo', '-c', sumo_cfg, '--no-step-log', 'true',
            '--waiting-time-memory', '10000', '--time-to-teleport', '300',
            '--seed', str(seed)]
    traci.start(args)
    traci.simulationStep()
    tls_id, lanes, green_phases = discover_intersection()
    force_static_program(tls_id)

    wait_samples, queue_samples, arrived = [], [], 0
    for _ in range(max_steps):
        traci.simulationStep()
        w, q = _metrics(lanes)
        arrived += traci.simulation.getArrivedNumber()
        wait_samples.append(w)
        queue_samples.append(q)
    traci.close()
    return {'avg_wait': float(np.mean(wait_samples)), 'avg_queue': float(np.mean(queue_samples)),
            'max_queue': int(np.max(queue_samples)), 'arrived_throughput': arrived}


def run_native_actuated(sumo_cfg, max_steps=1200, seed=0):
    """Run the network's OWN built-in program untouched. For a real OSM
    network generated with an actuated tlLogic (as osmWebWizard typically
    does), this is a genuine, properly-configured actuated controller --
    better than a hand-rolled approximation. For a plain static network
    this is equivalent to fixed-time."""
    args = ['sumo', '-c', sumo_cfg, '--no-step-log', 'true',
            '--waiting-time-memory', '10000', '--time-to-teleport', '300',
            '--seed', str(seed)]
    traci.start(args)
    traci.simulationStep()
    tls_id, lanes, green_phases = discover_intersection()

    wait_samples, queue_samples, arrived = [], [], 0
    for _ in range(max_steps):
        traci.simulationStep()
        w, q = _metrics(lanes)
        arrived += traci.simulation.getArrivedNumber()
        wait_samples.append(w)
        queue_samples.append(q)
    traci.close()
    return {'avg_wait': float(np.mean(wait_samples)), 'avg_queue': float(np.mean(queue_samples)),
            'max_queue': int(np.max(queue_samples)), 'arrived_throughput': arrived}


def run_actuated(sumo_cfg, max_steps=1200, seed=0, min_green=10, max_green=50, yellow_fallback=3):
    """Hand-rolled gap-out actuated control, generalised to however many
    green phases the discovered TLS has. Kept for networks whose native
    program is plain static (e.g. the bundled synthetic network) so there
    is still a meaningful 'actuated' baseline to compare against; for a
    network whose native program is already type='actuated', prefer
    run_native_actuated instead."""
    args = ['sumo', '-c', sumo_cfg, '--no-step-log', 'true',
            '--waiting-time-memory', '10000', '--time-to-teleport', '300',
            '--seed', str(seed)]
    traci.start(args)
    traci.simulationStep()
    tls_id, lanes, green_phases = discover_intersection()

    # lanes actually served (green, non-yellow) by each green phase, so the
    # gap-out rule looks at the right lanes regardless of network layout
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    controlled = traci.trafficlight.getControlledLanes(tls_id)
    phase_lanes = {}
    for gp in green_phases:
        state = logic.phases[gp].state
        served = {controlled[i] for i, ch in enumerate(state) if ch in ('G', 'g')}
        phase_lanes[gp] = [l for l in lanes if l in served]

    cur_idx, phase_timer = 0, 0
    traci.trafficlight.setPhase(tls_id, green_phases[0])

    def next_yellow(green_phase_idx):
        n = len(logic.phases)
        nxt = (green_phase_idx + 1) % n
        return nxt if 'y' in logic.phases[nxt].state else None

    wait_samples, queue_samples, arrived = [], [], 0
    step = 0
    while step < max_steps:
        active = sum(traci.lane.getLastStepVehicleNumber(l) for l in phase_lanes[green_phases[cur_idx]])
        should_switch = (phase_timer >= min_green and active == 0) or phase_timer >= max_green
        if should_switch:
            yellow = next_yellow(green_phases[cur_idx])
            if yellow is not None:
                traci.trafficlight.setPhase(tls_id, yellow)
                for _ in range(yellow_fallback):
                    traci.simulationStep()
                    step += 1
                    w, q = _metrics(lanes)
                    arrived += traci.simulation.getArrivedNumber()
                    wait_samples.append(w)
                    queue_samples.append(q)
            cur_idx = (cur_idx + 1) % len(green_phases)
            traci.trafficlight.setPhase(tls_id, green_phases[cur_idx])
            phase_timer = 0
        traci.simulationStep()
        step += 1
        phase_timer += 1
        w, q = _metrics(lanes)
        arrived += traci.simulation.getArrivedNumber()
        wait_samples.append(w)
        queue_samples.append(q)
    traci.close()
    return {'avg_wait': float(np.mean(wait_samples)), 'avg_queue': float(np.mean(queue_samples)),
            'max_queue': int(np.max(queue_samples)), 'arrived_throughput': arrived}
