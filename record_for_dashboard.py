"""record_for_dashboard.py : records every piece of data the dashboard
needs from a real (or synthetic) run: fixed-time vs DQN replay
trajectories (for the Live Replay tab), the real road geometry near the
junction, and a step-by-step state/action/reward trace for the trained
DQN (for the "How CV + DQN work" explainer tab).

Usage:
    python3 record_for_dashboard.py path/to/osm.sumocfg path/to/osm.net.xml[.gz] checkpoints/dqn_generic_latest.npz
"""
import sys
import os
import json

sys.path.append(os.environ.get('SUMO_HOME', '/usr/share/sumo') + '/tools')
import traci
import sumolib

from traffic_env_generic import TrafficEnvGeneric, discover_intersection
from baselines_generic import force_static_program
from dqn_agent import DQNAgent

SUMO_CFG = sys.argv[1] if len(sys.argv) > 1 else 'osm_real/osm.sumocfg'
NET_FILE = sys.argv[2] if len(sys.argv) > 2 else 'osm_real/osm.net.xml.gz'
CHECKPOINT = sys.argv[3] if len(sys.argv) > 3 else 'checkpoints/dqn_generic_latest.npz'
SEED = 2
DURATION = 1000
RADIUS = 220  # metres around the junction to keep in the replay (keeps file size sane)

os.makedirs('logs', exist_ok=True)
os.makedirs('screens', exist_ok=True)

# ---------------------------------------------------------------
# 0. find the junction centre (everything below is recorded relative to it)
# ---------------------------------------------------------------
traci.start(['sumo', '-c', SUMO_CFG, '--no-step-log', 'true', '--seed', str(SEED)])
traci.simulationStep()
tls_id, lanes, green_phases = discover_intersection()
JX, JY = traci.junction.getPosition(tls_id)
traci.close()
print(f'Junction: {tls_id}  |  lanes: {len(lanes)}  |  green phases: {green_phases}')

# ---------------------------------------------------------------
# 1. real road geometry near the junction (for realistic replay rendering)
# ---------------------------------------------------------------
net = sumolib.net.readNet(NET_FILE)
road_lines = []
for edge in net.getEdges():
    shape = edge.getShape()
    xs = [p[0] for p in shape]
    ys = [p[1] for p in shape]
    if max(abs(x - JX) for x in xs) > RADIUS + 60 and max(abs(y - JY) for y in ys) > RADIUS + 60:
        continue
    road_lines.append([[round(x - JX, 1), round(y - JY, 1)] for x, y in shape])
with open('logs/road_lines.json', 'w') as f:
    json.dump(road_lines, f)
print('saved logs/road_lines.json |', len(road_lines), 'segments')

with open('logs/tls_meta.json', 'w') as f:
    json.dump({'tls_id': tls_id, 'lanes': lanes, 'green_phases': green_phases}, f)


def snapshot(lane_list, tls):
    vlist = []
    for v in traci.vehicle.getIDList():
        x, y = traci.vehicle.getPosition(v)
        if abs(x - JX) > RADIUS or abs(y - JY) > RADIUS:
            continue
        vlist.append((v, x, y, traci.vehicle.getAngle(v)))
    q = sum(traci.lane.getLastStepHaltingNumber(l) for l in lane_list)
    w = sum(traci.lane.getWaitingTime(l) for l in lane_list)
    ph = traci.trafficlight.getPhase(tls)
    return vlist, q, w, ph


def compact_ids(frames):
    id_map = {}
    for fr in frames:
        new_v = []
        for vid, x, y, ang in fr['v']:
            if vid not in id_map:
                id_map[vid] = len(id_map)
            new_v.append([id_map[vid], round(x - JX, 1), round(y - JY, 1), ang])
        fr['v'] = new_v
    return frames


# ---------------------------------------------------------------
# 2. fixed-time replay
# ---------------------------------------------------------------
print('recording fixed-time replay...')
traci.start(['sumo', '-c', SUMO_CFG, '--no-step-log', 'true',
             '--waiting-time-memory', '10000', '--time-to-teleport', '300', '--seed', str(SEED)])
traci.simulationStep()
tls_id, lanes, green_phases = discover_intersection()
force_static_program(tls_id)
fixed_frames = []
for t in range(DURATION):
    traci.simulationStep()
    v, q, w, ph = snapshot(lanes, tls_id)
    fixed_frames.append({'t': t, 'v': v, 'q': q, 'w': round(w, 1), 'ph': ph})
traci.close()
fixed_frames = compact_ids(fixed_frames)
with open('logs/replay_fixed.json', 'w') as f:
    json.dump(fixed_frames, f)
print('saved logs/replay_fixed.json |', len(fixed_frames), 'frames')

# ---------------------------------------------------------------
# 3. DQN replay + explainer trace (state/action/reward per step)
# ---------------------------------------------------------------
print('recording DQN replay + explainer trace...')
env = TrafficEnvGeneric(sumo_cfg=SUMO_CFG, gui=False, max_steps=DURATION, seed=SEED)
state = env.reset()
agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
agent.load(CHECKPOINT)
agent.eps = 0.0

n_lanes = len(env.lanes)
dqn_frames = []
explain_frames = []
id_map = {}
done = False
prev_phase_idx = env.cur_phase_idx
while not done and env.step_count < DURATION:
    a = agent.act(state)
    counts = [round(float(state[i]) * 20, 1) for i in range(n_lanes)]
    queues = [round(float(state[n_lanes + i]) * 20, 1) for i in range(n_lanes)]

    nxt_state, reward, done, info = env.step(a)

    vlist_raw = []
    for v in traci.vehicle.getIDList():
        x, y = traci.vehicle.getPosition(v)
        if abs(x - JX) > RADIUS or abs(y - JY) > RADIUS:
            continue
        if v not in id_map:
            id_map[v] = len(id_map)
        vlist_raw.append([id_map[v], round(x - JX, 1), round(y - JY, 1), round(traci.vehicle.getAngle(v), 1)])

    q_total = sum(traci.lane.getLastStepHaltingNumber(l) for l in env.lanes)
    w_total = sum(traci.lane.getWaitingTime(l) for l in env.lanes)
    cur_phase_value = env.green_phases[env.cur_phase_idx]

    dqn_frames.append({'t': env.step_count, 'v': vlist_raw, 'q': q_total, 'w': round(w_total, 1), 'ph': cur_phase_value})
    explain_frames.append({
        't': env.step_count, 'v': vlist_raw, 'ph': cur_phase_value,
        'a': int(a), 'switched': int(env.cur_phase_idx != prev_phase_idx),
        'r': round(float(reward), 1), 'counts': counts, 'queues': queues,
    })
    prev_phase_idx = env.cur_phase_idx
    state = nxt_state

env.close()
with open('logs/replay_dqn.json', 'w') as f:
    json.dump(dqn_frames, f)
with open('logs/explain_dqn.json', 'w') as f:
    json.dump({'lanes': env.lanes, 'frames': explain_frames}, f)
print('saved logs/replay_dqn.json + logs/explain_dqn.json |', len(dqn_frames), 'frames')
print('Done. Next: run build_dashboard.py')
