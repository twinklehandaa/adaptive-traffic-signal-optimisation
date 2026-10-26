"""make_figures_generic.py : network topology + CV detection screenshots
for ANY network (point it at your osm.sumocfg).

Usage:
    python3 make_figures_generic.py path/to/osm.sumocfg path/to/osm.net.xml[.gz]
"""
import sys
import os
sys.path.append(os.environ.get('SUMO_HOME', '/usr/share/sumo') + '/tools')
import traci
import sumolib
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from traffic_env_generic import discover_intersection
from perception.perception import LaneDetector

SUMO_CFG = sys.argv[1] if len(sys.argv) > 1 else 'osm_real/osm.sumocfg'
NET_FILE = sys.argv[2] if len(sys.argv) > 2 else 'osm_real/osm.net.xml.gz'
SEED = 7
WARMUP_SECONDS = 400  # let traffic build up before capturing

# ---- 1. network topology figure ----
net = sumolib.net.readNet(NET_FILE)
traci.start(['sumo', '-c', SUMO_CFG, '--no-step-log', 'true', '--seed', str(SEED)])
traci.simulationStep()
tls_id, lanes, green_phases = discover_intersection()
jx, jy = traci.junction.getPosition(tls_id)
traci.close()

RADIUS = 180
fig, ax = plt.subplots(figsize=(7, 7))
for edge in net.getEdges():
    shape = edge.getShape()
    xs = [p[0] for p in shape]; ys = [p[1] for p in shape]
    if max(abs(x - jx) for x in xs) > RADIUS + 100 and max(abs(y - jy) for y in ys) > RADIUS + 100:
        continue
    ax.plot(xs, ys, color='#4a4a4a', linewidth=4, solid_capstyle='round', zorder=1, alpha=0.85)
ax.scatter([jx], [jy], s=260, color='#e74c3c', zorder=3)
ax.annotate('Signalised\nintersection', (jx, jy), textcoords="offset points", xytext=(14, 10), fontsize=10, weight='bold')
ax.set_xlim(jx - RADIUS, jx + RADIUS); ax.set_ylim(jy - RADIUS, jy + RADIUS)
ax.set_title('Real intersection imported from OpenStreetMap', fontsize=13, weight='bold')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_aspect('equal'); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('screens/osm_network.png', dpi=150)
print('saved screens/osm_network.png  (TLS:', tls_id, '| lanes:', len(lanes), '| green phases:', len(green_phases), ')')
plt.close(fig)

# ---- 2. CV detection screenshot ----
detector = LaneDetector(lanes, origin=(jx, jy))
traci.start(['sumo', '-c', SUMO_CFG, '--no-step-log', 'true', '--seed', str(SEED)])
frame = None
out = None
for step in range(WARMUP_SECONDS):
    traci.simulationStep()
    positions = []
    dists = {}
    for lane in lanes:
        d = []
        lane_len = traci.lane.getLength(lane)
        for vid in traci.lane.getLastStepVehicleIDs(lane):
            lane_pos = traci.vehicle.getLanePosition(vid)
            d.append(max(0.0, lane_len - lane_pos))
            x, y = traci.vehicle.getPosition(vid)
            positions.append((x, y, 0))
        dists[lane] = d
    frame = detector.render_frame(positions)
    out = detector.detect(frame, dists)
traci.close()

mask = detector.bg.apply(frame)
mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vis = frame.copy()
for c in contours:
    if cv2.contourArea(c) > 18:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)

y0 = 16
for lane, vals in out.items():
    if lane.startswith('_'):
        continue
    label = lane if len(lane) < 16 else lane[:13] + '...'
    cv2.putText(vis, f'{label}: n={vals["count"]} q={vals["queue"]}', (6, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    y0 += 15
cv2.putText(vis, f'CV Lane-Occupancy Detector (t={WARMUP_SECONDS}s)', (6, 465),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite('screens/osm_cv_detection.png', vis)
print('saved screens/osm_cv_detection.png')
