"""traffic_env.py : SUMO environment wrapper.

Gym-style reset/step interface over TraCI. The observation is built by
running the CV perception module (perception.LaneDetector) on a
rendered top-down frame each step, rather than pulling vehicle counts
straight out of TraCI -- so the RL agent only ever sees what the vision
layer reports, matching the layered CV+DRL architecture in the report.
"""
import os
import sys
import numpy as np

sys.path.append(os.environ.get('SUMO_HOME', '/usr/share/sumo') + '/tools')
import traci  # noqa: E402

from perception.perception import LaneDetector  # noqa: E402

LANES = ['N1_0', 'N1_1', 'E1_0', 'E1_1', 'S1_0', 'S1_1', 'W1_0', 'W1_1']
APPROACH_LANES = {
    'N1': ['N1_0', 'N1_1'], 'E1': ['E1_0', 'E1_1'],
    'S1': ['S1_0', 'S1_1'], 'W1': ['W1_0', 'W1_1'],
}
APPROACHES = ['N1', 'E1', 'S1', 'W1']


class TrafficEnv:
    """Gym-style wrapper around a single signalised SUMO intersection."""

    def __init__(self, sumo_cfg, tls_id='J1', gui=False,
                 min_green=10, yellow=3, max_steps=2400, seed=None):
        self.sumo_bin = 'sumo-gui' if gui else 'sumo'
        self.sumo_cfg = sumo_cfg
        self.tls_id = tls_id
        self.min_green = min_green
        self.yellow = yellow
        self.max_steps = max_steps
        self.seed = seed
        self.phases = [0, 2]  # NS-Green, EW-Green (even index = green in net.xml)
        self.detector = LaneDetector(APPROACHES)
        self.state_dim = 4 + 4 + 2 + 1  # count+queue per approach, phase one-hot, timer
        self.action_dim = 2             # 0 = hold, 1 = switch (subject to min green)

    def reset(self):
        if traci.isLoaded():
            traci.close()
        args = [self.sumo_bin, '-c', self.sumo_cfg, '--no-step-log', 'true',
                '--waiting-time-memory', '10000', '--time-to-teleport', '300']
        if self.seed is not None:
            args += ['--seed', str(self.seed)]
        traci.start(args)
        traci.trafficlight.setPhase(self.tls_id, 0)
        self.step_count = 0
        self.cur_phase = 0
        self.phase_timer = 0
        self.prev_wait = 0.0
        self.cum_wait = 0.0
        self.cum_through = 0
        self.wait_log = []
        self.queue_log = []
        self.arrived_total = 0
        return self._get_state()

    def _approach_vehicle_distances(self):
        """{approach_edge: [distance_to_stopline_m, ...]} for the perception frame."""
        out = {}
        positions = []
        for edge in APPROACH_LANES:
            dists = []
            for lane in APPROACH_LANES[edge]:
                for vid in traci.lane.getLastStepVehicleIDs(lane):
                    lane_len = traci.lane.getLength(lane)
                    lane_pos = traci.vehicle.getLanePosition(vid)
                    dist_to_stop = max(0.0, lane_len - lane_pos)
                    dists.append(dist_to_stop)
                    x, y = traci.vehicle.getPosition(vid)
                    angle = traci.vehicle.getAngle(vid)
                    positions.append((x, y, angle))
            out[edge] = dists
        return out, positions

    def _get_state(self):
        approach_dists, positions = self._approach_vehicle_distances()
        frame = self.detector.render_frame(positions)
        cv_out = self.detector.detect(frame, approach_dists)

        counts = [cv_out[a]['count'] for a in APPROACHES]
        queues = [cv_out[a]['queue'] for a in APPROACHES]
        phase = [1.0 if self.cur_phase == p else 0.0 for p in self.phases]
        timer = [min(self.phase_timer / 60.0, 1.0)]

        state = np.array(counts + queues + phase + timer, dtype=np.float32)
        state[:8] = state[:8] / 20.0
        self._last_cv = cv_out
        return state

    def _reward(self):
        wait = sum(traci.lane.getWaitingTime(l) for l in LANES)
        queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in LANES)
        thru = sum(traci.lane.getLastStepVehicleNumber(l) for l in LANES)
        r = -(wait - self.prev_wait) - 0.25 * queue + 0.10 * thru
        self.prev_wait = wait
        self.cum_wait = wait
        self.cum_through += thru
        return float(r)

    def _log_step(self):
        """Record per-second metrics -- called after every raw traci.simulationStep()
        so evaluation runs can be sampled at the same 1-second granularity as
        the baseline controllers, regardless of how many raw seconds a given
        env.step() call consumed (1 normally, or +yellow when switching)."""
        w = sum(traci.lane.getWaitingTime(l) for l in LANES)
        q = sum(traci.lane.getLastStepHaltingNumber(l) for l in LANES)
        self.wait_log.append(w)
        self.queue_log.append(q)
        self.arrived_total += traci.simulation.getArrivedNumber()

    def step(self, action):
        """action 0 = keep current phase, 1 = switch to the other phase."""
        target = self.cur_phase if action == 0 else \
            self.phases[(self.phases.index(self.cur_phase) + 1) % len(self.phases)]

        if target != self.cur_phase and self.phase_timer >= self.min_green:
            traci.trafficlight.setPhase(self.tls_id, self.cur_phase + 1)  # yellow
            for _ in range(self.yellow):
                traci.simulationStep()
                self.step_count += 1
                self._log_step()
            traci.trafficlight.setPhase(self.tls_id, target)
            self.cur_phase, self.phase_timer = target, 0
        traci.simulationStep()
        self.step_count += 1
        self._log_step()
        self.phase_timer += 1

        state = self._get_state()
        reward = self._reward()
        done = (self.step_count >= self.max_steps or
                traci.simulation.getMinExpectedNumber() <= 0)
        info = {'step': self.step_count, 'cum_wait': self.cum_wait,
                'cum_through': self.cum_through}
        return state, reward, done, info

    def close(self):
        if traci.isLoaded():
            traci.close()
