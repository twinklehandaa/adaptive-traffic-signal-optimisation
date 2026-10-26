"""traffic_env_multi.py : SUMO environment for MULTIPLE signalised
intersections, controlled at the same time by one DQN agent each.

Builds directly on traffic_env_generic.py: every traffic light in the
network is auto-discovered with the same discover_intersection() logic
(TLS id, controlled lanes, green phases), gets its own CV LaneDetector
centred on its own junction, and keeps its own phase/min-green/yellow
state. The differences from the single-intersection env are:

  * step() takes one action per intersection ({tls_id: 0 or 1}) and
    advances the simulation by exactly ONE second for everybody. The
    single env blocks for 3 s during a yellow; here a yellow is a
    per-intersection countdown, so one junction going yellow never
    freezes the decisions of the others.
  * Each agent's state is its own local state (same layout as the single
    env) plus a small "neighbour" block: the normalised queue and current
    phase of every adjacent signalised junction. That gives the agents
    the information they need to coordinate (e.g. hold green when a
    platoon is queued at the upstream neighbour).
  * Each agent gets its own local reward (same formula as the single
    env), optionally mixed with its neighbours' rewards via
    neighbour_weight, so agents are nudged not to dump queues onto the
    next junction.

Works on the synthetic grid from make_multi_network.py and on a real
OpenStreetMap import with several signals (osmWebWizard output).
"""
import os
import re
import sys

import numpy as np

sys.path.append(os.environ.get('SUMO_HOME', '/usr/share/sumo') + '/tools')
import traci  # noqa: E402

from perception.perception import LaneDetector  # noqa: E402
from traffic_env_generic import discover_intersection  # noqa: E402

MAX_NEIGHBOURS = 4      # neighbour slots in the state (padded with zeros)
HOLD_FOREVER = 1e6      # stops SUMO's own program from auto-advancing phases


def safe_name(tls_id):
    """TLS ids from OSM can contain '#', ':' etc. -- make them filename-safe."""
    return re.sub(r'[^A-Za-z0-9_.-]', '_', tls_id)


class IntersectionController:
    """Phase state machine + perception for one traffic light."""

    def __init__(self, tls_id, min_green, yellow_time):
        self.tls_id = tls_id
        _, self.lanes, self.green_phases = discover_intersection(tls_id)
        self.logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        self.min_green = min_green
        self.yellow_time = yellow_time

        # junction position: the node the controlled lanes flow INTO
        # (works even when the TLS id is not a junction id, as in OSM)
        first_edge = traci.lane.getEdgeID(self.lanes[0])
        self.junction_id = traci.edge.getToJunction(first_edge)
        jx, jy = traci.junction.getPosition(self.junction_id)
        self.origin = (jx, jy)
        self.detector = LaneDetector(self.lanes, origin=(jx, jy))

        # lanes served by each green phase (used by the actuated baseline)
        controlled = traci.trafficlight.getControlledLanes(tls_id)
        self.phase_lanes = {}
        for gp in self.green_phases:
            st = self.logic.phases[gp].state
            served = {controlled[i] for i, ch in enumerate(st) if ch in 'Gg'}
            self.phase_lanes[gp] = [l for l in self.lanes if l in served]

        self.neighbours = []  # filled in by the env once all TLS are known

    # ---------------- phase machinery ----------------
    def _set_phase(self, idx):
        traci.trafficlight.setPhase(self.tls_id, idx)
        traci.trafficlight.setPhaseDuration(self.tls_id, HOLD_FOREVER)

    def _yellow_after(self, green_idx):
        nxt = (green_idx + 1) % len(self.logic.phases)
        return nxt if 'y' in self.logic.phases[nxt].state else None

    def reset(self):
        self.cur_phase_idx = 0
        self.phase_timer = 0
        self.yellow_left = 0
        self.pending_idx = None
        self.prev_wait = 0.0
        self.last_cv_output = None
        self.last_reward_components = {}
        self._set_phase(self.green_phases[0])
        self.detector = LaneDetector(self.lanes, origin=self.origin)

    @property
    def in_yellow(self):
        return self.yellow_left > 0

    def apply_action(self, action):
        """Called BEFORE the simulation step. Returns an info dict."""
        info = {'requested_action': int(action),
                'action_name': 'HOLD' if action == 0 else 'SWITCH',
                'phase_before': int(self.green_phases[self.cur_phase_idx]),
                'switch_requested': False, 'switch_allowed': False,
                'blocked_by_min_green': False, 'ignored_in_yellow': False,
                'yellow_used': False}
        if self.in_yellow:
            info['ignored_in_yellow'] = action == 1
            return info
        if action != 1:
            return info
        info['switch_requested'] = True
        if self.phase_timer < self.min_green:
            info['blocked_by_min_green'] = True
            return info
        info['switch_allowed'] = True
        target = (self.cur_phase_idx + 1) % len(self.green_phases)
        yellow = self._yellow_after(self.green_phases[self.cur_phase_idx])
        if yellow is not None and self.yellow_time > 0:
            info['yellow_used'] = True
            self._set_phase(yellow)
            self.yellow_left = self.yellow_time
            self.pending_idx = target
        else:
            self._set_phase(self.green_phases[target])
            self.cur_phase_idx, self.phase_timer = target, 0
        return info

    def after_sim_step(self):
        """Called AFTER the simulation step: advance timers / finish yellow."""
        if self.in_yellow:
            self.yellow_left -= 1
            if self.yellow_left == 0:
                self._set_phase(self.green_phases[self.pending_idx])
                self.cur_phase_idx, self.phase_timer = self.pending_idx, 0
                self.pending_idx = None
        else:
            self.phase_timer += 1

    # ---------------- observation / reward ----------------
    def observe(self, use_perception=True):
        lane_dists, positions = {}, []
        for lane in self.lanes:
            lane_len = traci.lane.getLength(lane)
            dists = []
            for vid in traci.lane.getLastStepVehicleIDs(lane):
                dists.append(max(0.0, lane_len - traci.vehicle.getLanePosition(vid)))
                if use_perception:
                    x, y = traci.vehicle.getPosition(vid)
                    positions.append((x, y, 0))
            lane_dists[lane] = dists
        if use_perception:
            frame = self.detector.render_frame(positions)
            cv_out = self.detector.detect(frame, lane_dists)
        else:  # same counting rule, without rendering (fast path for baselines)
            from perception.perception import STOP_LINE_M
            cv_out = {l: {'count': len(d), 'queue': sum(1 for x in d if x <= STOP_LINE_M)}
                      for l, d in lane_dists.items()}
            cv_out['_detected_blobs_this_frame'] = sum(len(d) for d in lane_dists.values())
        self.last_cv_output = cv_out
        return cv_out

    def local_state(self):
        cv = self.last_cv_output
        n = len(self.lanes)
        counts = [cv[l]['count'] / 20.0 for l in self.lanes]
        queues = [cv[l]['queue'] / 20.0 for l in self.lanes]
        onehot = [1.0 if self.cur_phase_idx == i else 0.0 for i in range(len(self.green_phases))]
        timer = [min(self.phase_timer / 60.0, 1.0)]
        yellow = [1.0 if self.in_yellow else 0.0]
        assert len(counts) == n
        return counts + queues + onehot + timer + yellow

    def total_queue(self):
        return sum(traci.lane.getLastStepHaltingNumber(l) for l in self.lanes)

    def local_reward(self):
        wait = sum(traci.lane.getWaitingTime(l) for l in self.lanes)
        queue = self.total_queue()
        thru = sum(traci.lane.getLastStepVehicleNumber(l) for l in self.lanes)
        delta_wait = wait - self.prev_wait
        wait_term, queue_term, thru_term = -delta_wait, -0.25 * queue, 0.10 * thru
        r = wait_term + queue_term + thru_term
        self.prev_wait = wait
        self.last_reward_components = {
            'wait': float(wait), 'delta_wait': float(delta_wait),
            'wait_term': float(wait_term), 'queue': int(queue),
            'queue_term': float(queue_term), 'throughput': int(thru),
            'throughput_term': float(thru_term), 'local': float(r)}
        return float(r)


def find_neighbours(ctrls):
    """Two signals are neighbours if a lane controlled by one starts at a
    junction belonging to the other (i.e. traffic flows directly between them)."""
    junction_owner = {}
    for c in ctrls.values():
        for lane in c.lanes:
            junction_owner[traci.edge.getToJunction(traci.lane.getEdgeID(lane))] = c.tls_id
    for c in ctrls.values():
        nbrs = []
        for lane in c.lanes:
            upstream = traci.edge.getFromJunction(traci.lane.getEdgeID(lane))
            owner = junction_owner.get(upstream)
            if owner and owner != c.tls_id and owner not in nbrs:
                nbrs.append(owner)
        c.neighbours = sorted(nbrs)[:MAX_NEIGHBOURS]


class MultiTrafficEnv:
    """All signals in the network, one agent each, one shared SUMO run."""

    def __init__(self, sumo_cfg, gui=False, min_green=10, yellow_time=3,
                 max_steps=1200, seed=None, tls_ids=None, neighbour_weight=0.0,
                 use_perception=True):
        self.sumo_bin = 'sumo-gui' if gui else 'sumo'
        self.sumo_cfg = sumo_cfg
        self.min_green = min_green
        self.yellow_time = yellow_time
        self.max_steps = max_steps
        self.seed = seed
        self.requested_tls = tls_ids
        self.neighbour_weight = neighbour_weight
        self.use_perception = use_perception
        self.ctrls = None
        self.tls_ids = None
        self.state_dims = None
        self.action_dim = 2  # per intersection: 0 = hold, 1 = advance to next green

    # ------------------------------------------------------------------
    def _start_sumo(self):
        if traci.isLoaded():
            traci.close()
        args = [self.sumo_bin, '-c', self.sumo_cfg, '--no-step-log', 'true',
                '--waiting-time-memory', '10000', '--time-to-teleport', '300']
        if self.seed is not None:
            args += ['--seed', str(self.seed)]
        traci.start(args)
        traci.simulationStep()

    def _discover(self):
        all_tls = list(traci.trafficlight.getIDList())
        ids = self.requested_tls or all_tls
        ctrls = {}
        for tid in ids:
            try:
                ctrls[tid] = IntersectionController(tid, self.min_green, self.yellow_time)
            except RuntimeError as e:  # e.g. a pedestrian-only light with 1 green phase
                print(f'[env] skipping TLS "{tid}": {e}')
        if not ctrls:
            raise RuntimeError('No controllable traffic lights found.')
        find_neighbours(ctrls)
        self.ctrls = ctrls
        self.tls_ids = list(ctrls)
        print(f'[env] controlling {len(ctrls)} intersections:')
        for c in ctrls.values():
            print(f'      {c.tls_id:<12} lanes={len(c.lanes):<3} '
                  f'green phases={c.green_phases}  neighbours={c.neighbours}')

    def reset(self):
        self._start_sumo()
        if self.ctrls is None:
            self._discover()
        else:  # re-bind to the fresh simulation, keep discovered structure
            for c in self.ctrls.values():
                c.logic = traci.trafficlight.getAllProgramLogics(c.tls_id)[0]
        for c in self.ctrls.values():
            c.reset()
        self.step_count = 0
        self.wait_log, self.queue_log, self.arrived_total = [], [], 0
        self.per_tls_wait = {t: [] for t in self.tls_ids}
        self.per_tls_queue = {t: [] for t in self.tls_ids}
        states = self._observe_all()
        if self.state_dims is None:
            self.state_dims = {t: len(s) for t, s in states.items()}
        return states

    # ------------------------------------------------------------------
    def _observe_all(self):
        for c in self.ctrls.values():
            c.observe(self.use_perception)
        states = {}
        for tid, c in self.ctrls.items():
            nb = []
            for i in range(MAX_NEIGHBOURS):
                if i < len(c.neighbours):
                    n = self.ctrls[c.neighbours[i]]
                    q = sum(n.last_cv_output[l]['queue'] for l in n.lanes)
                    phase = n.cur_phase_idx / max(1, len(n.green_phases) - 1)
                    nb += [q / (5.0 * len(n.lanes)), phase]
                else:
                    nb += [0.0, 0.0]
            states[tid] = np.array(c.local_state() + nb, dtype=np.float32)
        return states

    def _log_step(self):
        tw = tq = 0.0
        for tid, c in self.ctrls.items():
            w = sum(traci.lane.getWaitingTime(l) for l in c.lanes)
            q = c.total_queue()
            self.per_tls_wait[tid].append(w)
            self.per_tls_queue[tid].append(q)
            tw += w
            tq += q
        self.wait_log.append(tw)
        self.queue_log.append(tq)
        self.arrived_total += traci.simulation.getArrivedNumber()

    def step(self, actions):
        """actions: {tls_id: 0/1}. Returns (states, rewards, done, infos), all dicts
        keyed by tls_id except done (one shared episode end)."""
        infos = {tid: c.apply_action(int(actions.get(tid, 0)))
                 for tid, c in self.ctrls.items()}
        traci.simulationStep()
        self.step_count += 1
        for c in self.ctrls.values():
            c.after_sim_step()
        self._log_step()

        states = self._observe_all()
        local = {tid: c.local_reward() for tid, c in self.ctrls.items()}
        rewards = {}
        for tid, c in self.ctrls.items():
            r = local[tid]
            if self.neighbour_weight and c.neighbours:
                r += self.neighbour_weight * float(np.mean([local[n] for n in c.neighbours]))
            rewards[tid] = r
            infos[tid].update({
                'step': self.step_count,
                'phase_after': int(c.green_phases[c.cur_phase_idx]),
                'phase_index': int(c.cur_phase_idx),
                'in_yellow': bool(c.in_yellow),
                'applied_switch': infos[tid]['switch_allowed'],
                'reward_components': dict(c.last_reward_components, total=float(r)),
            })
        done = (self.step_count >= self.max_steps or
                traci.simulation.getMinExpectedNumber() <= 0)
        return states, rewards, done, infos

    def close(self):
        if traci.isLoaded():
            traci.close()
