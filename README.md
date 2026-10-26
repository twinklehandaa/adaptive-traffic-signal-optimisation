# Adaptive Traffic Signal Optimization — Working Project (Review-2)

Real, runnable implementation of the CV + DQN adaptive traffic-signal
controller described in the report, built and trained end to end.

## What's real here
- **SUMO network** (`network/`): a genuine 4-approach signalised intersection
  built with netconvert, with realistic peak/off-peak demand (`routes.rou.xml`,
  ~1600 vehicles per 40-minute run).
- **CV perception** (`perception/perception.py`): OpenCV background-subtraction
  detector running on rendered overhead frames — not a stub.
- **RL environment** (`traffic_env.py`): Gym-style TraCI wrapper matching the
  report's design (state, reward, min-green/yellow enforcement).
- **DQN agent** (`dqn_agent.py`): implemented in pure NumPy (manual forward
  pass, backprop, Adam) instead of PyTorch — PyTorch's CUDA dependency chain
  didn't fit in the build sandbox's disk quota. Same architecture (2 hidden
  layers, replay buffer, target network, epsilon-greedy) as the report's
  reference PyTorch code in Section 11.2.3. Swap in `dqn_agent_torch.py`
  later if you get a machine with more disk space / a GPU — the interface
  (`act`, `remember`, `replay`, `save`, `load`) is identical so `train.py`
  won't need to change.
- **Baselines** (`baselines.py`): fixed-time (SUMO's native static program)
  and a gap-based actuated controller, for the comparison in Fig. 11.6.

## Results from the run already completed
- 120-episode training run: reward converges from ~600 to a stable ~800-860
  (see `logs/reward_history.npy`, `logs/train_console.log`).
- 3-seed comparison vs baselines (`logs/comparison_summary.json`): average
  network waiting time drops from ~292s (fixed-time) / ~298s (actuated) to
  ~19s under the trained DQN policy, with a lower average queue and slightly
  higher throughput.

## How to run it yourself
```bash
export SUMO_HOME=/usr/share/sumo   # adjust to your SUMO install
python3 train.py                   # retrain from scratch (~22 min on 1 CPU core)
python3 compare_baselines.py       # regenerate the baseline comparison + chart
python3 plot_reward.py             # regenerate the reward-convergence plot
```
Needs: SUMO + TraCI/sumolib, numpy, opencv-python(-headless), matplotlib.

## Honest gaps to close before Review-3
- Only 120 training episodes / 3 evaluation seeds so far — the project plan
  correctly lists this as "substantially complete" / "in progress", not done.
- No real (non-simulated) camera footage yet — vision runs on rendered SUMO
  frames, as scoped in the report.
- If you get PyTorch working on your own machine, porting `dqn_agent.py`
  back to the Section 11.2.3 reference implementation is a quick swap.

## Using a real intersection (OpenStreetMap import) instead of the synthetic one
The files above (`traffic_env.py`, `baselines.py`, `train.py`) hardcode lane
names like `N1_0` that only exist in the bundled synthetic network. To use
a real intersection instead:

1. **Generate a real network.** Run SUMO's built-in wizard:
   `python "%SUMO_HOME%\tools\osmWebWizard.py"` (Windows) or
   `python3 $SUMO_HOME/tools/osmWebWizard.py` (Linux/Mac). It opens a map in
   your browser -- zoom to a real signalised intersection, draw a selection
   box around it, click **Generate Scenario**. It downloads the real OSM
   road data and writes a timestamped folder (e.g. `2024-xx-xx-.../`)
   containing `osm.net.xml`, `osm.rou.xml`, `osm.sumocfg`, etc.
2. **Use the *generic* pipeline** -- `traffic_env_generic.py`,
   `baselines_generic.py`, `train_generic.py` -- instead of the hardcoded
   ones. These auto-detect the traffic light, its lanes, and its green
   phases from whatever network you point them at, so they work on the
   real OSM network without editing any lane names:
   ```
   python3 train_generic.py path/to/osm.sumocfg
   python3 -c "from baselines_generic import run_fixed_time, run_actuated; \
       print(run_fixed_time('path/to/osm.sumocfg')); \
       print(run_actuated('path/to/osm.sumocfg'))"
   ```
3. **Pick a fairly simple real intersection first** (a standard 4-way
   signal with 2 main green phases works best) -- the auto-discovery
   handles more complex multi-phase intersections too, but a simple one
   is easier to sanity-check and debug the night before a review.
4. I regression-tested `*_generic.py` against the bundled synthetic
   network and it reproduces the same behaviour as the hardcoded version
   (auto-detects TLS "J1", its 8 lanes, and 2 green phases correctly), but
   I was not able to test it against a *real* downloaded OSM network
   myself (no internet access to openstreetmap.org from the build
   sandbox) -- so budget time to debug rough edges if you go this route
   the night before a deadline.


## Final before-vs-after result

The console demonstration ends with a quantitative comparison of the traffic
conditions before and after DQN control:

- Average waiting time
- Average queue length
- Absolute change
- Percentage improvement

The comparison uses the first 30 observed simulation steps as the **before**
window and the final 30 controlled steps as the **after** window. The exact
values are generated from the running SUMO simulation, not hard-coded.

Example presentation output:

```text
========================================================================================
 BEFORE vs AFTER - DQN IMPACT
========================================================================================
METRIC                           BEFORE          AFTER         CHANGE     IMPROVEMENT
--------------------------------------------------------------------------------------
Average waiting time             18.42 s         11.37 s       -7.05 s          38.3%
Average queue length               7.20            4.10         -3.10            43.1%
========================================================================================
```


### Important run order

The console demo runs the fixed-time baseline first and waits for that SUMO/TraCI connection to close. Only then is the DQN environment started. This avoids TraCI's `Connection 'default' is already active` error.


## Multiple intersections (multi-agent extension)

The project now also controls **several signalised intersections at once**,
with one DQN agent per intersection, all running inside one shared SUMO
simulation so each agent's decisions affect its neighbours' traffic.

### New files
| File | What it does |
|---|---|
| `make_multi_network.py` | Builds an N×M grid of signals with `netgenerate` (default 2×2 = 4 signals) plus time-varying demand with a heavy peak corridor. Writes `network_multi/grid.sumocfg`. |
| `traffic_env_multi.py` | `MultiTrafficEnv`: auto-discovers **every** traffic light (same `discover_intersection()` as the generic env), gives each its own CV `LaneDetector`, phase state machine, local reward, and a neighbour block in its state. |
| `train_multi.py` | Independent DQN (one `DQNAgent` per signal, unchanged class). Saves `checkpoints/multi/dqn_<TLS>.npz` + `meta.json`, and the best late episode to `checkpoints/multi_best/`. |
| `baselines_multi.py` | Fixed-time vs actuated vs DQN on the whole network, network-wide and per intersection. Writes `logs/multi_comparison_summary.json` and `screens/multi_comparison.png`. |
| `console_multi_demo.py` | Console CV → DQN → actuation demo: one row per intersection per step, then a before/after table (network + per intersection). `--focus A0` adds the full per-lane / reward breakdown for one junction. |

### Design
- **State per agent** = the single-intersection state (lane counts, lane queues,
  phase one-hot, phase timer) + a yellow flag + for up to 4 neighbouring
  signals: their normalised queue and current phase. Neighbours are found
  automatically: two signals are neighbours if a lane of one starts at a
  junction of the other.
- **Reward per agent** = the same formula as before on its own lanes,
  plus `--neighbour-weight` × the mean of its neighbours' rewards (default 0.1),
  so an agent is discouraged from flushing its queue into the next junction.
- **Non-blocking yellow**: the single env advances SUMO 3 s during a yellow;
  with many signals that would freeze everyone, so here each signal has its
  own yellow countdown and the simulation always advances 1 s per step.
- **Phase holding**: after every `setPhase` the env calls
  `setPhaseDuration(tls, 1e6)` so SUMO's own static program can never
  auto-advance a phase behind the agent's back.

### Run order
```bash
python3 make_multi_network.py                 # or --nx 3 --ny 1 for a 3-signal arterial
python3 train_multi.py --fast                 # 120 episodes, ~25 min on 1 core
python3 baselines_multi.py                    # fixed vs actuated vs DQN, 3 seeds
python3 console_multi_demo.py --focus A0      # presentation demo
```
`--fast` skips drawing CV frames during training only; the counts/queues fed to
the agent are computed by the same rule, so the state is identical. The demo
always runs the full CV rendering.

### Results (2x2 grid, 1200 s, 3 seeds, logs/multi_comparison_summary.json)
| Controller | Network avg wait | Avg queue | Arrived |
|---|---|---|---|
| Fixed-time, network default 42 s greens | 439.8 s | 27.6 | 1090 |
| Fixed-time, 10 s greens (tuned on held-out seed) | 51.6 s | 11.1 | 1136 |
| Actuated (gap-out) | 537.2 s | 31.0 | 1064 |
| DQN, one agent per signal | 55.7 s | 10.8 | 1132 |

The DQN agents beat the default fixed-time and actuated controllers by a wide
margin, but a short, well-tuned fixed cycle is about as good (slightly better
on wait, slightly worse on queue). The DQN is behind mainly at A0, where both
peak corridors cross. Report against the tuned baseline, not only the default.

### Real multi-signal OSM area
Select a box with several signals in `osmWebWizard.py`, then:
```bash
python3 train_multi.py path/to/osm.sumocfg --fast
python3 baselines_multi.py path/to/osm.sumocfg
python3 console_multi_demo.py path/to/osm.sumocfg
```
Signals with fewer than two green phases (e.g. pedestrian-only lights) are
skipped automatically and listed at start-up.
