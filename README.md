
# 🚦 Adaptive Traffic Signal Optimization Using Deep Reinforcement Learning

An intelligent traffic management system that uses **Deep Reinforcement Learning (DQN)**, **SUMO traffic simulation**, and a **computer vision-based perception pipeline** to explore adaptive traffic signal control and traffic flow optimization.

## 📌 Overview

Traditional traffic signals often use fixed timings that cannot always adapt to changing traffic conditions. This project explores an adaptive approach in which a Deep Q-Network (DQN) agent observes traffic conditions and decides whether to maintain or switch the current traffic signal phase.

The system is developed using **Python and SUMO**, with support for single-intersection and multi-intersection experiments. It evaluates traffic performance using waiting-time metrics, queue lengths, and vehicle throughput.

## 🎯 Objectives

- Develop an adaptive traffic signal controller using Deep Reinforcement Learning.
- Reduce vehicle waiting time and traffic queue lengths.
- Improve traffic flow under varying traffic conditions.
- Integrate a computer vision-based perception pipeline.
- Extend traffic signal control to multiple intersections.
- Compare the learned controller against conventional traffic signal strategies.

## ✨ Key Features

- 🧠 **DQN-Based Control:** Learns signal-control policies through interaction with the simulation.
- 🚗 **SUMO Integration:** Simulates vehicles, road networks, and traffic signals.
- 🔄 **Adaptive Signal Switching:** Chooses between holding and switching the current phase.
- 👁️ **Computer Vision Pipeline:** Uses OpenCV-based synthetic frame processing and contour detection.
- 🛣️ **Multi-Intersection Control:** Supports multiple DQN agents with neighbouring-intersection information.
- 📊 **Performance Evaluation:** Tracks waiting-time metrics, queue lengths, throughput, and rewards.
- 📈 **Baseline Comparisons:** Includes fixed-time and actuated control strategies.
- 🌍 **OSM-Based Simulation:** Supports experiments using an OpenStreetMap-derived road network.
- 📉 **Visualization Dashboard:** Provides visual access to simulation and evaluation outputs.

## 🏗️ System Architecture

```text
Traffic Simulation (SUMO)
          |
          v
Traffic State Observation
          |
          v
Computer Vision / Perception Pipeline
          |
          v
DQN Agent
          |
          v
Action Selection
(Hold / Switch)
          |
          v
Traffic Signal Controller
          |
          v
Reward Calculation
          |
          v
Experience Replay & DQN Training
          |
          v
Performance Evaluation
```

The process repeats across simulation steps and training episodes, allowing the agent to learn a traffic signal control policy.

## 🧠 Methodology

### 1. Deep Q-Network (DQN)

The project uses a Deep Q-Network to estimate the expected cumulative reward associated with each possible action in a given traffic state.

The implementation includes:

- A neural network with two hidden layers.
- ReLU activation functions.
- Experience replay.
- A target network for more stable learning.
- Epsilon-greedy exploration.

### 2. State Space

The single-intersection state includes traffic-related features such as:

- Vehicle counts on approach lanes.
- Queue estimates on approach lanes.
- Current traffic signal phase.
- Normalized phase duration.

The multi-intersection implementation additionally incorporates neighbouring-signal information.

### 3. Action Space

The controller supports two actions:

| Action | Description |
|---|---|
| **Hold** | Maintain the current signal phase. |
| **Switch** | Transition to the next permitted signal phase. |

Signal switching incorporates configured minimum-green and yellow-light constraints.

### 4. Reward Function

The single-intersection environment uses the following reward formulation:

\[
r_t = -(W_t-W_{t-1}) - 0.25Q_t + 0.10V_t
\]

Where:

- \(W_t\): Aggregate lane waiting-time measure at the current step.
- \(W_{t-1}\): Corresponding measure at the previous step.
- \(Q_t\): Number of halting vehicles.
- \(V_t\): Number of vehicles on the relevant lanes.

The reward penalizes increases in waiting time and larger queues while adding a positive contribution based on vehicle presence.

### 5. Multi-Intersection Extension

The project includes a multi-intersection implementation in which separate DQN agents control individual intersections. Neighbouring-intersection information can be incorporated into the state representation, with optional neighbour-reward mixing.

This setup explores decentralized traffic control. However, independent agents do not guarantee globally optimal coordination, and performance depends on training and reward design.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| SUMO | Microscopic traffic simulation |
| TraCI | Python-SUMO communication |
| NumPy | Numerical computation |
| PyTorch / neural-network implementation | DQN computation, depending on the selected module |
| OpenCV | Image processing and perception |
| Matplotlib | Training plots and visualizations |
| HTML, CSS, JavaScript | Dashboard interface |
| OpenStreetMap | Geographic road-network data |

## 📂 Project Structure

```text
traffic_project_multi/
├── checkpoints/
│   ├── multi/
│   └── multi_best/
├── logs/
├── network/
├── network_multi/
├── osm_real/
├── perception/
├── baselines.py
├── baselines_generic.py
├── baselines_multi.py
├── build_dashboard.py
├── compare_baselines.py
├── compare_baselines_generic.py
├── console_multi_demo.py
├── console_pipeline_demo.py
├── dqn_agent.py
├── make_multi_network.py
└── dashboard.html
```

Additional environment and training scripts are included in the project.

### Main Components

| Component | Description |
|---|---|
| `dqn_agent.py` | DQN agent and learning logic |
| `baselines.py` | Baseline traffic control strategies |
| `baselines_generic.py` | Generic baseline controller logic |
| `baselines_multi.py` | Multi-intersection baseline strategies |
| `compare_baselines.py` | Single-intersection evaluation |
| `compare_baselines_generic.py` | Generic-network comparisons |
| `build_dashboard.py` | Dashboard generation |
| `console_multi_demo.py` | Multi-intersection console demonstration |
| `console_pipeline_demo.py` | Traffic-control pipeline demonstration |
| `network/` | Single-intersection SUMO network |
| `network_multi/` | Multi-intersection SUMO network |
| `osm_real/` | OSM-derived simulation files |
| `checkpoints/` | Saved model parameters |
| `logs/` | Experiment summaries and training histories |

## ⚙️ Installation

### Prerequisites

- Python 3.10 or a compatible version.
- [SUMO](https://sumo.dlr.de/docs/).
- TraCI Python interface.
- Required Python libraries.

Ensure that SUMO is installed and its executable tools are available through your system PATH.

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd traffic_project_multi
```

Replace `<YOUR_REPOSITORY_URL>` with your repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required packages used by the project:

```bash
pip install numpy matplotlib opencv-python
```

Install the appropriate PyTorch and SUMO/TraCI dependencies required by your selected scripts.

> **Note:** Use dependency versions verified against your local project setup. A `requirements.txt` file should be maintained for reproducible installation.

## ▶️ Running the Project

The repository contains scripts for training, demonstrations, and baseline comparisons.

The general workflow is:

1. Configure the SUMO network and traffic demand.
2. Start the appropriate simulation environment.
3. Run the single-intersection or multi-intersection training workflow.
4. Save and load the trained agent checkpoint.
5. Run the relevant demonstration or evaluation script.
6. Inspect the experiment summaries, plots, and dashboard.

The repository also includes an OSM-derived simulation configuration in `osm_real/`.

> **Note:** Verify the available training script names and their arguments in your current repository version before execution. SUMO must be correctly installed and configured.

## 📊 Evaluation Metrics

The project evaluates traffic performance using the following metrics:

- **Waiting Time:** Measures vehicle delay according to the environment's configured metric.
- **Queue Length:** Measures the extent of traffic queues.
- **Throughput:** Uses arrived-vehicle counts or another explicitly defined traffic-flow measure.
- **Cumulative Reward:** Helps track learning progress during training.

Lower waiting-time and queue metrics generally indicate better performance, while higher throughput can indicate improved traffic flow when evaluation conditions are comparable.

## 📈 Experimental Results

The repository contains comparison summaries for:

- Single-intersection traffic control.
- An OpenStreetMap-derived road network.
- Multi-intersection traffic control.

The recorded results vary across scenarios and metrics. The DQN controller shows promising results in some experiments but does **not** outperform every baseline in every scenario.

For example, in the recorded OSM comparison, DQN achieves a lower average queue than the native actuated controller, while the actuated controller records a lower waiting-time metric and higher throughput.

The multi-intersection results also indicate scope for further tuning and evaluation.

For reliable conclusions, comparisons should use equivalent traffic demand, simulation duration, random seeds, and metric definitions. Repeated trials are needed to measure variability.

## ⚠️ Limitations

- The system has primarily been evaluated in simulation.
- The current perception pipeline uses simulator-derived vehicle information for final lane-level counts and queue classifications.
- Independent multi-agent control does not guarantee globally optimal coordination.
- Real-world deployment requires additional safety engineering and validation.


## 🚀 Future Scope

- Integrate a trained object-detection model for image-based vehicle counting.
- Validate the perception pipeline using real traffic camera footage.
- Improve reward design and hyperparameter tuning.
- Compare DQN with algorithms such as PPO.
- Explore advanced multi-agent reinforcement learning approaches.
- Evaluate performance under varying traffic demand and network configurations.
- Add pedestrian-aware control and emergency vehicle prioritization.
- Conduct repeated experiments and statistical significance analysis.
- Investigate real-world deployment requirements.

## 🌐 Applications

Potential research applications include:

- Intelligent Transportation Systems (ITS).
- Adaptive traffic signal control.
- Smart city infrastructure.
- Reinforcement learning for transportation.
- Multi-intersection traffic management.
- Simulation-based traffic optimization.

## 📝 Conclusion

This project explores Deep Reinforcement Learning for adaptive traffic signal optimization. By combining SUMO-based traffic simulation, DQN-based decision-making, a modular perception pipeline, and multi-intersection experiments, it provides a framework for studying data-driven traffic signal control.

The results demonstrate both the potential and the limitations of reinforcement learning for traffic management. Improvements in perception, coordination, evaluation, and real-world validation are necessary before practical deployment.

