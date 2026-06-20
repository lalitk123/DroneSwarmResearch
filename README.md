# swarm_sim — Resilient Multi-Drone SAR Simulator (v0)

Lightweight Python simulator for comparing drone-swarm coordination strategies under
communication failures. See [CLAUDE.md](CLAUDE.md) for the research framing, literature
survey, and references.

**Project deliverables:**
- [`swarm_sim/`](swarm_sim) — the simulator + experiment modules (`full_matrix`,
  `relay_experiment`, `relay_k_sweep`, `scaling_study`, `relay_multibranch_experiment`).
- [`paper/paper.md`](paper/paper.md) — the full research write-up (methods, results, discussion).
- [`ros2_backend/`](ros2_backend) — ROS2 + Gazebo backend mirroring the same strategies and
  comm-failure models, for hardware-transfer validation.
- `results/` — all CSVs and degradation-curve figures.

## Install
```bash
pip install -r requirements.txt   # numpy, matplotlib
```

## Run the demo
```bash
python -m swarm_sim.experiments
```
Outputs to `results/`: `centralized_demo.csv`, `summary.json`, `degradation_curve.png`.

## Architecture (all pluggable)
- `config.py` — `SimConfig` dataclass (world size, drones, victims, ranges, seed).
- `world.py` — `World`: grid + obstacle clusters + victims, coverage & detection tracking.
- `drone.py` — `Drone`: kinematic motion, local sensing, per-drone belief map.
- `comm.py` — `CommModel` interface + `PerfectComm`, `PacketLoss`, `RangeComm`
  (disk / path-loss), `DeadZone`, `Latency`, `Composite`. Returns delivery latency or
  `None` (dropped).
- `strategies.py` — `CoordinationStrategy` interface + `CentralizedStrategy` (baseline).
  Decentralized-consensus and bio-inspired-stigmergy strategies plug in here next.
- `simulator.py` — `Simulator`: sense → coordinate → move loop, routes all messages
  through the comm model, handles node failures and delayed delivery, collects metrics.
- `metrics.py` — `EpisodeMetrics`, largest-connected-component connectivity,
  `resilience_auc` (area under performance-vs-severity curve; 1.0 = perfectly resilient).
- `experiments.py` — Monte-Carlo runner: scenario × severity × seeds → degradation curves.

## Scenarios
`S0_perfect`, `S1_packet_loss` (Bernoulli p sweep), `S2_range` (shrinking comm radius /
path-loss). Dead-zone, latency, node-failure, and combined scenarios are supported by the
comm layer and `Simulator(node_failure_rate=...)`; add them to `make_comm` to sweep.

## Note on the baseline
Under *independent per-step* packet loss the centralized baseline is robust (reassignment
is retried every step). The intended failure mode appears under *persistent* link loss
(`S2_range`, dead zones): coverage collapses ~0.97 → 0.13 as range shrinks. That curve is
the baseline the resilient strategies must beat.
