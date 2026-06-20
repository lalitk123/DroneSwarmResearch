# Resilient Multi-Agent Drone Swarms for Disaster Search-and-Rescue Under Communication Failures

## Project summary
Investigate how autonomous drone swarms continue search-and-rescue (SAR) missions when
inter-drone communication is unreliable or fails entirely. In a simulated disaster
environment (obstacles + victims), multiple coordination strategies are compared under
realistic communication degradation. Optional extension: adaptive relay drones that
reposition to restore links.

## Research findings (web survey, 2023–2026)

### 1. Simulation frameworks
- **ROS 2 + Gazebo (+ PX4/Ardupilot)**: high physical/SITL fidelity, standard in robotics
  research, but heavy — scaling past ~10 drones is slow, and it has *no built-in model of
  network degradation*. You bolt on a network layer (e.g. ns-3 via integration) to get
  packet loss/dead zones. Best when you eventually want hardware transfer.
- **Custom Python / lightweight gym envs**: fast to iterate, trivial to inject arbitrary
  comms failures, scales to 10–100+ agents. Lower physical fidelity. Dominant for
  algorithm-comparison studies where comms modeling (not aerodynamics) is the research
  question.
- **gym-pybullet-drones** (UTIAS): Gymnasium env, real PyBullet physics, single + multi
  agent. Good middle ground but its N×N adjacency representation limits large-swarm
  scaling.
- **Flightmare** (UZH-RPG): decouples Unity rendering from physics; can simulate hundreds
  of agents, physics up to ~200 kHz — fastest for large-swarm RL.
- **ns-3 / FANET simulators**: the *communication* gold standard (OLSR/AODV/DSDV routing,
  path-loss + mobility models) but not a physics/mission simulator.

### 2. Coordination strategies & resilience
- **Centralized** (single planner / CBBA-style allocation): optimal coordination but a
  single point of failure; degrades sharply when the link to the coordinator drops —
  the natural "baseline" to beat.
- **Decentralized consensus / gossip (e.g. CBBA, consensus-based bundle)**: mathematically
  elegant, but stability depends on *persistent connectivity*; intermittent links break
  convergence guarantees. Biomimetic/load-aware CBBA variants improve robustness.
- **Bio-inspired / stigmergy (ACO pheromone, PSO, Physarum)**: coordinate through the
  *environment* (shared/virtual pheromone map) rather than direct messaging, so they
  degrade gracefully and even work **communication-free**. Literature repeatedly finds
  stigmergy/bio-inspired methods the most resilient to comms loss; the tradeoff is
  redundant coverage and sensitivity to map staleness. **Strongest fit for this project's
  thesis.**

### 3. Communication failure modeling
Standard approaches researchers use:
- **Probabilistic packet loss**: per-message Bernoulli drop with probability `p`.
- **Geographic / range models**: unit-disk (binary cutoff radius) or distance-dependent
  path loss `d^-α` (Friis free-space, Nakagami-m fading) — link probability falls with
  distance.
- **Dead zones**: spatial regions where comms = 0 (model damaged infrastructure / terrain).
- **Latency / jitter**: delayed message delivery.
- **Node failure**: drones drop out entirely.
Mobility models: Random Waypoint is the common baseline. FANET = MANET in 3D.

### 4. Metrics & resilience quantification
Track: **area coverage %**, **victim detection rate / count**, **time-to-first-detection**,
**mission completion time**, **redundant coverage / overlap**, **energy or path length**,
**network connectivity** (fraction connected, largest connected component).
**Resilience** is typically quantified as *performance retention vs. a no-failure baseline*
— i.e. plot a chosen metric as a function of failure severity (packet-loss rate, dead-zone
size, % drones lost) and report the area-under-curve / degradation slope. A resilient
strategy has a flat curve.

### 5. Scaling to 10+ drones fast
For algorithm comparison with comms as the focus, a **custom 2D/2.5D Python grid+kinematic
sim is fastest**; reach for gym-pybullet-drones/Flightmare only if physical fidelity or RL
transfer matters. ROS 2 + Gazebo if hardware deployment is a future goal.

## Recommendations for this project

**Simulation framework:** Build a **custom lightweight Python simulator** (NumPy +
Gymnasium-style API; matplotlib/pygame viz). Rationale: the research question is *comms
resilience and coordination*, not aerodynamics. We need to scale to 10–50 drones, inject
arbitrary failure models, and run hundreds of Monte-Carlo trials cheaply. Keep a clean
interface so a Gazebo/PX4 backend could be swapped in later for hardware validation.

**Three coordination strategies to implement:**
1. **Centralized planner (baseline)** — global frontier/sector assignment from a base
   station. Shows the failure mode we want to beat.
2. **Decentralized consensus / gossip** — drones share local maps with in-range neighbors
   and merge; assignment via distributed greedy/CBBA-lite. Tests graceful-but-
   connectivity-dependent behavior.
3. **Bio-inspired stigmergy (digital pheromone / ACO-style)** — drones deposit "explored"
   and "attractant" markers on a shared spatial field, repelled from recently-visited
   cells. Near comms-free; expected most resilient. This is the headline hypothesis.

**Communication failure scenarios to test:**
- S0 Perfect comms (control)
- S1 Probabilistic packet loss, sweep p ∈ {0.1 … 0.9}
- S2 Range-limited (unit-disk radius sweep) + distance path-loss variant
- S3 Static dead zones (sweep area fraction)
- S4 Latency/jitter
- S5 Drone failures (sweep % lost, random + targeted)
- S6 Combined "realistic disaster" (range-limited + dead zones + losses)

**Key metrics:** coverage %, victim detection rate, time-to-first-victim, mission time,
redundant-coverage ratio, connectivity (largest connected component), energy/path length.
Report each as a **degradation curve vs. failure severity** + AUC resilience score.

**Extension:** adaptive relay drones — designate a subset to reposition to maximize
connectivity (max largest-connected-component) and measure resilience recovery.
**Implemented & validated** (`swarm_sim/relay.py`, `relay_experiment.py`). Relays require
multi-hop comm routing (`Simulator(multihop=True)`): a message is delivered iff a path of
deliverable links exists through the relay chain. K relays reposition each step to bridge
the base to the *farthest* searcher (the worst link). Result (40×40, 12 drones, 4 relays,
disk-range comms): relays **rescue the centralized strategy** under range loss —
coverage 0.41→0.91 at radius 18 and 0.26→0.89 at radius 12 (+0.5–0.64 absolute), with
connectivity rising in step; the gain tapers once the gap exceeds what 4 relays can span
(radius ≤ 6). Decentralized, already resilient, gains little. Lessons: (1) anchor relays on
the worst gap, not the swarm centroid — a centroid anchor steers relays into a *central*
dead zone; (2) relays only help if messages actually multi-hop through them.

## Metrics Design

**Coverage and detection are measured at mission completion** — i.e. at the step when
all victims have been found, at which point the episode terminates. This is a deliberate
choice that mirrors real disaster search-and-rescue operations: the mission *is* finding
the victims, not blanketing the map. A swarm that locates everyone quickly has succeeded
even if it never physically swept every free cell, so reporting coverage/detection at that
moment captures the operationally meaningful state. (Consequence: converged runs report
coverage ~0.91 rather than ~1.0, because the swarm stops exploring once the last victim is
detected — the un-swept cells are simply areas it never needed to visit.) If a strategy
fails to find all victims, the episode runs to the `max_steps` cap and coverage/detection
reflect the full mission's stalled end-state.

**The 600-step cap never binds on converging strategies.** Empirically (perfect comms,
10 drones / 15 victims): centralized and decentralized converge by ~75–76 steps,
stigmergy by ~183, with 0% of runs hitting the cap. Under heavy comms degradation
(range severity 0.8) the resilient strategies still converge early (~93 and ~210 steps,
0% capped), while the centralized baseline reaches 100% cap-hit at only ~0.30 coverage —
but that is genuine failure, not truncation: cut-off drones idle without assignments, so
extending `max_steps` would add only idle steps, not progress. The cap binds only on a
swarm that has already stalled, never on one still making progress — so no converging run
is cut short and the degradation curves are valid. (`Simulator.run(snapshot_steps=[...])`
supports per-step coverage/detection snapshots for verifying this.)

## Experimental findings (v0 harness, 3 strategies × failure matrix × 10 seeds)

Full sweep in `swarm_sim/full_matrix.py` (`python -m swarm_sim.full_matrix`); outputs
`results/matrix.csv`, `matrix_summary.json`, `winners.json`, `degradation_<type>.png`.
Headline metric: coverage at mission completion + resilience AUC (area under
coverage-vs-severity, 1.0 = flat/ideal). Default map 50×50, 10 drones, 15 victims;
node-failure scenarios use a harder 100×100 / 30-victim / 12-drone map (see below).

**Winner by failure type:**

| Failure family | Winner | Behavior |
|---|---|---|
| Perfect comms | decentralized (0.95) | All converge (cent/dec ~75 steps, stig ~183); small margins. |
| Packet loss 5–20% | centralized (AUC 1.03) | Transient drops don't bite; per-step reassignment recovers. All > 0.90. |
| Range degradation | **stigmergy** (AUC 1.02) | Centralized **collapses 0.97→0.30**; dec/stig stay flat. |
| Dead zone 10–30% | **stigmergy** (AUC 0.99) | Centralized → 0.62 at 30%; dec/stig unaffected. |
| Node failure (random) | **decentralized** (AUC 0.93) | dec > cent > stig; at 75% loss all fail to finish. |
| Node failure (targeted) | cent ≈ dec (AUC 0.87) | Killing best-connected hubs hurts dec's gossip relays — at 75% dec (0.49) drops *below* cent (0.51). |
| Combined disaster | stigmergy ≈ decentralized (~0.93) | Centralized **collapses to 0.40**. |

**Cross-scenario conclusions:**
- **Persistent, geographic comms loss (range / dead-zone / combined) is where centralized
  control dies** (0.97→0.30 range; 0.40 combined) and where **stigmergy and decentralized
  win decisively** — the project's core thesis, now quantified.
- **Transient packet loss hurts no one**; centralized is even marginally best because its
  global planner is optimal when enough messages still arrive.
- **Decentralized consensus is the best all-rounder** — never the outright loser in any
  scenario.
- **Stigmergy is the comms-loss specialist but pays for it elsewhere:** (1) it is the
  *least* resilient to losing agents (no mechanism to re-task survivors toward gaps), and
  (2) it does **not scale efficiently** — on the 100×100 map even the full fleet reaches
  only ~0.79 coverage with ~90% of runs hitting the 800-step cap, because its diffuse,
  redundant search costs throughput. Resilient to comms loss ≠ efficient.
- **Targeted node failure > random:** removing high-degree hubs disproportionately harms
  the gossip-dependent decentralized strategy (its relays *are* the hubs), the one case
  where centralized matches or beats it under node loss.

(Caveat logged: node failure is only discriminating on the harder map fired mid-mission;
on the small map survivors trivially finish, so the original t=150 result showed no effect.
`Simulator(scheduled_failures={step: n}, failure_mode='random'|'targeted')` controls this.)

## Key references / repos
- gym-pybullet-drones — https://github.com/utiasDSL/gym-pybullet-drones
- QuadSwarm (multi-quadrotor RL sim) — https://arxiv.org/pdf/2306.09537
- Flightmare — https://arxiv.org/abs/2009.00563
- Beyond Robustness: Taxonomy of Resilient Multi-Robot Systems — https://arxiv.org/pdf/2109.12343
- Byzantine Resilience at Swarm Scale — https://arxiv.org/pdf/2301.06977
- Robustness/Scalability of Incomplete Virtual Pheromone Maps (stigmergic exploration) — https://www.mdpi.com/2227-9717/12/10/2122
- Testing the limits of pheromone stigmergy in high-density swarms — https://royalsocietypublishing.org/doi/10.1098/rsos.190225
- Physarum-based decentralized mesh networks — https://www.nature.com/articles/s41598-025-33456-y
- Swarm Intelligence-Based Multi-Robotics review — https://www.mdpi.com/2673-9909/4/4/64
- Centrality-Driven Adaptive Consensus Bundle (CBBA biomimetic) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12838631/
- Multi-UAV Coverage Planning with Limited Endurance in Disaster Environment — https://arxiv.org/pdf/2201.10150
- Heterogeneous Fixed-wing Resilient Area Coverage — https://arxiv.org/pdf/2009.09857
- FANET resilient Q-learning routing — https://arxiv.org/pdf/2306.17360

## Deliverables (current state)
- **Simulator + experiments:** `swarm_sim/` (full_matrix, relay_experiment, relay_k_sweep,
  scaling_study, relay_multibranch_experiment). Simulator supports `multihop`,
  `scheduled_failures` (random/targeted), per-step `snapshots`; `DeadZone` has
  `block_crossing` (line-of-sight).
- **Paper:** `paper/paper.md` (~4.6k words, 12 sections, all numbers traced to `results/*`),
  plus `paper/README.md` and `paper/figures.md`.
- **ROS2/Gazebo backend:** `ros2_backend/` — `swarm_sar` (ament_python nodes: drone /
  comm_bridge / base_station / metrics, strategy adapters, Gazebo `disaster.world`, launch,
  `params.yaml`) + `swarm_sar_interfaces` (CommMessage/DroneState msgs). The comm_bridge
  injects the same CommModel degradation as the Python sim; see `ros2_backend/README.md` for
  the full sim→ROS2 mapping table and the PX4/MAVROS hardware path.

## Next phase goal
**Build the custom Python simulation harness (v0):** a grid/continuous 2.5D world with
configurable obstacles + victims, N drones with kinematic motion and local sensing, a
pluggable `CommModel` interface (perfect / packet-loss / range / dead-zone / latency /
node-failure), a pluggable `CoordinationStrategy` interface, and a metrics+logging layer
that produces degradation curves. Deliver one runnable end-to-end demo with the
**centralized baseline** strategy under S0 and S1, plus the experiment-runner scaffold for
Monte-Carlo sweeps. Strategies 2–3 follow once the harness is validated.
