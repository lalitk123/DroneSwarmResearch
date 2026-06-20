# Per-drone state published by drone_node and consumed by metrics_node /
# base_station_node. Mirrors the bookkeeping fields of swarm_sim/drone.py.

std_msgs/Header header

int32   drone_id
bool    alive                 # mirrors Drone.alive (node-failure model)
geometry_msgs/Point pos       # current (y, x, z) position
geometry_msgs/Point target    # current assigned goal cell (y, x, z)
float64 path_length           # accumulated path length (energy proxy)

# Local belief summary (full maps are too large for a status topic; the metrics
# node reconstructs global coverage from the union of per-drone covered cells
# streamed separately, or from these counts for lightweight monitoring).
int32   covered_cell_count    # number of cells this drone believes covered
int32   known_victim_count    # number of victims in this drone's local belief
