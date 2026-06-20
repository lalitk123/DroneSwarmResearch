# An inter-drone (or drone<->base) message routed THROUGH the comm_bridge_node,
# which applies the active CommModel (packet loss / range / dead zone / latency)
# before relaying. Mirrors the payload passed to `send(sender, receiver, payload)`
# in swarm_sim/strategies.py.

std_msgs/Header header        # stamp = send time (used for latency modeling)

int32   sender_id             # originating drone id (-1 = base station)
int32   receiver_id           # intended recipient id (-1 = base, -2 = broadcast)

# Sender pose at send time, so the bridge can evaluate range/dead-zone models
# without a separate pose lookup. (y, x, z) in world/grid coordinates.
geometry_msgs/Point sender_pos
geometry_msgs/Point receiver_pos

# Logical message kind, mirrors the string payloads used by the strategies:
#   "map"    - coverage/victim belief gossip (decentralized) or uplink (centralized)
#   "assign" - base->drone target assignment (centralized downlink)
string  kind

# Opaque JSON-encoded payload (coverage deltas, known victims, target cell, ...).
# JSON keeps the message schema stable while strategy payloads evolve.
string  payload_json
