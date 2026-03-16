from __future__ import annotations

from typing import TYPE_CHECKING
import math

if TYPE_CHECKING:
    from world.game_state import GameState


def evaluation_function(state: GameState) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    A good evaluation function can consider multiple factors, such as:
      (a) BFS distance from drone to nearest delivery point (closer is better).
          Uses actual path distance so walls and terrain are respected.
      (b) BFS distance from each hunter to the drone, traversing only normal
          terrain ('.' / ' ').  Hunters blocked by mountains, fog, or storms
          are treated as unreachable (distance = inf) and pose no threat.
      (c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
      (d) Number of pending deliveries (fewer is better).
      (e) Current score (higher is better).
      (f) Delivery urgency: reward the drone for being close to a delivery it can
          reach strictly before any hunter, so it commits to nearby pickups
          rather than oscillating in place out of excessive hunter fear.
      (g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.

    Returns a value in [-1000, +1000].

    Tips:
    - Use state.get_drone_position() to get the drone's current (x, y) position.
    - Use state.get_hunter_positions() to get the list of hunter (x, y) positions.
    - Use state.get_pending_deliveries() to get the set of pending delivery (x, y) positions.
    - Use state.get_score() to get the current game score.
    - Use state.get_layout() to get the current layout.
    - Use state.is_win() and state.is_lose() to check terminal states.
    - Use bfs_distance(layout, start, goal, hunter_restricted) from algorithms.utils
      for cached BFS distances. hunter_restricted=True for hunter-only terrain.
    - Use dijkstra(layout, start, goal) from algorithms.utils for cached
      terrain-weighted shortest paths, returning (cost, path).
    - Consider edge cases: no pending deliveries, no hunters nearby.
    - A good evaluation function balances delivery progress with hunter avoidance.
    """
    # ------------------------------------------------------------------ #
    # Terminal states: return large fixed rewards / penalties             #
    # ------------------------------------------------------------------ #
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0
 
    # ------------------------------------------------------------------ #
    # Gather state information                                            #
    # ------------------------------------------------------------------ #
    try:
        from algorithms.utils import bfs_distance
    except ImportError:
        bfs_distance = None
 
    drone_pos = state.get_drone_position()
    hunter_positions = state.get_hunter_positions()
    pending = state.get_pending_deliveries()
    layout = state.get_layout()
    score = state.get_score()
 
    # ------------------------------------------------------------------ #
    # Helper: Manhattan distance (fallback when BFS is unavailable)       #
    # ------------------------------------------------------------------ #
    def manhattan(a, b) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
 
    def path_distance(start, goal, hunter_restricted=False) -> float:
        """BFS distance with graceful fallback to Manhattan."""
        if bfs_distance is not None:
            try:
                d = bfs_distance(layout, start, goal, hunter_restricted)
                return d if d is not None and not math.isinf(d) else manhattan(start, goal)
            except Exception:
                pass
        return manhattan(start, goal)
 
    # ------------------------------------------------------------------ #
    # 1. Delivery signal                                                  #
    # ------------------------------------------------------------------ #
    # Reward for being close to the nearest pending delivery.
    # If all deliveries are done the agent should have won (handled above),
    # but we handle the edge case gracefully.
    if pending:
        nearest_delivery_dist = min(
            path_distance(drone_pos, dp) for dp in pending
        )
        # More deliveries remaining → higher urgency penalty
        delivery_score = -2.0 * nearest_delivery_dist - 10.0 * len(pending)
    else:
        delivery_score = 200.0  # all delivered (win should have triggered, but reward anyway)
 
    # ------------------------------------------------------------------ #
    # 2. Hunter threat signal                                             #
    # ------------------------------------------------------------------ #
    # Hunters move only on free terrain, so we use hunter_restricted BFS.
    DANGER_RADIUS = 5   # cells within which a hunter is considered dangerous
 
    hunter_score = 0.0
    if hunter_positions:
        hunter_dists = [
            path_distance(hp, drone_pos, hunter_restricted=True)
            for hp in hunter_positions
        ]
        nearest_hunter = min(hunter_dists)
 
        if nearest_hunter <= 1:
            # Imminent capture
            hunter_score = -800.0
        elif nearest_hunter <= DANGER_RADIUS:
            # Strong penalty that fades with distance
            hunter_score = -100.0 / nearest_hunter
        else:
            # Far away: small safety bonus
            hunter_score = 10.0
 
    # ------------------------------------------------------------------ #
    # 3. Global safety bonus                                              #
    # ------------------------------------------------------------------ #
    # Reward the drone when it maintains a comfortable gap from ALL hunters.
    if hunter_positions:
        min_dist_all = min(
            path_distance(hp, drone_pos, hunter_restricted=True)
            for hp in hunter_positions
        )
        safety_bonus = min(min_dist_all * 3.0, 30.0)
    else:
        safety_bonus = 30.0
 
    # ------------------------------------------------------------------ #
    # 4. Score signal                                                     #
    # ------------------------------------------------------------------ #
    score_signal = score * 5.0
 
    # ------------------------------------------------------------------ #
    # Combined value (clipped to [-1000, +1000])                         #
    # ------------------------------------------------------------------ #
    total = delivery_score + hunter_score + safety_bonus + score_signal
    return max(-999.0, min(999.0, total))
