from __future__ import annotations

from typing import TYPE_CHECKING
import math

if TYPE_CHECKING:
    from world.game_state import GameState


def evaluation_function(state: GameState) -> float:
    """
    Función de evaluación para estados no terminales del juego dron vs. cazadores.
 
    Combina cuatro señales:
      1. Progreso de entregas  
      2. Amenaza de cazadores  
      3. Bonificación de seguridad
      4. Señal de puntaje     
    Retorna un valor en [-1000, +1000].
    """

    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0
 
    try:
        from algorithms.utils import bfs_distance
    except ImportError:
        bfs_distance = None
 
    drone_pos = state.get_drone_position()
    hunter_positions = state.get_hunter_positions()
    pending = state.get_pending_deliveries()
    layout = state.get_layout()
    score = state.get_score()
 
    def manhattan(a, b) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
 
    def path_distance(start, goal, hunter_restricted=False) -> float:
        """Distancia BFS con respaldo a Manhattan en caso de error."""
        if bfs_distance is not None:
            try:
                d = bfs_distance(layout, start, goal, hunter_restricted)
                return d if d is not None and not math.isinf(d) else manhattan(start, goal)
            except Exception:
                pass
        return manhattan(start, goal)
 
    # ------------------------------------------------------------------ #
    # 1. Señal de entregas                                                #
    # ------------------------------------------------------------------ #
    if pending:
        nearest_delivery_dist = min(
            path_distance(drone_pos, dp) for dp in pending
        )
        # Más entregas pendientes → mayor penalización por urgencia
        delivery_score = -2.0 * nearest_delivery_dist - 10.0 * len(pending)
    else:
        delivery_score = 200.0  # todas entregadas (la victoria ya debió activarse, pero se recompensa igual)
 
    # ------------------------------------------------------------------ #
    # 2. Señal de amenaza de cazadores                                    #
    # ------------------------------------------------------------------ #
    DANGER_RADIUS = 5  
 
    hunter_score = 0.0
    if hunter_positions:
        hunter_dists = [
            path_distance(hp, drone_pos, hunter_restricted=True)
            for hp in hunter_positions
        ]
        nearest_hunter = min(hunter_dists)
 
        if nearest_hunter <= 1:
            hunter_score = -800.0
        elif nearest_hunter <= DANGER_RADIUS:
            hunter_score = -100.0 / nearest_hunter
        else:
            hunter_score = 10.0
 
    # ------------------------------------------------------------------ #
    # 3. Bonificación de seguridad global                                 #
    # ------------------------------------------------------------------ #
    if hunter_positions:
        min_dist_all = min(
            path_distance(hp, drone_pos, hunter_restricted=True)
            for hp in hunter_positions
        )
        safety_bonus = min(min_dist_all * 3.0, 30.0)
    else:
        safety_bonus = 30.0
 
    # ------------------------------------------------------------------ #
    # 4. Señal de puntaje                                                 #
    # ------------------------------------------------------------------ #
    score_signal = score * 5.0
 
    total = delivery_score + hunter_score + safety_bonus + score_signal
    return max(-999.0, min(999.0, total))