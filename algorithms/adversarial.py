from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import math

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax search.
        """
 
        def minimax(state: GameState, agent_index: int, depth: int) -> float:
            """
            Minimax recursivo. Retorna el valor minimax del estado dado.
 
            - agent_index=0 → dron (nodo MAX)
            - agent_index>0 → cazador (nodo MIN)
            - La profundidad disminuye 1 después de que TODOS los agentes hayan movido (ply completo).
            """
            # Caso base: estado terminal (victoria/derrota) o profundidad agotada
            if state.is_win() or state.is_lose():
                return self.evaluation_function(state)
 
            num_agents = state.get_num_agents()
            legal_actions = state.get_legal_actions(agent_index)
 
            
            if not legal_actions:
                return self.evaluation_function(state)
 
            # Cuando el último agente mueve, se completa un ply → decrementar profundidad
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth
 
            # Verificar límite de profundidad (solo al iniciar un nuevo ply del dron)
            if next_agent == 0 and depth == 0:
                return self.evaluation_function(state)
 
            if agent_index == 0:
                best = -math.inf
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    val = minimax(successor, next_agent, next_depth)
                    best = max(best, val)
                return best
            else:
                # Nodo MIN: cazador
                best = math.inf
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    val = minimax(successor, next_agent, next_depth)
                    best = min(best, val)
                return best
 
        # Llamada raíz: el dron es el agente 0; se elige la acción con mayor valor minimax
        legal_actions = state.get_legal_actions(self.index)
        if not legal_actions:
            return None
 
        num_agents = state.get_num_agents()
        best_action = None
        best_value = -math.inf
 
        for action in legal_actions:
            successor = state.generate_successor(self.index, action)
            next_agent = 1 % num_agents 
            next_depth = self.depth - 1 if next_agent == 0 else self.depth
            val = minimax(successor, next_agent, next_depth)
            if val > best_value:
                best_value = val
                best_action = action
 
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax search with alpha-beta pruning.
        """
 
        def alphabeta(
            state: GameState,
            agent_index: int,
            depth: int,
            alpha: float,
            beta: float,
        ) -> float:
            """
            Búsqueda alfa-beta recursiva. Retorna el valor minimax con poda.
            """
            # Caso base: estado terminal o profundidad agotada
            if state.is_win() or state.is_lose():
                return self.evaluation_function(state)
 
            num_agents = state.get_num_agents()
            legal_actions = state.get_legal_actions(agent_index)
 
            if not legal_actions:
                return self.evaluation_function(state)
 
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth
 
            if next_agent == 0 and depth == 0:
                return self.evaluation_function(state)
 
            if agent_index == 0:
                # Nodo MAX: dron
                value = -math.inf
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = max(
                        value,
                        alphabeta(successor, next_agent, next_depth, alpha, beta),
                    )
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:
                # Nodo MIN: cazador
                value = math.inf
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = min(
                        value,
                        alphabeta(successor, next_agent, next_depth, alpha, beta),
                    )
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value
 
        # Raíz: elegir la acción con el mejor valor alfa-beta
        legal_actions = state.get_legal_actions(self.index)
        if not legal_actions:
            return None
 
        num_agents = state.get_num_agents()
        best_action = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
 
        for action in legal_actions:
            successor = state.generate_successor(self.index, action)
            next_agent = 1 % num_agents
            next_depth = self.depth - 1 if next_agent == 0 else self.depth
            val = alphabeta(successor, next_agent, next_depth, alpha, beta)
            if val > best_value:
                best_value = val
                best_action = action
            alpha = max(alpha, best_value)
 
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.

    Each hunter acts randomly with probability self.prob and greedily
    (worst-case / MIN) with probability 1 - self.prob.

    * When prob = 0:  behaves like Minimax (hunters always play optimally).
    * When prob = 1:  pure expectimax (hunters always play uniformly at random).
    * When 0 < prob < 1: weighted combination that correctly models the
      actual MixedHunterAgent used at game-play time.

    Chance node formula:
        value = (1 - p) * min(child_values) + p * mean(child_values)
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax with mixed hunter model.

        Tips:
        - Drone nodes are MAX (same as Minimax).
        - Hunter nodes are CHANCE with mixed model: the hunter acts greedily with
          probability (1 - self.prob) and uniformly at random with probability self.prob.
        - Mixed expected value = (1-p) * min(child_values) + p * mean(child_values).
        - When p=0 this reduces to Minimax; when p=1 it is pure uniform expectimax.
        - Do NOT prune in expectimax (unlike alpha-beta).
        - self.prob is set via the constructor argument prob.
        """
        # TODO: Implement your code here
        return None
