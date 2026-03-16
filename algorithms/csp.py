from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    assignment: dict[str, str] = {}
    attempted_assignments = 0

    def backtrack() -> dict[str, str] | None:
        nonlocal attempted_assignments

        if csp.is_complete(assignment):
            return dict(assignment)

        var = csp.get_unassigned_variables(assignment)[0]
        for value in list(csp.domains[var]):
            attempted_assignments += 1
            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)
                result = backtrack()
                if result is not None:
                    return result
                csp.unassign(var, assignment)

        return None

    result = backtrack()
    print(f"Attempted assignments: {attempted_assignments}")
    return result


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    assignment: dict[str, str] = {}
    attempted_assignments = 0

    def copy_dominio() -> dict[str, list[str]]:
        return {var: list(values) for var, values in csp.domains.items()}

    def restaurar_dominios(saved: dict[str, list[str]]) -> None:
        csp.domains = {var: list(values) for var, values in saved.items()}

    def forward_check(var: str) -> bool:
        for neighbor in csp.get_neighbors(var):
            if neighbor in assignment:
                continue

            new_dom = [
                value
                for value in csp.domains[neighbor]
                if csp.is_consistent(neighbor, value, assignment)
            ]
            csp.domains[neighbor] = new_dom

            if not new_dom:
                return False

        return True

    def backtrack() -> dict[str, str] | None:
        nonlocal attempted_assignments

        if csp.is_complete(assignment):
            return dict(assignment)

        var = csp.get_unassigned_variables(assignment)[0]

        for value in list(csp.domains[var]):
            attempted_assignments += 1
            if not csp.is_consistent(var, value, assignment):
                continue

            guardado_dom = copy_dominio()
            csp.assign(var, value, assignment)
            csp.domains[var] = [value]

            if forward_check(var):
                result = backtrack()
                if result is not None:
                    return result

            restaurar_dominios(guardado_dom)
            csp.unassign(var, assignment)

        return None

    result = backtrack()
    print(f"Attempted assignments: {attempted_assignments}")
    return result



def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """

    from collections import deque

    assignment: dict[str, str] = {}
    attempted_assignments = 0

    def copy_dominio() -> dict[str, list[str]]:
        return {var: list(values) for var, values in csp.domains.items()}

    def restaurar_dominios(saved: dict[str, list[str]]) -> None:
        csp.domains = {var: list(values) for var, values in saved.items()}

    def values_compatible(
        xi: str, x_value: str, xj: str, y_value: str, current_assignment: dict[str, str]
    ) -> bool:
        base_assignment = {
            k: v for k, v in current_assignment.items() if k not in (xi, xj)
        }

        if x_value != y_value:
            return csp.is_consistent(xi, x_value, base_assignment) and csp.is_consistent(
                xj, y_value, base_assignment
            )

        def order_key(var: str) -> tuple[int, int]:
            t_early = csp.var_to_delivery[var].get("time_window", (0, 10**9))[0]
            return (t_early, csp._var_to_index[var])

        ordered_vars = sorted((xi, xj), key=order_key)
        first_var, second_var = ordered_vars

        pair_values = {xi: x_value, xj: y_value}
        first_value = pair_values[first_var]
        second_value = pair_values[second_var]

        if not csp.is_consistent(first_var, first_value, base_assignment):
            return False

        assignment_with_first = dict(base_assignment)
        assignment_with_first[first_var] = first_value

        return csp.is_consistent(second_var, second_value, assignment_with_first)

    def revise(xi: str, xj: str, current_assignment: dict[str, str]) -> bool:
        revisado = False
        supported_values: list[str] = []

        for x_value in csp.domains[xi]:
            has_support = any(
                values_compatible(xi, x_value, xj, y_value, current_assignment)
                for y_value in csp.domains[xj]
            )

            if has_support:
                supported_values.append(x_value)
            else:
                revisado = True

        if revisado:
            csp.domains[xi] = supported_values

        return revisado

    def ac3(
        current_assignment: dict[str, str],
        cola_inicial: deque[tuple[str, str]] | None = None,
    ) -> bool:
        if cola_inicial is None:
            queue = deque(
                (xi, xj)
                for xi in csp.variables
                for xj in csp.get_neighbors(xi)
            )
        else:
            queue = deque(cola_inicial)

        while queue:
            xi, xj = queue.popleft()

            if revise(xi, xj, current_assignment):
                if not csp.domains[xi]:
                    return False

                for xk in csp.get_neighbors(xi):
                    if xk != xj:
                        queue.append((xk, xi))

        return True

    dom_iniciales = copy_dominio()
    if not ac3(assignment):
        restaurar_dominios(dom_iniciales)
        print(f"Attempted assignments: {attempted_assignments}")
        return None

    def backtrack() -> dict[str, str] | None:
        nonlocal attempted_assignments

        if csp.is_complete(assignment):
            return dict(assignment)

        var = csp.get_unassigned_variables(assignment)[0]

        for value in list(csp.domains[var]):
            attempted_assignments += 1
            if not csp.is_consistent(var, value, assignment):
                continue

            guardado_dom = copy_dominio()
            csp.assign(var, value, assignment)
            csp.domains[var] = [value]

            arc_queue = deque(
                (neighbor, var)
                for neighbor in csp.get_neighbors(var)
                if neighbor not in assignment
            )

            if ac3(assignment, arc_queue):
                result = backtrack()
                if result is not None:
                    return result

            restaurar_dominios(guardado_dom)
            csp.unassign(var, assignment)

        return None

    result = backtrack()
    restaurar_dominios(dom_iniciales)
    print(f"Attempted assignments: {attempted_assignments}")
    return result


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    assignment: dict[str, str] = {}
    attempted_assignments = 0

    def copy_dominio() -> dict[str, list[str]]:
        return {var: list(values) for var, values in csp.domains.items()}

    def restaurar_dominios(saved: dict[str, list[str]]) -> None:
        csp.domains = {var: list(values) for var, values in saved.items()}

    def legal_values(var: str) -> list[str]:
        return [
            value
            for value in csp.domains[var]
            if csp.is_consistent(var, value, assignment)
        ]

    def select_mrv_var() -> str:
        unassigned = csp.get_unassigned_variables(assignment)

        def key_fn(var: str) -> tuple[int, int]:
            mrv = len(legal_values(var))
            degree = sum(1 for n in csp.get_neighbors(var) if n not in assignment)
            return (mrv, -degree)

        return min(unassigned, key=key_fn)

    def valores_ordenados(var: str) -> list[str]:
        values = legal_values(var)
        return sorted(values, key=lambda v: csp.get_num_conflicts(var, v, assignment))

    def forward_check(var: str) -> bool:
        for neighbor in csp.get_neighbors(var):
            if neighbor in assignment:
                continue

            new_dom = [
                value
                for value in csp.domains[neighbor]
                if csp.is_consistent(neighbor, value, assignment)
            ]
            csp.domains[neighbor] = new_dom

            if not new_dom:
                return False

        return True

    def backtrack() -> dict[str, str] | None:
        nonlocal attempted_assignments

        if csp.is_complete(assignment):
            return dict(assignment)

        var = select_mrv_var()
        for value in valores_ordenados(var):
            attempted_assignments += 1
            if not csp.is_consistent(var, value, assignment):
                continue

            guardado_dom = copy_dominio()
            csp.assign(var, value, assignment)
            csp.domains[var] = [value]

            if forward_check(var):
                result = backtrack()
                if result is not None:
                    return result

            restaurar_dominios(guardado_dom)
            csp.unassign(var, assignment)

        return None

    result = backtrack()
    print(f"Attempted assignments: {attempted_assignments}")
    return result