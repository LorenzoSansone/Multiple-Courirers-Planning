from pysat.solvers import Glucose3
from pysat.formula import CNF
import json
import os
import utils
def save_solution(sat_model, m, n, output_file):
    solution = {
        "gurobi": {
            "time": None,  # SAT does not have a time metric like Gurobi's runtime
            "optimal": True,  # Assuming a solution was found
            "obj": None,  # SAT does not directly optimize an objective
            "sol": []  # Solution route for each courier
        }
    }

    for i in range(m):
        route = []
        for j in range(n):
            if sat_model[i * n + j] > 0:  # If the literal is positive, it's part of the solution
                route.append(j + 1)  # Convert 0-based index to 1-based index
        solution["gurobi"]["sol"].append(route)

    output_dir = "res/SAT"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as outfile:
        json.dump(solution, outfile, indent=4)

    print(f"Solution saved to {output_file}")


def create_sat_problem(m, n, l, s, D, locations):
    formula = CNF()

    # Each item must be delivered by exactly one courier
    for j in range(n):
        # At least one courier delivers item j
        clause = []
        for i in range(m):
            clause.append(i * n + j + 1)  # Using 1-based index for SAT variables
        formula.append(clause)

        # No two couriers deliver the same item j
        for i1 in range(m):
            for i2 in range(i1 + 1, m):
                formula.append([-(i1 * n + j + 1), -(i2 * n + j + 1)])

    # Load constraints
    for i in range(m):
        for j in range(n):
            # If courier i picks up item j, the total load cannot exceed the capacity
            if s[j] > l[i]:
                formula.append([-(i * n + j + 1)])  # Courier i cannot pick up item j

    # Additional constraints (start/end, no subtours, etc.) could be added here
    # You would need to encode them in a similar way

    return formula


def solve_sat_problem(formula):
    solver = Glucose3()
    solver.append_formula(formula)

    if solver.solve():
        model = solver.get_model()
        print("SAT Solution Found")
        return model
    else:
        print("No solution found (UNSAT)")
        return None


if __name__ == "__main__":
    for i in range(1, 22):
        file_path = f'instances/inst{i:02d}.dat'
        print(f"################\nInstance: {file_path}")

        # Here, you would read the problem instance using your existing function
        m, n, l, s, D, locations = utils.read_input(file_path)

        formula = create_sat_problem(m, n, l, s, D, locations)
        sat_model = solve_sat_problem(formula)

        if sat_model is not None:
            output_file = f"res/SAT/inst{i:02d}.json"
            save_solution(sat_model, m, n, output_file)
        else:
            # No solution found, save a default empty solution
            save_solution([], m, n, f"res/SAT/inst{i:02d}_unsat.json")
