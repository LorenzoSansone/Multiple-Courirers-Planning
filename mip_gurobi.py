import gurobipy as gp
from gurobipy import GRB, quicksum
import json
import os
import math
import utils

def save_solution(solution, input_file, m, n):
    instance_number = input_file.split('/')[-1].split('.')[0].replace('inst', '')
    if solution['objective'] is None:
        solution_dict = {
            "gurobi": {
                "time": math.floor(model.Runtime),  # Time taken by the optimization
                "optimal": False,  # Indicate that no optimal solution was found
                "obj": None,  # No objective value
                "sol": []  # Empty solution
            }
        }
    else:
        # Prepare the solution dictionary in the required format
        solution_dict = {
            "gurobi": {
                "time": math.floor(model.Runtime),  # Time taken by the optimization
                "optimal": model.status == GRB.OPTIMAL,  # True if the solution is optimal
                "obj": solution['max_dist'],  # The objective value (max distance)
                "sol": []  # List of lists representing the solution (routes)
            }
        }

        for courier in range(m):
            route = []
            current_location = n  # Start at the origin (assuming last location is the origin)
            
            while True:
                next_location = None
                for j2 in range(n+1):
                    if solution['y'][courier][current_location][j2] == 1:
                        next_location = j2
                        break

                if next_location is None or next_location == n:
                    break  # No further movement or return to origin

                route.append(next_location + 1) 
                current_location = next_location
            
            solution_dict["gurobi"]["sol"].append(route)

    output_dir = "res/MIP"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{instance_number}.json")
    with open(output_file, 'w') as outfile:
        json.dump(solution_dict, outfile, indent=4)

    print(f"Solution saved to {output_file}")
    return output_file

def create_mcp_model(m, n, l, s, D, locations):
    model = gp.Model("MCP")
    # Variables
    x = model.addVars(m, n, vtype=GRB.BINARY, name='x')  # x[i, j] = 1 if courier i delivers item j
    y = model.addVars(m, locations, locations, vtype=GRB.BINARY, name='y')  # y[i, j1, j2] = 1 if courier i travels from j1 to j2
    Distance = model.addVars(m, vtype=GRB.INTEGER, name="Distance")  # Total distance for each courier
    max_distance = model.addVar(vtype=GRB.INTEGER, name="max_distance")  # Maximum distance among couriers
    # Add variables for subtour elimination
    u = model.addVars(m, n, vtype=GRB.CONTINUOUS, name="u")
    # Load constraints
    for i in range(m):
        model.addConstr(quicksum(x[i, j] * s[j] for j in range(n))  <= l[i],
                        name=f"Load constraint: courier {i}")

    # Each item must be delivered by exactly one courier
    for j in range(n):
        model.addConstr(quicksum(x[i, j] for i in range(m)) == 1,
                        name = f"Delivery by one courier constraint: courier {i}")

    # Each courier starts at the origin (location n) and ends at the origin
    for courier in range(m):
        model.addConstr(quicksum(y[courier, n, j] for j in range(n)) == 1, name = f"Start at origin, courier {i}")
        model.addConstr(quicksum(y[courier, j, n] for j in range(n)) == 1, name = f"End at origin, courier {i}")

    # Prevent couriers from traveling from a point to itself
    for i in range(m):
        for j in range(locations):
            model.addConstr(y[i, j, j] == 0,
                            name = f"Travel from point to itself, courier {i}, location {j}")
    # if a courier picks up an item, they must visit the item's location and leave it
    for i in range(m):
        for j in range(n):
            model.addConstr(quicksum(y[i,j,k] for k in range(locations)) == x[i,j],
                            name = f"Visit item location, courier {i}, location {j}")
            model.addConstr(quicksum(y[i,k,j] for k in range(locations)) == x[i,j],
                            name = f"Lave item location, courier {i}, location {j}")

    # Flow conservation constraints
    for i in range(m):  # for each courier
        for j in range(1, locations):  # for each location, excluding the origin
            # If the courier arrives at location j
            model.addConstr(quicksum(y[i, k, j] for k in range(locations)) ==
                            quicksum(y[i, j, k] for k in range(locations)),
                            name = f"Flow conservation constraint, couerier {i}, location {j}")

    # Subtour elimination constraints
    for i in range(m):  # for each courier
        for j in range(0, n):  # for each location, excluding the origin
            for k in range(0, n):
                if j != k:
                    model.addConstr(u[i, j] - u[i, k] + (n - 1) * y[i, j, k] <= n - 2,
                    name = f"Subtour elimination constraint, courier {i}, locations {j} and {k}")

    # Set bounds for u variables
    for i in range(m):
        for j in range(1, n):
            model.addConstr(u[i, j] >= 1)
            model.addConstr(u[i, j] <= n - 1,
                            name = f"Set bounds for u variable: courier {i}, location {j}")
    # Distance calculation constraint
    for i in range(m):
        model.addConstr(
            Distance[i] ==
            quicksum(
                y[i, j1, j2] * D[j1][j2]
                for j1 in range(locations)
                for j2 in range(locations)
            )
        , name = f"Distance calculation for courier {i}"
       )
    # Max distance objective
    for i in Distance:
        model.addConstr(max_distance >= Distance[i],
                        name = f"Max distance (objective)")
    model.setObjective(max_distance, GRB.MINIMIZE)
    return model, x, y, Distance, max_distance

def diagnose_infeasibility(model):
    model.computeIIS()
    print("\nThe following constraints are causing the model to be infeasible:")
    for c in model.getConstrs():
        if c.IISConstr:
            print(f"Infeasible constraint: {c.constrName}")

def extract_solution(model, m, n, x, y, distance, max_dist):
    model.Params.MIPGap = 0.05    # 5% gap on time limit
    model.Params.TimeLimit = 300
    model.optimize()
    
    # Check if the model was stopped due to time limit or other reasons
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.") 
    elif model.status == GRB.TIME_LIMIT:
        print("Time limit reached. Best feasible solution obtained.")
    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible; no solution found.")
        diagnose_infeasibility(model)
        return None
    else:
        print(f"Optimization was stopped with status {model.status}")
        return None

    # Extract the solution
    try:
        x_sol = model.getAttr('x', x)
        x_matrix = [[int(x_sol[i, j]) for j in range(n)] for i in range(m)]
        y_sol = model.getAttr('x', y)
        y_matrix = [[[int(y_sol[i, j1, j2]) for j2 in range(len(D))] for j1 in range(len(D))] for i in range(m)]
        distance_values = [distance[i].x for i in range(m)]
        max_dist_val = max(distance_values)
        return {
            "objective": max_dist_val,
            "x": x_matrix,
            "y": y_matrix,
            "tour_distance": distance_values,
            "max_dist": max_dist_val
        }
    except gp.GurobiError as e:
        print(f"Error retrieving solution: {e}")
        return None
def debug(x,y,m,n,s,l,D):
    utils.picked_up_objects(x, m, n)
    # Check load sizes
    print(utils.check_load_sizes(x, s, m, n, l))
    #check start and end
    print(f"All couriers start at the origin: {utils.check_if_every_courier_starts_at_origin(y, n, m)}")
    print(f"All couriers end at the origin: {utils.check_if_every_courier_ends_at_origin(y, n, m)}")
    #check if all items are being taken
    print(f"All items are being taken by a courier: {utils.check_if_items_are_taken_by_couriers(x, m, n)}")
    # Print routes
    print("Routes for each courier:")
    utils.print_routes_from_solution(solution)
    #Print distances
    print("Distances calculation for each courier:")
    utils.distances_check(D, solution['y'])


if __name__ == "__main__":
    for i in range(1,10):
        file_path = f'instances/inst{i:02d}.dat'
        print(f"################\nInstance: {file_path}")
        m, n, l, s, D, origin = utils.read_input(file_path)
        model, x, y, distance, max_dist = create_mcp_model(m, n, l, s, D, origin)
        solution = extract_solution(model, m, n, x, y, distance, max_dist)
        if solution is None: 
            # No solution found, save a default solution structure
            solution = {
                "objective": None,
                "x": [[0 for _ in range(n)] for _ in range(m)],
                "y": [[[0 for _ in range(n+1)] for _ in range(n+1)] for _ in range(m)],
                "tour_distance": [0 for _ in range(m)],
                "max_dist": None
            }
        instance_number = utils.get_instance_number(file_path)
        output_file = save_solution(solution, f"inst{instance_number}.dat", m, n)
        # debug functions
        #debug(solution['x'], solution['y'], m, n, s, l, D)
