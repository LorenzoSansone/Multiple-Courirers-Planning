import gurobipy as gp
from gurobipy import GRB, quicksum
import json
import os
import math





def read_input(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    m = int(lines[0].strip())  # number of couriers
    n = int(lines[1].strip())  # number of items
    l = list(map(int, lines[2].strip().split()))  # max load size for each courier
    s = list(map(int, lines[3].strip().split()))  # sizes of the items
    D = [list(map(int, line.strip().split())) for line in lines[4:]]  # distances
    locations = n + 1
    return m, n, l, s, D, locations

def check_load_sizes(x, s, m, n, l):
    load_sizes = [0] * m
    for courier in range(m):
        load_size = sum(x[courier][j] * s[j] for j in range(n))
        load_sizes[courier] = load_size
        print(f"Courier {courier} loaded: {load_sizes[courier]}, maximum capacity is {l[courier]}: {load_sizes[courier] <= l[courier]}")
    return load_sizes

def check_if_every_courier_starts_at_origin(y, n, m):
    origin = n+1
    for courier in range(m):
        starts_at_origin = any(y[courier][n][j] == 1 for j in range(n+1) if j != origin)
        if not starts_at_origin:
            print(f"Courier {courier} does not start at the origin.")
            return False

    return True

def check_if_every_courier_ends_at_origin(y, n, m):
    origin = n+1
    for courier in range(m):
        ends_at_origin = any(y[courier][j][n] == 1 for j in range(n+1) if j != origin-1   )
        if not ends_at_origin:
            print(f"Courier {courier} does not end at the origin.")
            return False
    return True

def picked_up_objects(x, m, n):
    for courier in range(m):
        print(f"Courier {courier} picked up objects: ")
        loaded_objects = []
        for item in range(n):
            if x[courier][item] == 1:
                loaded_objects.append(item)
        print(loaded_objects)

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
        model.addConstr(quicksum(x[i, j] * s[j] for j in range(n))  <= l[i])

    # Each item must be delivered by exactly one courier
    for j in range(n):
        model.addConstr(quicksum(x[i, j] for i in range(m)) == 1)

    # Each courier starts at the origin (location n) and ends at the origin
    for courier in range(m):
        model.addConstr(quicksum(y[courier, n, j] for j in range(n)) == 1)
        model.addConstr(quicksum(y[courier, j, n] for j in range(n)) == 1)

    # Prevent couriers from traveling from a point to itself
    for i in range(m):
        for j in range(locations):
            model.addConstr(y[i, j, j] == 0)
    # if a courier picks up an item, they must visit the item's location and leave it
    for i in range(m):
        for j in range(n):
            model.addConstr(quicksum(y[i,j,k] for k in range(locations)) == x[i,j])
            model.addConstr(quicksum(y[i,k,j] for k in range(locations)) == x[i,j])

    # Flow conservation constraints
    for i in range(m):  # for each courier
        for j in range(1, ):  # for each location, excluding the origin
            # If the courier arrives at location j
            model.addConstr(quicksum(y[i, k, j] for k in range(locations)) ==
                            quicksum(y[i, j, k] for k in range(locations)))

    # Subtour elimination constraints
    for i in range(m):  # for each courier
        for j in range(0, n):  # for each location, excluding the origin
            for k in range(0, n):
               if j != k:
                   model.addConstr(u[i, j] - u[i, k] + (n - 1) * y[i, j, k] <= n - 2)

    # Set bounds for u variables
    for i in range(m):
        for j in range(1, n):
            model.addConstr(u[i, j] >= 1)
            model.addConstr(u[i, j] <= n - 1)
    # Distance calculation constraint
    for i in range(m):
        model.addConstr(
            Distance[i] ==
            quicksum(
                y[i, j1, j2] * D[j1][j2]
                for j1 in range(locations)
                for j2 in range(locations)
            )
        )
    # Max distance objective
    for i in Distance:
        model.addConstr(max_distance >= Distance[i])
    model.setObjective(max_distance, GRB.MINIMIZE)
    return model, x, y, Distance, max_distance

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
        max_dist_val = max_dist.x
        distance_values = [distance[i].x for i in range(m)]
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

def distances_check(D, y_matrix):
    for courier_index in range(len(y_matrix)):
        route_distances = []
        total_distance = 0
        current_location = len(D) - 1  # Start from origin, assuming it's the last index

        while True:
            next_location = None
            for j2 in range(len(y_matrix[courier_index][current_location])):
                if y_matrix[courier_index][current_location][j2] == 1:
                    next_location = j2
                    break
            
            if next_location is None or next_location == len(D) - 1:
                break  # No further movement or return to origin

            # Calculate the distance between the current location and the next location
            distance = D[current_location][next_location]
            route_distances.append(distance)
            total_distance += distance

            # Move to the next location
            current_location = next_location

        # Add the distance back to the origin
        if current_location != len(D) - 1:
            distance_back_to_origin = D[current_location][len(D) - 1]
            route_distances.append(distance_back_to_origin)
            total_distance += distance_back_to_origin

        # Prepare the distance string
        distance_str = " + ".join(map(str, route_distances)) + f" = {total_distance}"
        output_str = f"Courier {courier_index}: {distance_str}"
        
        # Print to the console
        print(output_str)

def print_routes_from_solution(solution):
    y = solution['y']
    n = len(y[0]) - 1
    for courier_index in range(len(y)):
        route = []
        current_location = n  # Start from origin (assuming the origin is at index 'n')
        
        while True:
            route.append(current_location + 1) #correct indexing
            next_location = None
            for j2 in range(len(y[courier_index][current_location])):
                if y[courier_index][current_location][j2] == 1:
                    next_location = j2
                    break
            
            if next_location == n or next_location is None:
                break  
            else:
                current_location = next_location
        
        route.append(n + 1)
        
        print(f"Courier {courier_index}: {' -> '.join(map(str, route))}")

if __name__ == "__main__":
    OBJ_THRESHOLD = 8000.0
    for i in range(1,22):
        file_path = f'instances/inst{i:02d}.dat'
        print(f"Instance: {file_path}")
        m, n, l, s, D, origin = read_input(file_path)
        model, x, y, distance, max_dist = create_mcp_model(m, n, l, s, D, origin)
        solution = extract_solution(model, m, n, x, y, distance, max_dist)
        if solution is None or solution['objective'] > OBJ_THRESHOLD:
            # No solution found, save a default solution structure
            solution = {
                "objective": None,
                "x": [[0 for _ in range(n)] for _ in range(m)],
                "y": [[[0 for _ in range(n+1)] for _ in range(n+1)] for _ in range(m)],
                "tour_distance": [0 for _ in range(m)],
                "max_dist": None
            }

        instance_number = file_path.split('/')[-1].split('.')[0].replace('inst', '')
        output_file = save_solution(solution, f"inst{instance_number}.dat", m, n)

        # Other useful debugging stuff below
        #
        #
        # print (f"x = {solution['x']}")
        # print(f"y = {solution['y']}")
        # Print picked up objects
        # picked_objects = picked_up_objects(solution['x'], m, n)
        # Check load sizes
        # calculated_load_sizes = check_load_sizes(solution['x'], s, m, n, l)
        # all_start_at_origin = check_if_every_courier_starts_at_origin(solution['y'], n, m)
        # all_end_at_origin = check_if_every_courier_ends_at_origin(solution['y'], n, m)
        # print(f"All couriers start at the origin: {all_start_at_origin}")
        # print(f"All couriers end at the origin: {all_end_at_origin}")
        # Print routes
        #print_routes_from_solution(solution)
        #Print distances
        #distances_check(D, solution['y'])