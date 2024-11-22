import math, os, json
import numpy as np
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
def save_solution_by_model(input_file, m, n, model_name, time_limit=300, result=None):
    # Determine the output file path
    instance_number = input_file.split('/')[-1].split('.')[0].replace('inst', '')
    output_dir = "res/MIP"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{instance_number}.json")
    
    # Prepare the solution dictionary
    if result is None or result.solution is None:
        # Default solution if no result or solution found
        solution = {
            "objective": None,
            "x": [[0 for _ in range(n)] for _ in range(m)],
            "y": [[[0 for _ in range(n+1)] for _ in range(n+1)] for _ in range(m)],
            "tour_distance": [0 for _ in range(m)],
            "max_dist": None
        }
        optimal = False
        objective = None
    else:
        solution = result.solution
        optimal = (result.status.name == 'OPTIMAL_SOLUTION')
        objective = result.objective if result.objective is not None else None

    # Create a dictionary to store solution details
    solution_data = {
        "y": [[[0 for _ in range(n+1)] for _ in range(n+1)] for _ in range(m)]
    }

    # Try to extract 'y' from solution if it exists
    if hasattr(solution, "y"):
        solution_data["y"] = solution.y

    # Prepare the solver-specific solution dictionary
    solver_solution_dict = {
        "time": time_limit if result.status.name != 'OPTIMAL_SOLUTION' else math.floor(result.statistics['solveTime'].total_seconds()),
        "optimal": optimal,
        "obj": objective,
        "sol": []
    }

    # Populate the solution routes
    if hasattr(solution, "y"):
        for courier in range(m):
            route = []
            current_location = n  # Start at the origin (assuming last location is the origin)
            while True:
                next_location = None
                for j2 in range(n+1):
                    if solution_data["y"][courier][current_location][j2] == 1:
                        next_location = j2
                        break

                if next_location is None or next_location == n:
                    break  # No further movement or return to origin

                route.append(next_location + 1)
                current_location = next_location
            
            solver_solution_dict["sol"].append(route)
    else:
        solver_solution_dict["sol"] = []

    # Read existing solutions or create new dictionary
    try:
        with open(output_file, 'r') as infile:
            existing_solutions = json.load(infile)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_solutions = {}

    # Add or update the current solver's solution
    existing_solutions[model_name] = solver_solution_dict

    # Write updated solutions back to the file
    with open(output_file, 'w') as outfile:
        json.dump(existing_solutions, outfile, indent=4)

    print(f"Solution saved to {output_file}")
    return output_file


def save_solution_by_solver(input_file, m, n, solver_name, time_limit=300, result=None):
    # Determine the output file path
    instance_number = input_file.split('/')[-1].split('.')[0].replace('inst', '')
    output_dir = "res/MIP"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{instance_number}.json")
    
    # Prepare the solution dictionary
    if result is None or result.solution is None:
        # Default solution if no result or solution found
        solution = {
            "objective": None,
            "x": [[0 for _ in range(n)] for _ in range(m)],
            "y": [[[0 for _ in range(n+1)] for _ in range(n+1)] for _ in range(m)],
            "tour_distance": [0 for _ in range(m)],
            "max_dist": None
        }
        optimal = False
        objective = None
    else:
        solution = result.solution
        optimal = (result.status.name == 'OPTIMAL_SOLUTION')
        objective = result.objective if result.objective is not None else None

    # Create a dictionary to store solution details
    solution_data = {
        "y": [[[0 for _ in range(n+1)] for _ in range(n+1)] for _ in range(m)]
    }

    # Try to extract 'y' from solution if it exists
    if hasattr(solution, "y"):
        solution_data["y"] = solution.y

    # Prepare the solver-specific solution dictionary
    solver_solution_dict = {
        "time": math.floor(result.statistics['solveTime'].total_seconds()) if result.status.name == 'OPTIMAL_SOLUTION' else time_limit,
        "optimal": optimal,
        "obj": objective,
        "sol": []
    }

    # Populate the solution routes
    if hasattr(solution, "y"):
        for courier in range(m):
            route = []
            current_location = n  # Start at the origin (assuming last location is the origin)
            while True:
                next_location = None
                for j2 in range(n+1):
                    if solution_data["y"][courier][current_location][j2] == 1:
                        next_location = j2
                        break

                if next_location is None or next_location == n:
                    break  # No further movement or return to origin

                route.append(next_location + 1)
                current_location = next_location
            
            solver_solution_dict["sol"].append(route)
    else:
        solver_solution_dict["sol"] = []

    # Read existing solutions or create new dictionary
    try:
        with open(output_file, 'r') as infile:
            existing_solutions = json.load(infile)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_solutions = {}

    # Add or update the current solver's solution
    existing_solutions[solver_name] = solver_solution_dict

    # Write updated solutions back to the file
    with open(output_file, 'w') as outfile:
        json.dump(existing_solutions, outfile, indent=4)

    print(f"Solution saved to {output_file}")
    return output_file
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
def print_routes_from_solution(y):
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
def get_instance_number(file_path):
    return file_path.split('/')[-1].split('.')[0].replace('inst', '')
def check_if_items_are_taken_by_couriers(x, m, n):
    
    for item in range(n):
        count = 0
        for courier in range(m):
            if x[courier][item] == 1:
                count += 1
        if count != 1:
            print(f"Item {item} is delivered by {count} couriers instead of exactly one.")
            return False
    return True
def find_boundaries_standard(m, n, l, s, D):
    distances = np.array(D)  # Convert distance matrix to numpy array for easier handling
    
    # Lower bound (LB): Maximum of minimum distances for a round trip to each item
    min_dist_dep_list = []
    for i in range(n):
        min_dist_dep_list.append(distances[n, i] + distances[i, n])  # To and from the origin
    LB = max(min_dist_dep_list)  # The largest of these minimum round trips

    # Upper bound (UB): Greedy estimate for total tour distance
    UB = distances[n, 0]  # Start from origin
    for i in range(n - 1):
        UB += distances[i, i + 1]  # Add distances between consecutive locations
    UB += distances[n - 1, n]  # Return to the origin
    UB = int(UB)

    return 0, UB, LB, UB  # Return values as in the example

def find_boundaries_hybrid(m: int, n: int, l: list, s: list, D: list):
    """
    Calculate bounds for MCP returning single LB and UB
    Args:
        m: number of couriers
        n: number of items
        l: courier capacities
        s: item sizes
        D: distance matrix
    Returns:
        Tuple of (LB, UB)
    """
    distances = np.array(D)
    
    # Lower Bound: Maximum round-trip
    min_dist_dep_list = []
    for i in range(n):
        min_dist_dep_list.append(distances[n, i] + distances[i, n])
    LB = max(min_dist_dep_list)
    
    # Upper Bound: Enhanced nearest neighbor
    current = n  # Start at origin
    unvisited = set(range(n))
    UB = 0
    
    while unvisited:
        # Find nearest unvisited point
        if len(unvisited) == 1:
            next_point = unvisited.pop()
        else:
            next_point = min(unvisited, 
                           key=lambda x: distances[current, x])
            unvisited.remove(next_point)
        
        # Add distance to next point
        UB += distances[current, next_point]
        current = next_point
    
    # Return to origin
    UB += distances[current, n]
    UB = int(UB)
    
    return LB, UB
def debug(x,y,m,n,s,l,D):
    picked_up_objects(x, m, n)
    # Check load sizes
    print(check_load_sizes(x, s, m, n, l))
    #check start and end
    print(f"All couriers start at the origin: {check_if_every_courier_starts_at_origin(y, n, m)}")
    print(f"All couriers end at the origin: {check_if_every_courier_ends_at_origin(y, n, m)}")
    #check if all items are being taken
    print(f"All items are being taken by a courier: {check_if_items_are_taken_by_couriers(x, m, n)}")
    # Print routes
    print("Routes for each courier:")
    print_routes_from_solution(y)
    #Print distances
    print("Distances calculation for each courier:")
    distances_check(D, y)


def verify_solution(x_sol, y_sol, l, s, D, n, m, locations):
    # Check load constraints
    load_valid = all(
        sum(x_sol[i-1][j-1] * s[j-1] for j in range(1, n+1)) <= l[i-1] 
        for i in range(1, m+1)
    )
    
    # Check item assignment constraint
    item_assignment_valid = all(
        sum(x_sol[i-1][j-1] for i in range(1, m+1)) == 1 
        for j in range(1, n+1)
    )
    
    # Check courier starts and ends at origin
    origin_constraint_valid = all(
        (sum(y_sol[i-1][locations-1][j-1] for j in range(1, n+1)) == 1) and
        (sum(y_sol[i-1][j-1][locations-1] for j in range(1, n+1)) == 1)
        for i in range(1, m+1)
    )
    
    # Check flow conservation
    flow_conservation_valid = all(
        sum(y_sol[i-1][k-1][j-1] for k in range(1, locations+1)) ==
        sum(y_sol[i-1][j-1][k-1] for k in range(1, locations+1))
        for i in range(1, m+1)
        for j in range(2, locations+1)
    )
    
    # Calculate total distances
    distances = [
        sum(y_sol[i-1][j1-1][j2-1] * D[j1-1][j2-1] 
        for j1 in range(1, locations+1) 
        for j2 in range(1, locations+1))
        for i in range(1, m+1)
    ]
    
    return {
        'load_valid': load_valid,
        'item_assignment_valid': item_assignment_valid,
        'origin_constraint_valid': origin_constraint_valid,
        'flow_conservation_valid': flow_conservation_valid,
        'distances': distances,
        'is_valid': (load_valid and item_assignment_valid and 
                     origin_constraint_valid and flow_conservation_valid)
    }