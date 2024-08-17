import gurobipy as gp
from gurobipy import GRB, quicksum

def read_input(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    m = int(lines[0].strip())  # number of couriers
    n = int(lines[1].strip())  # number of items
    l = list(map(int, lines[2].strip().split()))  # max load size for each courier
    s = list(map(int, lines[3].strip().split()))  # sizes of the items
    D = [list(map(int, line.strip().split())) for line in lines[4:]]  # distances
    locations = n + 1
    print(l)
    return m, n, l, s, D, locations

def create_mcp_model(m, n, l, s, D, locations):
    model = gp.Model("MCP")

    # Variables
    x = model.addVars(m, n, vtype=GRB.BINARY, name='x')  # x[i, j] = 1 if courier i delivers item j
    y = model.addVars(m, locations, locations, vtype=GRB.BINARY, name='y')  # y[i, j1, j2] = 1 if courier i travels from j1 to j2
    Distance = model.addVars(m, vtype=GRB.INTEGER, name="Distance")  # Total distance for each courier
    max_distance = model.addVar(vtype=GRB.INTEGER, name="max_distance")  # Maximum distance among couriers

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

    # Distance constraints
    for i in range(m):
        model.addConstr(Distance[i] == quicksum(D[j1][j2] * y[i, j1, j2] for j1 in range(locations) for j2 in range(locations)))

    # Max distance constraint
    model.addConstr(max_distance == quicksum(Distance[i] for i in range(m)) / m)

    # Objective
    model.setObjective(max_distance, GRB.MINIMIZE)
    return model, x, y, Distance, max_distance

def extract_solution(model, m, n, x, y, distance, max_dist):
    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise Exception("No optimal solution found")
    x_sol = model.getAttr('x', x)
    x_matrix = [[int(x_sol[i, j]) for j in range(n)] for i in range(m)]
    y_sol = model.getAttr('x', y)
    y_matrix = [[[int(y_sol[i, j1, j2]) for j2 in range(len(D))] for j1 in range(len(D))] for i in range(m)]
    max_dist_val = max_dist.x
    tour_distance = [sum(D[j1][j2] * y_sol[i, j1, j2] for j1 in range(len(D)) for j2 in range(len(D))) for i in range(m)]
    return {
        "objective": max_dist_val,
        "x": x_matrix,
        "y": y_matrix,
        "tour_distance": tour_distance,
        "max_dist": max_dist_val
    }

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
def print_routes(y_matrices, m, n):

    for courier in range(m):
        print(f"Courier {courier}'s route:")
        route = []
        locations = len(y_matrices[courier])
        current_location = None
        
        # Find the starting location (origin)
        for j1 in range(locations):
            if y_matrices[courier][j1].count(1) > 0:  # if there's any outgoing edge from j1
                current_location = j1
                break

        if current_location is None:
            print("  No starting location found.")
            continue

        # Traverse the route
        visited_locations = set()
        visited_locations.add(current_location)
        while True:
            next_location = None

            # Find the next location to travel to
            for j2 in range(locations):
                if y_matrices[courier][current_location][j2] == 1 and j2 not in visited_locations:
                    next_location = j2
                    break

            # If no next location is found, or we return to origin and we're not in the loop yet, end the route
            if next_location is None or (next_location == 7 and current_location == 7):
                break

            # Add the path to the route and update current_location
            route.append((current_location, next_location))
            visited_locations.add(next_location)
            current_location = next_location

        # Print the route
        if route:
            for (from_loc, to_loc) in route:
                print(f"  From location {from_loc} to location {to_loc}")
        else:
            print("  No valid route found.")

# Example usage
if __name__ == "__main__":
    file_path = 'instances/inst03.dat'
    m, n, l, s, D, origin = read_input(file_path)
    model, x, y, distance, max_dist = create_mcp_model(m, n, l, s, D, origin)
    solution = extract_solution(model, m, n, x, y, distance, max_dist)
    print(f"Objective = {solution['objective']}")
    print(f"x = {solution['x']}")
    print(f"y = {solution['y']}")
    print(f"tour_distance = {solution['tour_distance']}")
    print(f"max_dist = {solution['max_dist']}")

    # Print picked up objects
    picked_objects = picked_up_objects(solution['x'], m, n)

    # Check load sizes
    calculated_load_sizes = check_load_sizes(solution['x'], s, m, n, l)
    
    # Check if every courier starts and ends at the origin
    all_start_at_origin = check_if_every_courier_starts_at_origin(solution['y'], n, m)
    all_end_at_origin = check_if_every_courier_ends_at_origin(solution['y'], n, m)
    print(f"All couriers start at the origin: {all_start_at_origin}")
    print(f"All couriers end at the origin: {all_end_at_origin}")
    #Print routes
    print_routes(solution['y'], m, n)