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
    locations = n+1
    return m, n, l, s, D, locations

def create_mcp_model(m, n, l, s, D, locations):
    model = gp.Model("MCP")
    # Variables
    # n = number of items
    # locations = number of locations (#items + 1 (origin))
    x = model.addVars(m, n, vtype=GRB.BINARY, name='x')  # x[i, j] = 1 if courier i delivers item j
    y = model.addVars(m, locations, locations, vtype=GRB.BINARY, name='y')  # y[i, j1, j2] = 1 if courier i travels from j1 to j2
    # first location is index 0,
    # second location is index 1,
    # ... last location is index n-1
    # origin's location is index
    Distance = model.addVars(m, vtype=GRB.INTEGER, name="Distance")  # Total distance for each courier
    max_distance = model.addVar(vtype=GRB.INTEGER, name="max_distance")  # Maximum distance among couriers

# 1: All must start from the origin
    for i in range(m):  # For each courier
        model.addConstr(quicksum(y[i, n, j] for j in range(locations)) == 1)  # Courier i must leave the origin
# 2: All must end at the origin
    for i in range(m):  # For each courier
        model.addConstr(quicksum(y[i, j, n] for j in range(locations)) == 1)  # Courier i must return to the origin
# 3: Load size must be respected
    for i in range(m):
        model.addConstr(quicksum(s[j]*x[i,j] for j in range(n)) <= l[i])
# 4: each courier must visit each location (except from origin) maximum one time

# max distance is the maximum among all the distances
    for i in range(m):
        model.addConstr(
            Distance[i] == quicksum(D[j1][j2] * y[i,j1,j2] 
                                    for j1 in range(locations) 
                                    for j2 in range(locations)),
                                    name=f"Distance_{i}")
    
# Objective
    model.setObjective(max_distance, GRB.MINIMIZE)
    return model, x, y, Distance, max_distance

def extract_solution(model, m, n, x, y, distance, max_dist):
    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise Exception("No optimal solution found")
    # Extract the solution
    x_sol = model.getAttr('x', x)
    # Format x into a binary matrix
    x_matrix = [[int(x_sol[i, j]) for j in range(n)] for i in range(m)]
    y_sol = model.getAttr('x', y)
    # Format y into a binary matrix
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

# Example usage
if __name__ == "__main__":
    file_path = 'instances/inst00.dat'
    m, n, l, s, D, origin = read_input(file_path)
    model, x, y, distance, max_dist = create_mcp_model(m, n, l, s, D, origin)
    solution = extract_solution(model, m, n, x, y, distance, max_dist)
    print(f"Objective = {solution['objective']}")
    print(f"x = {solution['x']}")
    print(f"y = {solution['y']}")
    print(f"tour_distance = {solution['tour_distance']}")
    print(f"max_dist = {solution['max_dist']}")

