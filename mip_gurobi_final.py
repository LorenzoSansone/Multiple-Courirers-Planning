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

def create_model(m,n,l,s,D,locations):
    model = gp.Model('mcp')
    x = model.addVars(m,n, vtype = GRB.BINARY, name = 'x')
    y = model.addVars(m, locations, locations, vtype = GRB.BINARY, name = 'y')
    Distance = model.addVars(m, vtype = GRB.INTEGER, name = 'Distance')
    max_distance = model.addVar(vtype = GRB.INTEGER, name = 'max_distance')
    # constraints
    #1 : load of each courier must be respected
    for i in range(m):
        model.addConstr(quicksum(x[i,j] * s[j] for j in range(n)) <= l[i])    
    # each item must be taken  by one courier # THIS WORKS
    for j in range(n):
        model.addConstr(quicksum(x[i,j] for i in range(m)) == 1)
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


if __name__ == "__main__":
    file_path = 'instances/inst00.dat'
    m, n, l, s, D, locations = read_input(file_path)
    model, x, y, Distance, max_distance = create_model(m,n,l,s,D,locations)
    solution = extract_solution(model, m, n ,x, y, Distance, max_distance)
