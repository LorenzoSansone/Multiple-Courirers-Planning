import minizinc, time
# read the instance file and save values in variables m,n,l,s,D
def read_instance(file_path):
    with open(file_path, "r") as file:
        m = int(file.readline().strip())
        n = int(file.readline().strip())
        l = list(map(int, file.readline().strip().split()))
        s = list(map(int, file.readline().strip().split()))
        D = []
        for _ in range(n + 1):  # n+1 rows for the distance matrix
            D.append(list(map(int, file.readline().strip().split())))
        return m, n, l, s, D
    
def solve_mcp(custom_model, file_path, origin_location):
    # Load data
    m, n, l, s, D = read_instance(file_path)
    # Load model
    model = minizinc.Model(custom_model)
    # solver
    gecode = minizinc.Solver.lookup("gecode")
    # Create minizinc instance
    instance = minizinc.Instance(gecode, model)
    instance["m"] = m
    instance["n"] = n
    instance["l"] = l
    instance["s"] = s
    instance["D"] = D
    instance["o"] = origin_location 

    # Solve the problem
    result = instance.solve()

    return result

if __name__ == "__main__":
    file_path = "instances/instance_4.txt"
    model = "custom_mip_model.mzn"
    origin_location = 1
    start_time = time.time()
    result = solve_mcp(model, file_path, origin_location)
    end_time = time.time()
    print(result)
    print(f"Elapsed time: {round(end_time - start_time,3)} seconds")
