import minizinc, time
import os
import json
import math
from datetime import timedelta

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

def solve_mcp(custom_model, file_path, origin_location, time_limit=300):
    # Load data
    m, n, l, s, D = read_instance(file_path)
    model = minizinc.Model(custom_model)
    gecode = minizinc.Solver.lookup("gecode")
    instance = minizinc.Instance(gecode, model)
    instance["m"] = m
    instance["n"] = n
    instance["l"] = l
    instance["s"] = s
    instance["D"] = D
    instance["o"] = origin_location 
    result = instance.solve(timeout=timedelta(seconds=time_limit))
    return result

def save_solution(result, elapsed_time, file_path, instance_number, approach_name):
    optimal = result.status == minizinc.result.Status.OPTIMAL_SOLUTION
    obj = result.objective if result.objective is not None else None
    sol = []  # List of routes for each courier
    for courier in result["y"]:
        route = []
        for i, location in enumerate(courier):
            for j, visit in enumerate(location[:-1]):  
                if visit == 1:
                    route.append(j + 1)  
        sol.append(route)
    solution_data = {
        approach_name: {
            "time": math.floor(elapsed_time),
            "optimal": optimal,
            "obj": obj,
            "sol": sol
        }
    }
    output_dir = os.path.join("res", "MIP")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{instance_number}.json")
    with open(output_file, "w") as json_file:
        json.dump(solution_data, json_file, indent=4)

if __name__ == "__main__":
    instances_dir = "instances"
    model = "custom_mip_model.mzn"
    origin_location = 0
    approach_name = "gecode"
    num_instances = len([f for f in os.listdir(instances_dir) if f.startswith("inst") and f.endswith(".dat")])
    # solve all the instances
    for instance_number in range(1, num_instances):
        instance_string = f"{instance_number:02}"  
        file_path = os.path.join(instances_dir, f"inst{instance_string}.dat")
        print(f"Processing {file_path}...")
        start_time = time.time()
        result = solve_mcp(model, file_path, origin_location, 300)
        end_time = time.time()
        print(result.statistics)
        if result.solution is not None and result['x'] and result['y']:
            x = result['x']
            y = result['y']
            #print(result)
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {round(elapsed_time, 3)} seconds")            
            save_solution(result, elapsed_time, file_path, instance_number, approach_name)
            #plotter.plot_courier_routes(y)
        else:
             print(f"No solution found for {file_path} within the time limit.")
        
        # Plotting the routes (optional)
        #plotter.plot_courier_routes(y)
