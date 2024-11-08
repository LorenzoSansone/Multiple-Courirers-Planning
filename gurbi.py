from minizinc import Instance, Model, Solver
import utils
import os, math, json, datetime
def save_solution(result, input_file, m, n):
    if result is None:
        solution = {
            "objective": None,
            "x": [[0 for _ in range(n)] for _ in range(m)],
            "y": [[[0 for _ in range(n+1)] for _ in range(n+1)] for _ in range(m)],
            "tour_distance": [0 for _ in range(m)],
            "max_dist": None
        }
    else:
        solution = result.solution
    solver = 'gurobi'
    #print(f"Elapsed time: {result.statistics['time'].total_seconds()} seconds")
    #elapsed_time = math.floor(result.statistics['time'].total_seconds())
    #print(elapsed_time)
    optimal = result.status.name == 'OPTIMAL_SOLUTION'
    objective = result.objective if result.objective is not None else None
    instance_number = input_file.split('/')[-1].split('.')[0].replace('inst', '')
    solution_dict = {
            solver: {
                "time": 0, ### TODO: FIX THIS # Time taken by the optimization
                "optimal": optimal,    # Indicate that no optimal solution was found
                "obj": objective,  # No objective value
                "sol": []  # Empty solution
            }
        }

    for courier in range(m):
            route = []
            current_location = n  # Start at the origin (assuming last location is the origin)     
            while True:
                next_location = None
                for j2 in range(n+1):
                    if solution.y[courier][current_location][j2] == 1:
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

if __name__ == "__main__":
    time_limit = 300  # time limit of 5 minutes
    for i in range(11, 22):
        print(f"\nSolving instance: inst{i:02d}.dat")
        file_path = f'instances/inst{i:02d}.dat'
        m, n, l, s, D, locations = utils.read_input(file_path)
        model = Model("gurbi.mzn")
        solver = Solver.lookup("gurobi")
        instance = Instance(solver, model)
        instance["m"] = m
        instance["n"] = n
        instance["l"] = l
        instance["s"] = s
        instance["D"] = D
        instance["locations"] = locations
        result = instance.solve(timeout=datetime.timedelta(seconds = time_limit))    
        print(f"Solved instance: {file_path}")
        instance_number = utils.get_instance_number(file_path)
        output_file = save_solution(result, f"inst{instance_number}.dat", m, n)