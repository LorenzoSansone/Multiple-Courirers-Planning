import json
import os
from utils_sat import *
import time
from z3 import *

def save_solution(sat_model, m, n, output_file):
    solution = {
        "glucose3": {
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
        solution["glucose3"]["sol"].append(route)

    output_dir = "res/SAT"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as outfile:
        json.dump(solution, outfile, indent=4)

    print(f"Solution saved to {output_file}")

def mcp_sat(m, n, l, s, D):
    start_time = time.time()

    s = Solver()

    deposit = n #if we count from 0

    #VARIABLES

    # path[i][j][k] = T if the courier i delivers the package j at the k-th step 
    path = [[[Bool(f"p_{courier}_{package}_{step}") for step in range(n+2)] for package in range(n+1)] for courier in range(m)]
    
    # weight[i][j] = T if the courier i take the package  j
    weight = [[Bool(f"w_{courier}_{package}")for package in range(n+1)] for courier in range(m)]

    
    #CONSTRAINTS

    #1: the courier delivers exactly one package at each step
    for courier in range(m):
        for step in range(n+2):
            s.add(exactly_one_he(path[courier][:][step]))
    
    
    #2: Each package is carried only once
    for package in [x for x in range(n+1) if x!=deposit]:
        s.add(exactly_one_he(path[courier][package][:]))

    #3: Couriers start and end at the deposit
    for courier in range(m):
        s.add(path[courier][deposit][0])
        s.add(path[courier][deposit][n+1]) #n+1 means n+2 but we count from zero

    #4
    
    
    
    
    
    end_time = time.time()
    



    return ""


if __name__ == "__main__":

    first_instance = 1
    last_instance = 1
    file_name_save = 'result_model.txt'
    file_name_error = 'error_model.txt'
    mode_save = 'w'
    mode_save_error = "a"

    for i in range(first_instance, last_instance+1):
        file_path = f'instances/inst{i:02d}.dat'
        inst_i = f"inst{i:02d}" 
        data_path = f"../instances_dzn/{inst_i}.dzn"
        m, n, l, s, D = read_instance(data_path)
        mcp_sat(m, n, l, s, D)
    

        
