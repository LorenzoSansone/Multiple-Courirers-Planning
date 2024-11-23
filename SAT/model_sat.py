import json
import os
from utils_sat import *
import time
from z3 import *
import numpy as np

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
    max_load = sum(s)

    #VARIABLES

    # path[i][j][k] = T if the courier i delivers the package j at the k-th step 
    path = [[[Bool(f"p_{courier}_{package}_{step}") for step in range(n+2)] for package in range(n+1)] for courier in range(m)]
    
    # courier_weights[i][j] = T if the courier i take the package  j
    courier_weights = [[Bool(f"w_{courier}_{package}")for package in range(n)] for courier in range(m)]

    # courier_loads_i = it represents binary representation of actual load carried by each courier
    courier_loads = [[Bool(f"cl_{courier}_{bit}") for bit in range(num_bits(max_load))] for courier in range(m)]



    #Binary conversion of s
    s_max = max(s)
    s_b = [int_to_binary(s_value, num_bits(s_max)) for s_value in s]

    #Binary conversion of l
    l_max = max(l)
    l_b = [int_to_binary(l_value, num_bits(l_max)) for l_value in l]

    #Binary conversion of D
    D_max = np.matrix(D).max()
    D_b = [[int_to_binary(D[i][j], num_bits(D_max)) for j in range(n+1)] for i in range(n+1)]

    
    #-----CONSTRAINTS-----

    # Binding the weight and path
    for courier in range(m):
        for step in range(n+2):
            for package in range(n): #we don't consider the depoist
                s.add(Implies(path[courier][package][step], courier_weights[courier][package]))

    #1: the courier delivers exactly one package at each step
    for courier in range(m):
        for step in range(n+2):
            s.add(at_most_one_he([path[courier][package][step] for package in range(n+1)]))
    
    #2: Each package is carried only once
    for package in range(n): #not consider n+1 (deposit)
        #s.add(exactly_one_he(path[courier][package][:]))
        s.add(exactly_one_he([path[courier][package][step] for courier in range(m) for step in range(n+2)]))

    #3: Couriers start and end at the deposit
    for courier in range(m):
        s.add(path[courier][deposit][0])
        s.add(path[courier][deposit][n+1]) #n+1 means n+2 because we count from zero

    #4: Every courier has a maximum load capacity to respect
    for i in range(m):
        #s.add(conditional_sum_K_bin(a[i], s_bin, courier_loads[i], f"compute_courier_load_{i}"))
        s.add(geq(l_b[i],courier_loads[i]))
    

    #5: All couriers must start as soon as possible
    #1 is the first step
    #range(n) because they have to pick one package and not choose the deposit (n+1)
    for courier in range(m):
        s.add(at_least_one_he([path[courier][i][1] for i in range(n)]))


    #6: if a courier doesn't take the a pack at position j, also at position j+1 doesn't take any pack
    # So if a courier is in the deposit at step 1 (it starts at 0) it means that he will not deliver any pack
    # it also means that the courier can come back to the deposit if he has to deliver other packagages
    for courier in range(m):
        for step in range(1,n):
            s.add(Implies(path[courier][deposit][step], path[courier][deposit][step+1]))
        
    end_time = time.time()
    
    return ""


if __name__ == "__main__":

    first_instance = 1
    last_instance = 1
    file_name_save = 'result_model.txt'
    file_name_error = 'error_model.txt'
    mode_file_result = 'w'
    mode_file_error = "a"
    """
    for i in range(first_instance, last_instance+1):
        file_path = f'instances/inst{i:02d}.dat'
        inst_i = f"inst{i:02d}" 
        data_path = f"../instances_dzn/{inst_i}.dzn"
        m, n, l, s, D = read_instance(data_path)
        #mcp_sat(m, n, l, s, D)
    """
    n = 5
    m = 2
    s = Solver()
    max_load = 100
    #courier_loads = [[Bool(f"cl_{courier}_{bit}") for bit in range(num_bits(max_load))] for courier in range(m)]
    path = [[[Bool(f"p_{courier}_{package}_{step}") for step in range(n+2)] for package in range(n+1)] for courier in range(m)]

    """
    for courier in range(m):
        for step in range(n+2):
            s.add(at_most_one_he(path[courier][:][step]))
    

    if s.check() == sat:
        print(s.model())
    else:
        print(s.check())
    """
    """
    x = [Bool("x_0")]
    y = [Bool("y_0"),Bool("y_1")]
    #y = [Bool("y_0"),Bool("y_1"),Bool("y_2")]
    res = [Bool("res_0")]

    s = Solver()
    s.add(x[0] == True)
    s.add(y[0] == True)

    #s.add(x[1] == False)
    s.add(y[1] == False)

    #s.add(y[2] == False)
   
    
    s.add(geq(y,x))

    if s.check() == sat:
        print(s.model())
    else:
        print(s.check())
    """

    

        
