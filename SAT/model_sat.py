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


def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(matrix[i][j], end = " ")
        print()

def mcp_sat(m, n, l, s, D, simm_constr = False, search = "linear"):
    start_time = time.time()

    solver = Solver()

    deposit = n #if we count from 0
    max_load = sum(s)
    D_max = np.matrix(D).max()

    print("deposit",n)
    print("max_load",max_load)
    print("D_max",D_max)

    max_dist = D[deposit][0] + sum([D[i][i+1] for i in range(len(D[0])-1)])

    print("max_dist",max_dist)
    #VARIABLES

    # path[i][j][k] = T if the courier i delivers the package j at the k-th step 
    path = [[[Bool(f"p_{courier}_{package}_{step}") for step in range(n+2)] for package in range(n+1)] for courier in range(m)]
    
    #courier_stops
    courier_stops = [[[Bool(f"s_{courier}_{package1}_{package2}") for package1 in range(n+1)] for package2 in range(n+1)] for courier in range(m)]
    
    # courier_weights[i][j] = T if the courier i take the package  j
    courier_weights = [[Bool(f"w_{courier}_{package}")for package in range(n)] for courier in range(m)]

    # courier_loads_i = it represents binary representation of actual load carried by each courier
    courier_loads = [[Bool(f"cl_{courier}_{bit}") for bit in range(num_bits(max_load))] for courier in range(m)]

    # courier_dists_i = it represents binary representation of actual dist by each courier
    c_dist_tot = [[Bool(f"cdt_{courier}_{bit}") for bit in range(num_bits(max_dist))] for courier in range(m)]

    #  partial_dist 
    c_dist_par = [[[Bool(f"cpt_{courier}_{step}_{bit}") for bit in range(num_bits(D_max))] for step in range(n+1)] for courier in range(m)]
    
    #max var
    max_dist_b = [Bool(f"max_d_{bit}") for bit in range(num_bits(max_dist))]

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
                solver.add(Implies(path[courier][package][step], courier_weights[courier][package]))

    #1: the courier delivers exactly one package at each step
    for courier in range(m):
        for step in range(n+2):
            solver.add(exactly_one_he([path[courier][package][step] for package in range(n+1)], f"one_p_s_{m}_{step}"))
    
    #2: Each package is carried only once
    for package in range(n): #not consider n+1 (deposit)
        #s.add(exactly_one_he(path[courier][package][:]))
        solver.add(exactly_one_he([path[courier][package][step] for courier in range(m) for step in range(n+2)], f"one_t_{package}"))

    #3: Couriers start and end at the deposit
    for courier in range(m):
        solver.add(path[courier][deposit][0] == True)
        solver.add(path[courier][deposit][n+1] == True) #n+1 means n+2 because we count from zero

    #4: Every courier has a maximum load capacity to respect
    for i in range(m):
        solver.add(cond_sum_bin(s_b, courier_weights[i], courier_loads[i], f"def_courier_load_{i}"))
        solver.add(geq(l_b[i],courier_loads[i]))
    
    
    #5: All couriers must start as soon as possible
    #1 is the first step
    #range(n) because they have to pick one package and not choose the deposit (n+1)
    for courier in range(m):
        solver.add(at_least_one_he([path[courier][i][1] for i in range(n)]))

    #6: if a courier doesn't take the a pack at position j, also at position j+1 doesn't take any pack
    # So if a courier is in the deposit at step 1 (it starts at 0) it means that he will not deliver any pack
    # it also means that the courier can come back to the deposit if he has to deliver other packagages
    for courier in range(m):
        for step in range(1,n):
            solver.add(Implies(path[courier][deposit][step], path[courier][deposit][step+1]))
    
    #Objective function
    for courier in range(m):
        for step in range(n+1):
            for package_start in range(n+1):
                for package_end in range(n+1):
                    solver.add(Implies(And(path[courier][package_start][step], path[courier][package_end][step+1]), 
                                  eq_bin(D_b[package_start][package_end],c_dist_par[courier][step])))
    
    for i in range(m):
        solver.add(cond_sum_bin(c_dist_par[i], [BoolVal(True) for _ in range(n+1)], c_dist_tot[i], f"def_courier_dist_{i}"))
    
    solver.add(max_var(c_dist_tot, max_dist_b))
    
    upper_bound = 500
    upper_bound_b = int_to_binary(upper_bound, num_bits(upper_bound))

    solver.add(geq(upper_bound_b,max_dist_b))
    """
    for courier in range(m):
        for step in range(n+1):
            for package_start in range(n+1):
                for package_end in range(n+1):
                    s.add(Implies(And(path[courier][package_start][step], path[courier][package_end][step+1]), 
                                  courier_stops[courier][package_start][package_end]) )
    
    for courier in range(m):
        for package_start in range(n+1):
            for package_end in range(n+1):
                s.add(Implies(courier_stops[courier][package_start][package_end], eq_bin(D_b[package_start][package_end])))
    """
    
    #Objective function
  
    
    if simm_constr == True:
        pass
    if search == "linear":
        pass
    end_time = time.time()
    
    return solver

if __name__ == "__main__":
    first_instance = 1
    last_instance = 1
    file_name_save = 'result_model.txt'
    file_name_error = 'error_model.txt'
    mode_file_result = 'w'
    mode_file_error = "a"

    for i in range(first_instance, last_instance+1):
        file_path = f'instances/inst{i:02d}.dat'
        inst_i = f"inst{i:02d}" 
        data_path = f"../instances_dzn/{inst_i}.dzn"
        m, n, l, s, D = read_instance(data_path)
        print(m, n, l, s, D)
        s = mcp_sat(m, n, l, s, D)

        if s.check() == sat:
            m = s.model()
        else:
            print(s.check())

    """
    n = 3
    m = 1
    D = [[0,3,7,6],
         [3,0,5,5],
         [2,8,0,3],
         [1,2,7,0]]
    
    
    max_load = 100
    max_dist =  D[n][0] + sum([D[i][i+1] for i in range(len(D[0])-1)])
    s = Solver()
    #courier_loads = [[Bool(f"cl_{courier}_{bit}") for bit in range(num_bits(max_load))] for courier in range(m)]
    D_max = np.matrix(D).max()
    
    D_b = [[int_to_binary(D[i][j], num_bits(D_max)) for j in range(n+1)] for i in range(n+1)]
    path = [[[Bool(f"p_{courier}_{package}_{step}") for step in range(n+2)] for package in range(n+1)] for courier in range(m)]
    
    c_dist_tot = [[Bool(f"cd_{courier}_{bit}") for bit in range(num_bits(max_dist))] for courier in range(m)]
    
    #  partial_dist 
    c_dist_par = [[[Bool(f"cp_{courier}_{step}_{bit}") for bit in range(num_bits(D_max))] for step in range(n+1)] for courier in range(m)]

    #max var
    max_dist_b = [Bool(f"max_d_{bit}") for bit in range(num_bits(max_dist))]

    """
    """
    for courier in range(m):
        for step in range(n+1):
            for package_start in range(n+1):
                for package_end in range(n+1):
                    s.add(Implies(And(path[courier][package_start][step], path[courier][package_end][step+1]), 
                                  eq_bin(D_b[package_start][package_end],c_dist_par[courier][step])))
    
    for i in range(m):
        s.add(cond_sum_bin(c_dist_par[i], [BoolVal(True) for _ in range(n+1)], c_dist_tot[i], f"def_courier_load_{i}"))
    
    s.add(max_var(c_dist_tot, max_dist_b))
    """
    """
    for i in range(n+1):
        for j in range(n+2):
            print(path[0][i][j], end = " ")
        print()
    """
    """
    s.add(path[0][n][0] == True)
    s.add(path[0][0][0] == False)
    s.add(path[0][1][0] == False)
    s.add(path[0][2][0] == False)

    s.add(path[0][n][1] == False)
    s.add(path[0][2][1] == False)
    s.add(path[0][1][1] == False)
    s.add(path[0][0][1] == True)

    s.add(path[0][1][2] == False)
    s.add(path[0][0][2] == False)
    s.add(path[0][2][2] == True)
    s.add(path[0][n][2] == False)

    s.add(path[0][1][3] == True)
    s.add(path[0][0][3] == False)
    s.add(path[0][2][3] == False)
    s.add(path[0][n][3] == False)

    s.add(path[0][0][4] == False)
    s.add(path[0][1][4] == False)
    s.add(path[0][2][4] == False)
    s.add(path[0][n][4] == True)
    """

    """
    print(c_dist_par[0][0])
    print(c_dist_par[0][1])
    print(c_dist_par[0][2])
    print(c_dist_par[0][3])
    """ 
    """
    s.add(path[0][n][0] == True)
    s.add( eq_bin(D_b[package_start][package_end],c_dist_par[0][package_start]))
    s.add( eq_bin(D_b[package_start][package_end],c_dist_par[0][package_start]))
    s.add( eq_bin(D_b[package_start][package_end],c_dist_par[0][package_start]))
    s.add( eq_bin(D_b[package_start][package_end],c_dist_par[0][package_start]))
    """
 

    """
    res = [Bool("x1"), Bool("x2"), Bool("x3")]
    
    
    d_temp = [[BoolVal(False), BoolVal(False), BoolVal(True)],
              [BoolVal(False), BoolVal(False), BoolVal(True)]]
    
    var_temp =[BoolVal(False), BoolVal(True)]
    
    s.add(Implies(var_temp[0],eq_bin(d_temp[0],res)))
    s.add(Implies(var_temp[1],eq_bin(d_temp[1],res)))
    """
    
    #res_temp =  [[BoolVal(False) for i in range(len(res1))]] + [[Bool(f"res_t_{i}_{j}") for j in range(len(res))] for i in range(len(elems))]

    
    # path = [[[Bool(f"p_{courier}_{package}_{step}") for step in range(n+2)] for package in range(n+1)] for courier in range(m)]
    #x = [Bool(f"x1"),Bool(f"x1")]

    """
    #TEST SUM
    x = [[Bool(f"x_{i}_{j}") for j in range(3)] for i in range(3)]
    mask = [Bool(f"m_{i}") for i in range(3)]
    res = [Bool(f"r_{i}") for i in range(3)]
    s.add(mask[0]  == False)
    s.add(mask[1]  == False)
    s.add(mask[2]  == False)

    s.add(x[0][0]  == False)
    s.add(x[0][1]  == False)
    s.add(x[0][2]  == True)
    
    
    s.add(x[1][0]  == False)
    s.add(x[1][1]  == True)
    s.add(x[1][2]  == True)
    
    
    s.add(x[2][0]  == False)
    s.add(x[2][1]  == False)
    s.add(x[2][2]  == True)
    
    s.add(cond_sum_bin(x,mask, res, "sum1"))
    """
    #s.add(full_adder(x[0], x[1], res, name = ""))
    #sum_bin(x, y, res, name= "", mask = True):
    #s.add(sum_bin(x[0], x[1], res, name = ""))
    #s.add(conditiona_sum_K_bin(mask, x, res, name = ""))
    """
    if s.check() == sat:
        m = s.model()
        
        
        for list_var in c_dist_par[0]:  
            for var in list_var:
            #f str(res) == "r_1" or str(res) == "r_0" or str(res) == "r_2":
                #print("Var: ", var, "Value:", m[var], end = " ")
                print(m[var], end = " ")
            print()
        print("---")
        for bit in max_dist_b:
            print(m[bit], end = " ")
        
        
        
    else:
        print(s.check())
    """
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

    

        
