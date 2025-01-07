import json
import os

from prettytable import PrettyTable
from utils_sat import *
import time
from z3 import *
import numpy as np
import multiprocessing as mp

def save_solution(name_sol, output_directory, output_file, data):
    solution = {
        name_sol: {
            "time": data[0],  
            "optimal": data[1],  
            "obj": data[2], 
            "sol": data[3] 
        }
    }

    output_path_file = output_directory + "/" + output_file + ".json"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    data = {}
    if os.path.isfile(output_path_file):
        with open(output_path_file, 'r') as file:
            data = json.load(file)
            
    data.update(solution)
    
    # write the solution to the output file
    with open(output_path_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(matrix[i][j], end = " ")
        print()

def mcp_sat(m, n, l, s, D, shared_res, symm_constr = False, search = "linear"):

    #Start to record execution time
    start_time = time.time()

    solver = Solver()

    deposit = n #Deposit index (if we count from 0)
    max_load = sum(s) #max theoretical load for one courier
    D_max = np.matrix(D).max() #Max distance on matrix

    #Maximum theoretical distance that a courier can run 
    matrix_D = np.array(D)
    flat = matrix_D.flatten()
    flat.sort()
    flat = flat[::-1]
    max_dist = sum([flat[i] for i in range(n)]) #Sum the gretest distances on the matrix
    
    #-----VARIABLES-----

    # path[i][j][k] = T if the courier i delivers the package j at the k-th step 
    path = [[[Bool(f"p_{courier}_{package}_{step}") for step in range(n+2)] for package in range(n+1)] for courier in range(m)]
    
    # courier_weights[i][j] = T if the courier i take the package  j
    courier_weights = [[Bool(f"w_{courier}_{package}") for package in range(n)] for courier in range(m)]

    # courier_loads_i = it represents binary representation of actual load carried by each courier
    courier_loads = [[Bool(f"cl_{courier}_{bit}") for bit in range(num_bits(max_load))] for courier in range(m)]

    # courier_dists_i = it represents binary representation of actual dist by each courier
    c_dist_tot = [[Bool(f"cdt_{courier}_{bit}") for bit in range(num_bits(max_dist))] for courier in range(m)]

    # partial_dist 
    c_dist_par = [[[Bool(f"cpt_{courier}_{step}_{bit}") for bit in range(num_bits(D_max))] for step in range(n+1)] for courier in range(m)]
    
    # Binary max variable representation (it is used for objective function)
    max_dist_b = [Bool(f"max_d_{bit}") for bit in range(num_bits(max_dist))]

    #Binary conversion of list s
    s_max = max(s)
    s_b = [int_to_binary(s_value, num_bits(s_max)) for s_value in s]

    #Binary conversion of list l
    l_max = max(l)
    l_b = [int_to_binary(l_value, num_bits(l_max)) for l_value in l]

    #Binary conversion of matrix D
    D_b = [[int_to_binary(D[i][j], num_bits(D_max)) for j in range(n+1)] for i in range(n+1)]

    
    #-----CONSTRAINTS-----

    #Fundamental constraints

    #Binding the weight and path
    for courier in range(m):
        for step in range(n+2): #it could be optimized due to the fact that the first and last step is always n (deposit)
            for package in range(n): #we don't consider the depoist
                solver.add(Implies(path[courier][package][step], courier_weights[courier][package]))

    #C1 the courier delivers exactly one package at each step (and impose that courier visit something - at least the deposit)
    for courier in range(m):
        for step in range(n+2):
            solver.add(exactly_one_he([path[courier][package][step] for package in range(n+1)], f"one_p_s_{courier}_{step}"))
    
    #C2: Each package is carried only once 
    for package in range(n): #not consider n+1 (deposit)
        solver.add(exactly_one_he([path[courier][package][step] for courier in range(m) for step in range(n+2)], f"one_t_{package}"))
    
    #The constraint is also replicated for courier_weights
    for package in range(n):
        solver.add(exactly_one_he([courier_weights[courier][package] for courier in range(m)], f"one_t_w_{package}"))
    
    #C3: Couriers start and end at the deposit
    for courier in range(m):
        solver.add(path[courier][deposit][0] == True)
        solver.add(path[courier][deposit][n+1] == True) #n+1 means n+2 because we count from zero

    #C4: Every courier has a maximum load capacity to respect
    for courier in range(m):
        solver.add(cond_sum_bin(s_b, courier_weights[courier], courier_loads[courier], f"courier_load_{courier}")) #Make the sum for each courier
        solver.add(geq(l_b[courier],courier_loads[courier])) #Impose the maximum load for each courier
    
    #C5: All couriers must start as soon as possible
        #1 is the first step
        #range(n) because they have to pick one package and not choose the deposit (n+1)
    for courier in range(m):
        solver.add(at_least_one_he([path[courier][package][1] for package in range(n)]))
    
    #C6: if a courier doesn't take the a pack at position j, also at position j+1 doesn't take any pack
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
    
    #Get the max distance
    solver.add(max_var(c_dist_tot, max_dist_b))
    
    #Symmetric breaking constraint
    if symm_constr == True:
        l_sorted = [(l[i],i) for i in range(len(l))]
        l_sorted.sort(key = lambda x: x[0], reverse= True)

        for i in range(m-1):
            if l_sorted[i][0] == l_sorted[i+1][0]:

                #S1 if the load of two couriers are equal then define the lexicographical order
                solver.add(lex_leq(courier_weights[l_sorted[i][1]], courier_weights[l_sorted[i+1][1]]))

            else:
                #S2 if the load of the first courier is greater than the load of the second courier then impose an order
                solver.add(geq(courier_loads[l_sorted[i][1]],courier_loads[l_sorted[i+1][1]]))
                
        

    if search == "linear":
        print("START LINEAR")
        satisfiable = True
        last_model_sat = None

        #First upper bound
        upper_bound = D[deposit][0] + sum([D[i][i+1] for i in range(len(D[0])-1)])
        print("UPPER_BOUND",upper_bound)
        while satisfiable:
            #print("Linear UB",upper_bound)
            #Convert upper bound
            upper_bound_b = int_to_binary(upper_bound, num_bits(upper_bound))
            
            solver.push()

            #Add constraint 
            solver.add(greater(upper_bound_b,max_dist_b))

            #Check satisfiability
            if solver.check() == sat:
                
                model = solver.model()
                last_model_sat = model
                upper_bound =  binary_to_int([model[val_bin] for val_bin in max_dist_b])

                #Save result on shared variable
                path_res, obj_value = process_model(model, path, max_dist_b,m,n)
                #print("PATH",path_res)
                shared_res["res"] = [shared_res["res"][0], shared_res["res"][1], obj_value, path_res]
                #print("DATA Linear mid SAT:", shared_res["res"])
                               
            else:
                #Save result
                if shared_res["res"][2] != None and shared_res["res"][3] != []:
                    shared_res["res"] = [int(time.time() - start_time), True, shared_res["res"][2], shared_res["res"][3]]
                    #print("DAT Linear mid UNSAT:",shared_res["res"])
                satisfiable = False

                model = last_model_sat
                #return last_model_sat, path, max_dist_b
            solver.pop()
            print("END LINEAR")

    if search == "binary":
        satisfiable = True
        last_model_sat = None

        upper_bound = D[deposit][0] + sum([D[i][i+1] for i in range(len(D[0])-1)])
        lower_bound = max([D[deposit][i] + D[i][deposit] for i in range(n-1)])

        while satisfiable:
            #print("LB",lower_bound, " UB",upper_bound, end = " ")
            if (upper_bound - lower_bound) <= 1:
                satisfiable = False
            if (upper_bound - lower_bound) == 1:
                middle_bound = lower_bound
            else:
                middle_bound = (upper_bound + lower_bound) // 2
            #print("MB",middle_bound, end = " ")
            
            upper_bound_b = int_to_binary(upper_bound, num_bits(upper_bound))
            #lower_bound_b = int_to_binary(lower_bound, num_bits(lower_bound))
            
            middle_bound_b = int_to_binary(middle_bound, num_bits(middle_bound))
            
            solver.push()

            solver.add(geq(middle_bound_b,max_dist_b))

            if solver.check() == sat:
                model = solver.model()
                last_model_sat = model
                upper_bound = binary_to_int([model[val_bin] for val_bin in max_dist_b])
                #print(" sat")

                #Save result on shared variable
                path_res, obj_value = process_model(model, path, max_dist_b, m, n)
                shared_res["res"] = [shared_res["res"][0], shared_res["res"][1], obj_value, path_res]

            elif (solver.check() == unsat) or (solver.check() == unknown):
                #model = solver.model()
                lower_bound = middle_bound
                #print(" unsat")

            solver.pop()

        if shared_res["res"][2] != None and shared_res["res"][3] != []:
            shared_res["res"] = [int(time.time() - start_time), True, shared_res["res"][2], shared_res["res"][3]]
            print("DAT Binary mid UNSAT:",shared_res["res"])

   #return solver, path, courier_weights, courier_loads, c_dist_par, c_dist_tot,max_dist_b

def process_model(model, path_b, max_dist_b, m, n):
    res_path = []
    for courier in range(m):
        courier_packages = []
        for j in range(len(path_b[courier][0])):
            for i in range(len(path_b[courier])):
                if model[path_b[courier][i][j]] == True and i!=n:
                    courier_packages.append(i+1)
        res_path.append(courier_packages)
        obj_value =  binary_to_int([model[val_bin] for val_bin in max_dist_b])    

    return res_path,obj_value


def print_matrix(matrix, model, title = "--------"):
    print(title)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if model[matrix[i][j]] == True:
                print('\033[92m' + f"{matrix[i][j]}:1" + '\033[0m', end = " ")
            else:
                print(f"{matrix[i][j]}:0", end = " ")
        print()

def get_name_test(search_strategy,symm_break_constr):
    res_name = ""
    if search_strategy == "linear":
        res_name = res_name + "LNS"

    if search_strategy == "binary":
        res_name = res_name + "BNS"
    
    if symm_break_constr == True:
        res_name = res_name + "_SYB"
    
    return res_name

def check_solution(time_exe, opt, obj, path, time_execution):
    if opt == False:
        return time_execution, opt, obj, path
    return time_exe, opt, obj, path

def solve_problem(m, n, l, s, D,  symm_constr = False, search = "linear", time_execution = 300):

    with mp.Manager() as manager:

        #Define the shared variable 
        shared_res = manager.dict()
        shared_res["res"] = [time_execution, False, None, []]#[300, opt, obj_value, path]
        print("START solve problem with data:",shared_res)

        #Define and run the process to solve the problem 
        mcp_sat_process = mp.Process(target = mcp_sat, args=(m, n, l, s, D, shared_res, symm_constr, search))
        mcp_sat_process.start()

        #Wait until the time is finished
        time.sleep(time_execution)

        #Terminate/Stop (similar to kill) the process that solves the problem
        mcp_sat_process.terminate()
        
        print("END solve problem with data:",shared_res)
        time_exe, opt, obj, path = shared_res["res"]

    return time_exe, opt, obj, path

if __name__ == "__main__":
    first_instance = 1
    last_instance = 21
    file_name_save = 'result_model.txt'
    file_name_error = 'error_model.txt'
    mode_file_result = 'w'
    mode_file_error = "a"
    output_directory = "../res/SAT"
    mp.set_start_method("spawn")

    configs = [["linear", True],
              ["linear", False],
              ["binary", True],
              ["binary", False]]
    #configs = [["linear",False]]
    for config in configs:
        # Hyperparameter
        search_strategy = config[0]
        symm_break_constr = config[1]
        time_execution = 300


        for i in range(first_instance, last_instance+1):
            file_path = f'instances/inst{i:02d}.dat'
            inst_i = f"inst{i:02d}" 
            data_path = f"../instances_dzn/{inst_i}.dzn"

            #Read the data
            m, n, l, s, D = read_instance(data_path)

            #Call the function to set and solve the problem 
            data = solve_problem(m, n, l, s, D, symm_constr = symm_break_constr, search = search_strategy, time_execution = time_execution)
            print("Final data",data)

            #Define a correct name label for the test 
            sol_name = get_name_test(search_strategy,symm_break_constr)

            #Save the solution
            save_solution(sol_name, output_directory, f"{i:02d}",data)

        
