from itertools import combinations
from z3 import *
import time
import math
import re
import utils_sat
import numpy as np
from utils_sat import *


def parse_value(value):
    return value.strip().strip(';')
    
def read_instance(file_path):
    # Initialize data structures
    m = None
    n = None
    l = []
    s = []
    distances = []
    data = []
    # Read the data file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        m = int(parse_value(lines[0].split('=')[1]))
        n = int(parse_value(lines[1].split('=')[1]))
        
        for c in re.findall(r'\b\d+\b', parse_value(lines[2].split('=')[1])):
            l.append(int(c))
        
        for c in re.findall(r'\b\d+\b', parse_value(lines[3].split('=')[1])):
            s.append(int(c))
    
        for line in lines[4:]:
            row = []
            #print(line.split(','))
            for el in line.split(','):
                r =re.findall(r'\b\d+\b', el)
                for c in r: 
                    row.append(int(c))
            distances.append(row)
    return m, n, l, s, distances

def problem(m, n, l, s, D):
        
        matrix_D = np.array(D)
        flat = matrix_D.flatten()
        flat.sort()
        flat = flat[::-1]
        max_dist = sum([flat[i] for i in range(n)])
        max_load = sum(s)   

        path = [[[Bool(f"p_{courier}_{package}_{step}") for step in range(n+2)] for package in range(n+1)] for courier in range(m)]
        
        s_max = max(s)
        s_b = [int_to_binary(s_value, num_bits(s_max)) for s_value in s]

        l_max = max(l)
        l_b = [int_to_binary(l_value, num_bits(l_max)) for l_value in l]
        
        
        courier_weights = [[Bool(f"w_{courier}_{package}")for package in range(n)] for courier in range(m)]
        # courier_loads_i = it represents binary representation of actual load carried by each courier
        courier_loads = [[Bool(f"cl_{courier}_{bit}") for bit in range(num_bits(max_load))] for courier in range(m)]



        solver = Solver()
        for courier in range(m):
            print("COURIER_WEIGHT", courier_weights[courier])
            solver.add(cond_sum_bin(s_b, courier_weights[courier], courier_loads[courier], f"def_courier_load_{courier}"))
            #solver.add(geq(l_b[courier],courier_loads[courier]))
        solver.add( courier_weights[0][0] == True )
        solver.add( courier_weights[0][1] == True )
        solver.add( courier_weights[0][2] == True )
        solver.add( courier_weights[0][3] == True )
        solver.add( courier_weights[0][4] == True )
        solver.add( courier_weights[1][4] == False )
        return solver, courier_loads, courier_weights, s_b, l_b

def print_matrix(matrix, model, title = "--------"):
    print(title)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if model[matrix[i][j]] == True:
                print('\033[92m' + f"{matrix[i][j]}:1" + '\033[0m', end = " ")
            else:
                print(f"{matrix[i][j]}:0", end = " ")
        print()


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
        s,courier_loads, courier_weights, s_b,l_b = problem(m, n, l, s, D )

        if s.check() == sat:
            m = s.model()
            print(s.check())
            print_matrix(courier_loads, m, "------courier_loads-----")
            print_matrix(courier_weights, m, "------courier_weights-----")
            print("------s_b-----", s_b)
            print("------l_b-----", l_b)
            
        else:
            print(s.check())
        
    


        
        
        