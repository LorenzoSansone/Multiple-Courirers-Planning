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
        
        path = [[[Bool(f"p_{courier}_{package}_{step}") for step in range(n+2)] for package in range(n+1)] for courier in range(m)]

        solver = Solver()
        for courier in range(m):
            for step in range(n+2):
                
                solver.add(exactly_one_seq([path[courier][package][step] for package in range(n+1)], f"one_p_s_{m}_{step}"))
        
        print([path[0][package][0] for package in range(n+1)])
        solver.add(path[0][0][0] == True)
        solver.add(path[1][3][0] == True)
        
        return solver, path

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
        s,path = problem(m, n, l, s, D )

        if s.check() == sat:
            m = s.model()
            print(s.check())
            print_matrix(path[0], m, "------PATH-----")
            print_matrix(path[1], m, "------PATH-----")
        else:
            print(s.check())
        
    


        
        
        