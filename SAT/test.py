import json
import os
from utils_sat import *
import time
from z3 import *
import numpy as np

#solver = Solver()

#m = 2
#n = 6
#path = [[[Bool(f"p_{courier}_{package}_{step}") for step in range(n+2)] for package in range(n+1)] for courier in range(m)]
def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(matrix[i][j], end = " ")
        print()
def print_matrix_val(matrix,model):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(model[matrix[i][j]], end = " ")
        print()

n = 3
m = 1
D = [[0,3,7,6],
        [3,0,5,5],
        [2,8,0,3],
        [1,2,7,0]]


#max_load = 100

####max_dist 
matrix_D = np.array(D)
flat = matrix_D.flatten()
flat.sort()
flat = flat[::-1]
max_dist = sum([flat[i] for i in range(n)])


s = Solver()
#courier_loads = [[Bool(f"cl_{courier}_{bit}") for bit in range(num_bits(max_load))] for courier in range(m)]
D_max = np.matrix(D).max()
D_b = [[int_to_binary(D[i][j], num_bits(D_max)) for j in range(n+1)] for i in range(n+1)]
    # path[i][j][k] = T if the courier i delivers the package j at the k-th step 
path = [[[Bool(f"p_{courier}_{package}_{step}") for step in range(n+2)] for package in range(n+1)] for courier in range(m)]

c_dist_tot = [[Bool(f"cdt_{courier}_{bit}") for bit in range(num_bits(max_dist))] for courier in range(m)]

#  partial_dist 
c_dist_par = [[[Bool(f"cpt_{courier}_{step}_{bit}") for bit in range(num_bits(D_max))] for step in range(n+1)] for courier in range(m)]
    
#max var
max_dist_b = [Bool(f"max_d_{bit}") for bit in range(num_bits(max_dist))]

if __name__ == "__main__":

    for courier in range(m):
        for step in range(n+1):
            for package_start in range(n+1):
                for package_end in range(n+1):
                    s.add(Implies(And(path[courier][package_start][step], path[courier][package_end][step+1]), 
                                  eq_bin(D_b[package_start][package_end],c_dist_par[courier][step])))
    
    for i in range(m):
        s.add(cond_sum_bin(c_dist_par[i], [BoolVal(True) for _ in range(n+1)], c_dist_tot[i], f"def_courier_dist_{i}"))
    
    s.add(max_var(c_dist_tot, max_dist_b))
    print_matrix(c_dist_tot)
    print_matrix(path[0])
    print_matrix(c_dist_par[0])
    print("PASS")
    s.add(path[0][n][0] == True)
    s.add(path[0][0][0] == False)
    s.add(path[0][1][0] == False)
    s.add(path[0][2][0] == False)
    print("PASS")

    s.add(path[0][n][1] == False)
    s.add(path[0][2][1] == False)
    s.add(path[0][1][1] == False)
    s.add(path[0][0][1] == True)

    s.add(path[0][1][2] == False)
    s.add(path[0][0][2] == True)
    s.add(path[0][2][2] == False)#s.add(path[0][2][2] == True)
    s.add(path[0][n][2] == False)

    s.add(path[0][1][3] == True)
    s.add(path[0][0][3] == False)
    s.add(path[0][2][3] == False)
    s.add(path[0][n][3] == False)

    s.add(path[0][0][4] == False)
    s.add(path[0][1][4] == False)
    s.add(path[0][2][4] == False)
    s.add(path[0][n][4] == True)
    print(s.check())
    
    #print_matrix_val(path[0],s.model())
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
            
            