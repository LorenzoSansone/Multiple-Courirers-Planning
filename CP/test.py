from prettytable import PrettyTable

import time

import json
import os

from prettytable import PrettyTable
from utils_sat import *
import time
from z3 import *


def print_matrix(matrix, model, title = "--------"):
    print(title)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if model[matrix[i][j]] == True:
                print('\033[92m' + f"{matrix[i][j]}:1" + '\033[0m', end = " ")
            else:
                print(f"{matrix[i][j]}:0", end = " ")
        print()

def load_data(output_directory, output_file):
    output_path_file = output_directory + "/" + output_file + ".json"
    data = {}
    if os.path.isfile(output_path_file):
        with open(output_path_file, 'r') as file:
            data = json.load(file)
    return data
        
if __name__ == "__main__":
    output_directory = "../res/CP"
    first_instance = 1
    last_instance = 21   
    tableRes = PrettyTable(["I", "BSG", "BS_HEUG", "BS_HEU_IMPLG", "BS_HEU_SymG","BS_HEU_Sym_ImplG",
                                    "BSC", "BS_HEUC", "BS_HEU_IMPLC", "HueSYMCh","HueSymImpCh"])

    for i in [x for x in range(first_instance, last_instance+1)]:
        data = load_data(output_directory,f"{i:02d}")
        print(i)
        if data["BS_Gec"]["optimal"] == True:
            str_BS = '\033[92m' + f"{data["BS_Gec"]["obj"]}" + '\033[0m'
        else:
            str_BS = f"{data["BS_Gec"]["obj"]}"

        if data["BS_HEU_Gec"]["optimal"] == True:
            str_BS_HEU = '\033[92m' + f"{data["BS_HEU_Gec"]["obj"]}" + '\033[0m'
        else:
            str_BS_HEU = f"{data["BS_HEU_Gec"]["obj"]}"

        if data["BS_HEU_IMPL_Gec"]["optimal"] == True:
            str_BS_HEU_IMPL = '\033[92m' + f"{data["BS_HEU_IMPL_Gec"]["obj"]}" + '\033[0m'
        else:
            str_BS_HEU_IMPL = f"{data["BS_HEU_IMPL_Gec"]["obj"]}"

        if data["BS_HEU_SYM_Gec"]["optimal"] == True:
            str_BS_HEU_SYM = '\033[92m' + f"{data["BS_HEU_SYM_Gec"]["obj"]}" + '\033[0m'
        else:
            str_BS_HEU_SYM = f"{data["BS_HEU_SYM_Gec"]["obj"]}"    

        if data["BS_HEU_SYM_IMPL_Gec"]["optimal"] == True:
            str_BS_HEU_SYM_IMPL = '\033[92m' + f"{data["BS_HEU_SYM_IMPL_Gec"]["obj"]}" + '\033[0m'
        else:
            str_BS_HEU_SYM_IMPL = f"{data["BS_HEU_SYM_IMPL_Gec"]["obj"]}"    
        ##################
        if data["BS_Chuf"]["optimal"] == True:
            str_BS_Chuf = '\033[92m' + f"{data["BS_Chuf"]["obj"]}" + '\033[0m'
        else:
            str_BS_Chuf = f"{data["BS_Chuf"]["obj"]}"

        if data["BS_HEU_Chuf"]["optimal"] == True:
            str_BS_HEU_Chuf = '\033[92m' + f"{data["BS_HEU_Chuf"]["obj"]}" + '\033[0m'
        else:
            str_BS_HEU_Chuf = f"{data["BS_HEU_Chuf"]["obj"]}"

        if data["BS_HEU_IMPL_Chuf"]["optimal"] == True:
            str_BS_HEU_IMPL_Chuf = '\033[92m' + f"{data["BS_HEU_IMPL_Chuf"]["obj"]}" + '\033[0m'
        else:
            str_BS_HEU_IMPL_Chuf = f"{data["BS_HEU_IMPL_Chuf"]["obj"]}"

        if data["BS_HEU_SYM_Chuf"]["optimal"] == True:
            str_BS_HEU_SYM_Chuf = '\033[92m' + f"{data["BS_HEU_SYM_Chuf"]["obj"]}" + '\033[0m'
        else:
            str_BS_HEU_SYM_Chuf = f"{data["BS_HEU_SYM_Chuf"]["obj"]}"    

        if data["BS_HEU_SYM_IMPL_Chuf"]["optimal"] == True:
            str_BS_HEU_SYM_IMPL_Chuf = '\033[92m' + f"{data["BS_HEU_SYM_IMPL_Chuf"]["obj"]}" + '\033[0m'
        else:
            str_BS_HEU_SYM_IMPL_Chuf = f"{data["BS_HEU_SYM_IMPL_Chuf"]["obj"]}" 
      
        tableRes.add_row([f"{i:02d}",
                          str_BS,
                          str_BS_HEU,
                          str_BS_HEU_IMPL,
                          str_BS_HEU_SYM,
                          str_BS_HEU_SYM_IMPL,
                          str_BS_Chuf,
                          str_BS_HEU_Chuf,
                          str_BS_HEU_IMPL_Chuf,
                          str_BS_HEU_SYM_Chuf,
                          str_BS_HEU_SYM_IMPL_Chuf])
    print(tableRes)
    
    file = open("file_prove.txt", "w")  # append mode
    file.write(str(tableRes))
    file.close()

    print()
    #tableRes.add_row
    #l = [190, 185, 185, 190, 195, 185]
    """
    l = [190,190]
    n = 3
    m = len(l)
    max_load = 200
    courier_loads = [[Bool(f"cl_{courier}_{bit}") for bit in range(num_bits(max_load))] for courier in range(m)]
    courier_weights = [[Bool(f"w_{courier}_{package}") for package in range(n)] for courier in range(m)]

    solver = Solver()

    solver.add(courier_weights[0][0] == True)
    solver.add(courier_weights[0][1] == False)
    solver.add(courier_weights[0][2] == False)

    solver.add(courier_weights[1][0] == True)
    solver.add(courier_weights[1][1] == True)
    solver.add(courier_weights[1][2] == False)

    solver.add(courier_loads[0][0] == False)
    solver.add(courier_loads[0][1] == False)
    solver.add(courier_loads[0][2] == False)
    solver.add(courier_loads[0][3] == False)
    solver.add(courier_loads[0][4] == False)
    solver.add(courier_loads[0][5] == False)
    solver.add(courier_loads[0][6] == False)
    solver.add(courier_loads[0][7] == False)

    solver.add(courier_loads[1][0] == False)
    solver.add(courier_loads[1][1] == True)
    solver.add(courier_loads[1][2] == False)
    solver.add(courier_loads[1][3] == False)
    solver.add(courier_loads[1][4] == False)
    solver.add(courier_loads[1][5] == False)
    solver.add(courier_loads[1][6] == False)
    solver.add(courier_loads[1][7] == False)

    l_sorted = [(l[i],i) for i in range(len(l))]
    l_sorted.sort(key = lambda x: x[0], reverse= True)
    print(l)
    print(l_sorted)
    
    for i in range(m-1):
        if l_sorted[i][0] == l_sorted[i+1][0]:
            print(l_sorted[i][0],"==",l_sorted[i+1][0]," ",l_sorted[i][1],"-",l_sorted[i+1][1])
            
            solver.add(lex_leq(courier_weights[l_sorted[i][1]], courier_weights[l_sorted[i+1][1]]))
                
        else:
                print(l_sorted[i][0],">=",l_sorted[i+1][0]," ",l_sorted[i][1],"-",l_sorted[i+1][1])
                solver.add(geq(courier_loads[l_sorted[i][1]],courier_loads[l_sorted[i+1][1]]))
    if solver.check() == sat:
        print_matrix(courier_weights,solver.model())
        print_matrix(courier_loads,solver.model())
    if solver.check() == unsat:
        print("UNSAT")
    """
    
