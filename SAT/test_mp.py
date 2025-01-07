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
    output_directory = "../res/SAT"
    first_instance = 1
    last_instance = 21   
    tableRes = PrettyTable(["Inst", "LNS_SYB", "LNS", "BNS_SYB", "BNS"])

    for i in range(first_instance,last_instance +1):
        data = load_data(output_directory,f"{i:02d}")
        print(data)
        if data["LNS_SYB"]["optimal"] == True:
            str_LNS_SYB = '\033[92m' + f"{data["LNS_SYB"]["obj"]}" + '\033[0m'
        else:
            str_LNS_SYB = f"{data["LNS_SYB"]["obj"]}"

        if data["LNS"]["optimal"] == True:
            str_LNS = '\033[92m' + f"{data["LNS"]["obj"]}" + '\033[0m'
        else:
            str_LNS = f"{data["LNS"]["obj"]}"

        if data["BNS_SYB"]["optimal"] == True:
            str_BNS_SYB = '\033[92m' + f"{data["BNS_SYB"]["obj"]}" + '\033[0m'
        else:
            str_BNS_SYB= f"{data["BNS_SYB"]["obj"]}"

        if data["BNS"]["optimal"] == True:
            str_BNS = '\033[92m' + f"{data["BNS"]["obj"]}" + '\033[0m'
        else:
            str_BNS = f"{data["BNS"]["obj"]}"    
      
        tableRes.add_row([f"{i:02d}",
                          str_LNS_SYB,
                          str_LNS,
                          str_BNS_SYB,
                          str_BNS])
    print(tableRes)
     
    
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
    
