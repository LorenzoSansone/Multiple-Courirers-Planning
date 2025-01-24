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
