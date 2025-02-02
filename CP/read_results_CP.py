from prettytable import PrettyTable

import time

import json
import os

from prettytable import PrettyTable
from utils_CP import *
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

        #GECODE
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

        ################## CHUFFED
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
    
