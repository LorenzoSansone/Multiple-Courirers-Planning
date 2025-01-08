# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 22:39:52 2024

@author: 32057
"""
from minizinc import Instance, Model, Solver, Status
import minizinc 
import re
import os
import asyncio
import nest_asyncio
import datetime
import math
import json
import numpy as np
from prettytable import PrettyTable 
import time

# Function to parse the value after the equal sign
def parse_value(value):
    return value.strip().strip(';')
    
#read the instances ".dnz"
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

def save_file(path, mode, text):
    file = open(path, mode)  # append mode
    file.write(text)
    file.close()

def find_boundaries_standard(m, n, l, s, D):
    distances = np.array(D)
    min_dist_dep_list = []
    
    #Define LB
    for i in range(n):
        min_dist_dep_list.append(distances[n,i] + distances[i,n])
    LB = max(min_dist_dep_list)

    #Define UB
    UB = distances[n,0]
    for i in range(n):
        UB = UB + distances[i,i+1]
    UB = int(UB)

    return 0, UB, LB, UB 

def find_boundaries_advanced(m, n, l, s, D):
    distances = np.array(D)
    min_dist_dep_list = []

    #Define LB
    for i in range(n):
        dist_one = distances[n,i] + distances[i,n]
        min_dist_dep_list.append(dist_one)
    LB = max(min_dist_dep_list)

    #Define UB
    UB = distances[n,0]
    for i in range(n):
        UB = UB + distances[i,i+1]
    UB = int(UB)
   
    max_path_len = n-m+1
    return min(min_dist_dep_list), UB, LB, UB, max_path_len

def find_boundaries_optimized(m, n, l, s, D):
    min_dist, max_dist, LB_standard, UB_standard = find_boundaries_standard(m, n, l, s, D)

    UB_list = []
    UB_list.append(UB_standard)

    params = {"m":m, 
            "n":n,
            "l":l,
            "s":s,
            "D":D,
            "LB":LB_standard,
            "UB":UB_standard,
            "min_dist":min_dist,
            "max_dist":max_dist}
    
    ###### UB OPTIMIZED
    timeLimit = 70
    model_path = "UB_model_optimized.mzn"
    solver = "gecode"
    ################ MODEL ################
    res = solve_model(model_path, timeLimit, params, solver)
    if res.objective is not None and isinstance(res.objective, int):
        UB_list.append(res.objective)
    
    UB_opt = min(UB_list)
    print(UB_list,UB_opt)
    return min_dist, UB_opt, LB_standard, UB_opt, UB_list

def solve_model(custom_model, timeLimit ,params, solver):   
  # Load model
  model = minizinc.Model(custom_model)

  gecode = minizinc.Solver.lookup(solver)
  # Create minizinc instance
  instance = minizinc.Instance(gecode, model)
  for k,v in params.items():
      instance[k] = v

  # Solve the problem
  result = instance.solve(timeout=datetime.timedelta(seconds = timeLimit))
  return result

def output_path_prepare(path): 
    path_out = []
    for path_i in path:
        path_i_out = []
        n = max(path_i)
        for k in path_i:
            if k != n:
                path_i_out.append(k)
        path_out.append(path_i_out)
    return path_out

        
def save_solution(title, res, data_path, save_path, timeLimit):
    # extract number from data_path
    match = re.search(r"inst(\d+)\.dzn", data_path)
    number = match.group(1)
    #output = f"{number}"
    output_directory = save_path #"res/CP"
    # prepare the solution dictionary
    if res.objective is None:
        time = timeLimit
        optimal = False
        obj = None
        sol = []
    else:
        if res.statistics['solveTime'] is None:
            time = timeLimit
        else:
            time = math.floor(res.statistics['solveTime'].total_seconds())
        optimal = True if res.status == minizinc.result.Status.OPTIMAL_SOLUTION else False
        obj = res['objective']
        sol = output_path_prepare(res['path'])  
    solution_dict = {
        title: {
            "time": time,
            "optimal": optimal,
            "obj": obj,
            "sol": sol
        }
    }
    
    # ensure the directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    output_file = output_directory + "/" + number + ".json"


    data = {}
    if os.path.isfile(output_file):
        with open(output_file, 'r') as file:
                data = json.load(file)
                

    data.update(solution_dict)
    
    # write the solution to the output file
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    print(f"Solution saved to {output_file}")
    return output_file

def get_name_test(model, params = "standard", solver = "gecode"):
    name_test = ""
    if model == "CP_base.mzn":
        name_test = "BS"
    elif model == "CP_heu_LNS.mzn":
        name_test = ""

    if solver == "gecode":
        name_test = name_test + "Gec"
    elif solver == "chuffed":
        name_test = name_test + "Chuf"
    return name_test



if __name__ == "__main__":    
    solver = "chuffed"
    #models_params_path_list = ["CP_base.mzn", "CP_heu_LNS.mzn", "CP_heu_LNS_sym.mzn","CP_heu_LNS_sym_impl.mzn","CP_heu_LNS_sym_impl2.mzn","CP_heu_LNS_sym2_impl.mzn"]
    models_params_path_list = ["model_all_start_chuffed.mzn"]

    first_instance = 15
    last_instance = 15
    file_name_save = 'result_models_standard_chuffed.txt'
    file_name_error = 'error_model.txt'
    mode_save = 'w'
    mode_save_error = "a"
    
    selectParams = "standard"

    tableRes = PrettyTable(["Instance"] + models_params_path_list) 
    tableRes.title = "MODEL " + solver + " " + selectParams
    save_file(file_name_save, mode_save ,str(tableRes))
    print("START")
    #for i in range(first_instance, last_instance+1):
    for i in [x for x in range(first_instance, last_instance+1) if x!=14]:
        print(i)
        timeLimit = 300

        ################ READ INSTANCES ################
        inst_i = f"inst{i:02d}" #or: inst_i = f"0{i}" if i<10 else i
        data_path = f"../instances_dzn/{inst_i}.dzn"
        m, n, l, s, D = read_instance(data_path)
        
        #START PRE-SOLVING: Compute additional parameters of the models (UB, LB, ...)
        start_pre_solving = time.time()
        if selectParams == "standard":
            min_dist, max_dist, LB, UB = find_boundaries_standard(m, n, l, s, D)
            max_pack = n
        elif selectParams == "advanced":
            min_dist, max_dist, LB, UB, max_pack = find_boundaries_advanced(m, n, l, s, D)
        end_pre_solving = time.time()
        #END PRE-SOLVING

        delta_pre_solving = end_pre_solving - start_pre_solving
        
        #Subtract the pre-solving time from time limit
        timeLimit = int(timeLimit - delta_pre_solving)

        row_table = [inst_i + " (mD:" +  str(min_dist) + " MD:" + str(max_dist) + " LB:" +  str(LB) + " UB:" + str(UB) + " T:" + str(timeLimit) +")"]
  
        params = {"m":m, 
                  "n":n,
                  "l":l,
                  "s":s,
                  "D":D,
                  "LB":LB,
                  "UB":UB,
                  "min_dist":min_dist,
                  "max_dist":max_dist}
        
        ################ MODEL ################
        for model_path in models_params_path_list:
            title_json_test = get_name_test(model_path, selectParams )
            title_json = model_path.replace(".mzn","")

            if selectParams != "standard":
                title_json = title_json + "_adv"
            if model_path == "model_path_opt.mzn":
                params.update({"max_pack": max_pack})
            
            save_solution_path = f"."#f"res/CP" 
            res = None
            try:
                res = solve_model(model_path, timeLimit, params, solver)
            except Exception as e:
    
                row_table.append(str("Error"))
                save_file(file_name_error, mode_save_error ,str(e))
            else:
                """
                if res.objective is not None and isinstance(res.objective, int):
                    flag = "" 
                    if res.status is Status.OPTIMAL_SOLUTION:
                        flag = "(O)"
                    row_table.append(str(res.objective) + flag)
                else:
                    row_table.append(str(res.status))
                """
                save_solution(title_json_test,res, data_path, save_solution_path, timeLimit)

        tableRes.add_row(row_table) 
        print(f"Instance: {inst_i}", row_table)

        save_file(file_name_save, mode_save ,str(tableRes))

        ################################

    ################ RESULT ################
    print(tableRes)
    save_file(file_name_save, mode_save ,str(tableRes))

    ################################
    
        

    
