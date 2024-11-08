# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:54:38 2024

@author: 32057
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 22:39:52 2024

@author: 32057
"""
from minizinc import Instance, Model, Solver
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

def find_boundaries(m, n, l, s, D):
    distances = np.array(D)
    min_dist_dep_list = []
    
    for i in range(n):
        min_dist_dep_list.append(distances[n,i] + distances[i,n])
    LB = max(min_dist_dep_list)

    UB = distances[n,0]
    for i in range(n):
        UB = UB + distances[i,i+1]

    
    return min(min_dist_dep_list), UB, LB, UB    

def solve_model(custom_model, timeLimit ,params):   
  # Load model
  model = minizinc.Model(custom_model)

  gecode = minizinc.Solver.lookup("gecode")
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

        
def save_solution(res, data_path, timeLimit):
    # extract number from data_path
    match = re.search(r"inst(\d+)\.dzn", data_path)
    number = match.group(1)
    #output = f"{number}"
    output_directory = "res/CP"
    # prepare the solution dictionary
    if res.objective is None:
        time = timeLimit
        optimal = False
        obj = None
        sol = []
    else:
        time = math.floor(res.statistics['solveTime'].total_seconds())
        optimal = True if res.status == minizinc.result.Status.OPTIMAL_SOLUTION else False
        obj = res['objective']
        sol = output_path_prepare(res['path'])  
    solution_dict = {
        "gecode": {
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
    # write the solution to the output file
    with open(output_file, 'w') as outfile:
        json.dump(solution_dict, outfile, indent=4)

    print(f"Solution saved to {output_file}")
    return output_file
"""
def findLB(m, n, l, s, D):
    distances = np.array(D)
    min_dist_dep_list = []
    dict_res = {}
    
    for i in range(n):
        min_dist_dep_list.append(distances[n,i] + distances[i,n])
        dict_res.update({f'n > {i} > n': distances[n,i] + distances[i,n]})
    min_dist_dep = min(min_dist_dep_list)
    return min_dist_dep, dict_res
"""
def findB(m, n, l, s, D):
    return (n-m)+1
if __name__ == "__main__":  
    models_params_path_list = ["UB_model.mzn", "UB_model_optimized.mzn"]
    timeLimit = 70
    first_instance = 0
    last_instance = 21

    tableRes = PrettyTable(["Instance", "LB_standard", "UB_standard", "UB_model","UB_model_optimized"]) 
    tableRes.title = "PARAMETER LB UB"

    for i in range(first_instance, last_instance+1):

        ################ SET PARAMETERS ################
        inst_i = f"inst{i:02d}" #or: inst_i = f"0{i}" if i<10 else i
        data_path = f"../instances_dzn/{inst_i}.dzn"
        m, n, l, s, D = read_instance(data_path)
        min_dist, max_dist, LB_standard, UB_standard = find_boundaries(m, n, l, s, D)
        B = findB(m, n, l, s, D)
        print(i, "----")
        print(min_dist, max_dist, LB_standard, UB_standard)
        print("B",B)
        row_table = [inst_i, str(LB_standard), str(UB_standard)]
        """
        params = {"m":m, 
                  "n":n,
                  "l":l,
                  "s":s,
                  "D":D,
                  "LB":LB_standard,
                  "UB":UB_standard,
                  "min_dist":min_dist,
                  "max_dist":max_dist}
        ################################

        ################ MODEL ################
        for model_path in models_params_path_list:
            res = solve_model(model_path, timeLimit, params)
            if res.objective is not None and isinstance(res.objective, int):
                row_table.append(str(res.objective))
            else:
                row_table.append(str(res.status))
        tableRes.add_row(row_table) 
        print(f"Instance: {inst_i}", row_table)
        
        ################################

    ################ RESULT ################
    print(tableRes)
    with open('parameters_UB_LB.txt', 'w') as w:
        w.write(str(tableRes))
    ################################
        """
       
    