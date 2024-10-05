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

async def find_LB_model(custom_model, file_path, timeLimit):
  m, n, l, s, D = read_instance(file_path)
  LB, UB = find_LB_UB(m, n, l, s, D)
  #min_dist = 0
  #max_dist = UB
  

  # Load model
  model = minizinc.Model(custom_model)

  gecode = minizinc.Solver.lookup("gecode")
  # Create minizinc instance
  instance = minizinc.Instance(gecode, model)
  instance["n"] = n
  instance["D"] = D
  instance["LB"] = LB
  instance["UB"] = UB
  instance["min_dist"] = 0
  instance["max_dist"] = UB
  
  #instance["o"] = origin_location
  # Solve the problem
  result = await instance.solve_async(timeout=datetime.timedelta(seconds = timeLimit))

  return result


def find_LB_UB(m, n, l, s, D):
    distances = np.array(D)
    min_dist_dep_list = []
    
    for i in range(n):
        min_dist_dep_list.append(distances[n,i] + distances[i,n])
    min_dist_dep = max(min_dist_dep_list)

    max_dist_all_pack = distances[n,0]
    for i in range(n):
        max_dist_all_pack = max_dist_all_pack + distances[i,i+1]

    return min_dist_dep, max_dist_all_pack    



async def solve_mcp(custom_model, file_path, timeLimit):   
  m, n, l, s, D = read_instance(file_path)
  LB, UB = find_LB_UB(m, n, l, s, D)
  #min_dist = 0
  #max_dist = UB
  

  # Load model
  model = minizinc.Model(custom_model)

  gecode = minizinc.Solver.lookup("gecode")
  # Create minizinc instance
  instance = minizinc.Instance(gecode, model)
  instance["m"] = m
  instance["n"] = n
  instance["l"] = l
  instance["s"] = s
  instance["D"] = D
  instance["LB"] = LB
  instance["UB"] = UB
  instance["min_dist"] = 0
  instance["max_dist"] = UB
  
  #instance["o"] = origin_location
  # Solve the problem
  result = await instance.solve_async(timeout=datetime.timedelta(seconds = timeLimit))

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



if __name__ == "__main__":    
    timeLimit = 60  #seconds
    first_instance = 0
    last_instance = 21
    
    for i in range(first_instance, last_instance+1):
        inst_i = f"inst{i:02d}" #or: inst_i = f"0{i}" if i<10 else i
        print(f"Instance: {inst_i}")
        data_path = f"instances_dnz/{inst_i}.dzn"

        m, n, l, s, D = read_instance(data_path)
        LB, UB = find_LB_UB(m, n, l, s, D)
        print(f"LB:{LB} UB:{UB}")
        print("")
        #res_model = find_LB_model("CP/UB_model.dzn",data_path,timeLimit)

        #res_model = asyncio.run(find_LB_model("CP/UB_model.dzn",data_path,timeLimit))
        #res_model_s = asyncio.run(find_LB_model("CP/UB_model_s.dzn",data_path,timeLimit))
    
    """
    nest_asyncio.apply()

    data_path = f"instances_dnz/inst00.dzn"
    
    m, n, l, s, D = read_instance(data_path)
    LB, UB = find_LB_UB(m, n, l, s, D)
    res_model = asyncio.run(solve_mcp("CP/CP.mzn",data_path,timeLimit))
    #res_model = asyncio.run(find_LB_model("CP/UB_model.mzn",data_path,timeLimit))
    #res_model_s = asyncio.run(find_LB_model("CP/UB_model_s.mzn",data_path,timeLimit))
    print(data_path)
    print("---------")
    print("ANALYTICAL")
    print(f"LB: {LB}")
    print(f"UB: {UB}")
    print("---------")
    print("MODEL")
    print(res_model)

    print("---------")
    print("MODEL FAST")
    #print(res_model_s)

    """
  
    """  
    model_path = "CP/CP.mzn"
    timeLimit = 3  #seconds
    first_instance = 11
    last_instance = 21

    for i in range(first_instance, last_instance+1):
        inst_i = f"inst{i:02d}" #or: inst_i = f"0{i}" if i<10 else i
        print(f"Instance: {inst_i}")
        data_path = f"instances_dnz/{inst_i}.dzn"
        nest_asyncio.apply()
        res = asyncio.run(solve_mcp(model_path, data_path, timeLimit))
        print(f"Solution: {res.objective}, status: {res.status}, time: {math.floor(res.statistics['solveTime'].total_seconds()) if res.objective is not None else timeLimit}")
        save_solution(res, data_path, timeLimit)
    """

    
