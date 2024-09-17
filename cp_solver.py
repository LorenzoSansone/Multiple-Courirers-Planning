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

def findLBUB(m, n, l, s, D):
    distances = np.array(D)
    deposit = n 
    min_dist_dep_list = []
    min_dist_dep = 0
    max_dist_dep = distances[n,0]
    max_dist_all_pack = distances[n,0]
    for i in range(n):
        #print(f"deposit -> items_{i+1} = {distances[n,i]}")
        min_dist_dep_list.append(distances[n,i] + distances[i,n])
        if max_dist_dep < distances[n,i]:
            max_dist_dep = distances[n,i]
        
        #print(f"Load couriers:{l}")
        #print(f"Weight items:{s}")
    min_dist_dep = max(min_dist_dep_list)
    min_disep = min(min_dist_dep_list)
    print(f"LB={min_dist_dep}")
    #print(f"max_dist={max_dist_dep}")
    print(f"min_dist:{min_disep}")
    for i in range(n):
        #print(i,i+1)
        max_dist_all_pack = max_dist_all_pack + distances[i,i+1]
    print(f"max_dist and UB:{max_dist_all_pack}")
    print(f"UB={max_dist_all_pack}")
    print(f"LB={min_dist_dep}")
    return min_dist_dep, max_dist_all_pack    
async def solve_mcp(custom_model, file_path):   
  m, n, l, s, D = read_instance(file_path)
  LB, UB = findLBUB(m, n, l, s, D)
  min_dist = 0
  max_dist = UB
  

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
  
  result = await instance.solve_async(timeout=datetime.timedelta(seconds = 300))

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

        
def save_solution(res, data_path):
    # extract number from data_path
    match = re.search(r"inst(\d+)\.dzn", data_path)
    number = match.group(1)
    output = f"{number}"
    output_directory = "res/CP"
    
    # calculate solve time
    solveTime = math.floor(res.statistics['solveTime'].total_seconds())
    
    # check if a solution was found and whether it is optimal
    if res.status.has_solution():
        optimal = res.status == minizinc.result.Status.OPTIMAL_SOLUTION
    else:
        optimal = False
    
    # prepare the solution dictionary
    if res['objective'] is None:
        solution_dict = {
            "gecode": {
                "time": solveTime,
                "optimal": False,
                "obj": None,
                "sol": []
            }
        }
    else:
        output_path = output_path_prepare(res['path'])  # Assumed this function exists
        solution_dict = {
            "gecode": {
                "time": solveTime,
                "optimal": optimal,
                "obj": res['objective'],
                "sol": output_path
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

    model_path = "CP/CP.mzn"

    for i in range(1,11):
        inst_i = f"inst{i:02d}"
        print(f"Instance: {inst_i}")
        data_path = f"instances_dnz/{inst_i}.dzn"
        #model_path = os.getcwd() + "\Desktop\CMDO\project_test\Multiple-Courirers-Planning\CP\\" + model_name
        #data_path= os.getcwd() + "\Desktop\CMDO\project_test\Multiple-Courirers-Planning\instances_dnz\\" + data_name
        nest_asyncio.apply()
        res = asyncio.run(solve_mcp(model_path, data_path))
        print(res)
        save_solution(res, data_path)

