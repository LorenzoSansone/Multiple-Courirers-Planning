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

async def solve_mcp(custom_model, file_path):
  #m, n, l, s, D = read_instance(file_path)
  """
  m = "3;"
  n = "7;"
  l = "[15, 10, 7];"
  s = "[3, 2, 6, 8, 5, 4, 4];"
  D ="[|0, 3, 3, 6, 5, 6, 6, 2 | 3, 0, 4, 3, 4, 7, 7, 3 | 3, 4, 0, 7, 6, 3, 5, 3 | 6, 3, 7, 0, 3, 6, 6, 4 | 5, 4, 6, 3, 0, 3, 3, 3 | 6, 7, 3, 6, 3, 0, 2, 4 | 6, 7, 5, 6, 3, 2, 0, 4 | 2, 3, 3, 4, 3, 4, 4, 0 |];"
  LB = "0;"
  UB = "50;"

  
  """
  m = 3
  n = 7
  l = [15, 10, 7]
  s = [3, 2, 6, 8, 5, 4, 4]
  D =[[0, 3, 3, 6, 5, 6, 6, 2 ],
        [ 3, 0, 4, 3, 4, 7, 7, 3 ],
        [ 3, 4, 0, 7, 6, 3, 5, 3 ],
        [ 6, 3, 7, 0, 3, 6, 6, 4 ],
        [ 5, 4, 6, 3, 0, 3, 3, 3 ],
        [ 6, 7, 3, 6, 3, 0, 2, 4 ],
        [ 6, 7, 5, 6, 3, 2, 0, 4 ],
        [ 2, 3, 3, 4, 3, 4, 4, 0 ]]
  LB = 0
  UB = 50
  min_dist = 0
  max_dist = 1000
  

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
  instance["min_dist"] = UB
  instance["max_dist"] = UB
  
  #instance["o"] = origin_location
  # Solve the problem
  
  result = await instance.solve_async()

  return result

if __name__ == "__main__":    
    model_name = "CP.mzn"
    data_name = "inst01.dzn"
    
    model_path = "./" + model_name
    data_path = "../instances_dnz/" + data_name
  
    #model_path = os.getcwd() + "\Desktop\CMDO\project_test\Multiple-Courirers-Planning\CP\\" + model_name
    #data_path= os.getcwd() + "\Desktop\CMDO\project_test\Multiple-Courirers-Planning\instances_dnz\\" + data_name
    nest_asyncio.apply()
    res = asyncio.run(solve_mcp(model_path, data_path))
    print(res)
