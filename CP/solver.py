# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 22:39:52 2024

@author: 32057
"""
from minizinc import Instance, Model, Solver
import minizinc
import re



# Function to parse the value after the equal sign
def parse_value(value):
    return value.strip().strip(';')
    
"""
def read_instance1(file_path):
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
        for line in lines:
            line = line.strip()
            data.append(line)
            if line.startswith('m'):
                m = int(parse_value(line.split('=')[1]))
            elif line.startswith('n'):
                n = int(parse_value(line.split('=')[1]))
            elif line.startswith('l'):
                l = parse_value(line.split('=')[1])
            elif line.startswith('s'):
                s = parse_value(line.split('=')[1])
            elif line.startswith('D') or line.startswith('|'):
                # Start reading the distances array

                #distance_lines = distance_lines.strip(" ")[1:]
                distance_lines = line.split('=')[1]
                row = []
                for c in distance_lines:
                    if c.isnumeric():
                        row.append(int(c))

                distances.append(row)
                for line in lines:
                    print("line")
                    
                    #row = parse_value(line.split('=')[1])
                    #print(f"row:{row}{type(row)}")
                
                for i in range(n+1):
                    if not distance_lines:
                        distance_lines = next(file).strip()
                    row = parse_value(distance_lines.split('],')[0] + ']')
                    distances.append(row)
                    distance_lines = next(file).strip()
    
    # Display parsed data
    print("m =", m)
    print("n =", n)
    print("l =", l)
    print("s =", s)
    print("distances =", distances)
    return m, n, l, s, distances
"""
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

def solve_mcp(custom_model, file_path):
  #m, n, l, s, D = read_instance(file_path)
  
  m = "3;"
  n = "7;"
  l = "[15, 10, 7];"
  s = "[3, 2, 6, 8, 5, 4, 4];"
  D ="[|0, 3, 3, 6, 5, 6, 6, 2 | 3, 0, 4, 3, 4, 7, 7, 3 | 3, 4, 0, 7, 6, 3, 5, 3 | 6, 3, 7, 0, 3, 6, 6, 4 | 5, 4, 6, 3, 0, 3, 3, 3 | 6, 7, 3, 6, 3, 0, 2, 4 | 6, 7, 5, 6, 3, 2, 0, 4 | 2, 3, 3, 4, 3, 4, 4, 0 |];"
  LB = "0;"
  UB = "50;"
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
  #instance["o"] = origin_location
  # Solve the problem
  
  result = instance.solve()

  return result


res = solve_mcp("./CP.mzn","../instances_dnz/inst01.dnz")
print(res)