# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 19:41:55 2024

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

if __name__ == "__main__":    
    model_name = "CP.mzn"
    data_name = "inst21.dzn"
    
    model_path = "./" + model_name
    data_path = "../instances_dnz/" + data_name
    m, n, l, s, distances = read_instance(data_path)
    distances = np.array(distances)
    deposit = n 
    print(f"{data_name}")
    print(f"Couriers:{m}")
    print(f"Items:{n}")
    print(f"1)m < n {m<n}")
    print(f"2)min(l(m))> min(w(s)){min(l)> min(s)}")
    print(f"Load couriers:{l}")
    print(f"Weight items:{s}")
    print(f"Sum weights{sum(s)}")
# obj_lowerbound = max(i in ITEMS)(D[n+1,i] + D[i,n+1]);
    min_dist_dep_list = []
    min_dist_dep = 0
    max_dist_dep = distances[n,0]
    max_dist_all_pack = distances[n,0]
    for i in range(n):
        print(f"deposit -> items_{i+1} = {distances[n,i]}")
        min_dist_dep_list.append(distances[n,i] + distances[i,n])
        if max_dist_dep < distances[n,i]:
            max_dist_dep = distances[n,i]
        
        #print(f"Load couriers:{l}")
        #print(f"Weight items:{s}")
    min_dist_dep = max(min_dist_dep_list)
    print(f"min_dist={min_dist_dep}")
    print(f"max_dist={max_dist_dep}")
    for i in range(n):
        print(i,i+1)
        max_dist_all_pack = max_dist_all_pack + distances[i,i+1]
    print(f"Max distances:{max_dist_all_pack}")
        
    #model_path = os.getcwd() + "\Desktop\CMDO\project_test\Multiple-Courirers-Planning\CP\\" + model_name
    #data_path= os.getcwd() + "\Desktop\CMDO\project_test\Multiple-Courirers-Planning\instances_dnz\\" + data_name
"""  
    for i in range(0,10):
        data_name = "inst0" + str(i) + ".dzn"
        data_path = "../instances_dnz/" + data_name
        m, n, l, s, distances = read_instance(data_path)
        print(f"ISTANCES_0{i}")
        print(f"Couriers:{m}")
        print(f"Items:{n}")
        print(f"1)m < n {m<n}")
        print(f"2)min(l(m))> min(w(s)){min(l)> min(s)}")
        #print(f"Load couriers:{l}")
        #print(f"Weight items:{s}")
        print()
        
        
        print()
    for i in range(10,22):
        data_name = "inst" + str(i) + ".dzn"
        data_path = "../instances_dnz/" + data_name
        m, n, l, s, distances = read_instance(data_path)
        print(f"ISTANCES_{i}")
        print(f"Couriers:{m}")
        print(f"Items:{n}")
        print(f"1)m < n {m<n}")
        print(f"2)min(l(m))> min(w(s)){min(l)> min(s)}")
        #print(f"Load couriers:{l}")
        #print(f"Weight items:{s}")
        print()
"""
        
