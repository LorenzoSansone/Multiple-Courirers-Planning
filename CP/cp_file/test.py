# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:02:31 2024

@author: 32057
"""
from minizinc import Instance, Model, Solver
import os
# Load n-Queens model from file
 
first_instance = 0
last_instance = 21
for i in [x for x in range(first_instance, last_instance+1) if x != 14]:
    print(i, end = " ")
print()
for i in range(first_instance, last_instance+1):
    print(i, end = " ")
