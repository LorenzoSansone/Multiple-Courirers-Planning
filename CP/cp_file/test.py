# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:02:31 2024

@author: 32057
"""
from minizinc import Instance, Model, Solver
import os

from prettytable import PrettyTable
# Load n-Queens model from file
 
model_path = "opt_model.mzn"
timeLimit = 300
first_instance = 0
last_instance = 21
file_name_save = 'test_a.txt'
file_name_error = 'error_model.txt'
mode_save = 'w'
mode_save_error = "a"


for i in range(first_instance, last_instance+1):
    print(i, end = " ")
print()
for i in [x for x in range(first_instance, last_instance+1) if x!=14]:
    print(i, end = " ")
