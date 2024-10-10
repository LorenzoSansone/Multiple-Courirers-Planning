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
first_instance = 11
last_instance = 10
file_name_save = 'test_a.txt'
file_name_error = 'error_model.txt'
mode_save = 'w'
mode_save_error = "a"



tableRes = PrettyTable(["Instance"] + [model_path])
print(tableRes) 