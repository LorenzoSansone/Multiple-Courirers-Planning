# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 19:41:55 2024

@author: 32057
"""

from minizinc import Instance, Model, Solver

# Load n-Queens model from file
model = Model("./CP.mzn")
# Find the MiniZinc solver configuration for Gecode
gecode = Solver.lookup("gecode")
# Create an Instance of the n-Queens model for Gecode
instance = Instance(gecode, model)


m = "3;"
n = "7;"
l = "[15, 10, 7];"
s = "[3, 2, 6, 8, 5, 4, 4];"
D ="[|0, 3, 3, 6, 5, 6, 6, 2 | 3, 0, 4, 3, 4, 7, 7, 3 | 3, 4, 0, 7, 6, 3, 5, 3 | 6, 3, 7, 0, 3, 6, 6, 4 | 5, 4, 6, 3, 0, 3, 3, 3 | 6, 7, 3, 6, 3, 0, 2, 4 | 6, 7, 5, 6, 3, 2, 0, 4 | 2, 3, 3, 4, 3, 4, 4, 0 |];"
LB = "0;"
UB = "50;"

instance["m"] = m
instance["n"] = n
instance["l"] = l
instance["s"] = s
instance["D"] = D
instance["LB"] = LB
instance["UB"] = UB

# Assign 4 to n

result = instance.solve()
# Output the array q
print(result["x"])