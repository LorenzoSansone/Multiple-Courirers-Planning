# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:02:31 2024

@author: 32057
"""
from minizinc import Instance, Model, Solver

# Load n-Queens model from file
nqueens = Model("./nqueens.mzn")
# Find the MiniZinc solver configuration for Gecode
gecode = Solver.lookup("gecode")
# Create an Instance of the n-Queens model for Gecode
instance = Instance(gecode, nqueens)
# Assign 4 to n
instance["n"] = 4
result = instance.solve()
# Output the array q
print(result["q"])