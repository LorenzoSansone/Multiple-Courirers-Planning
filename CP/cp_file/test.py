# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:02:31 2024

@author: 32057
"""
from minizinc import Instance, Model, Solver
import os
# Load n-Queens model from file


nqueens = Model( "./nqueens.mzn")
# Find the MiniZinc solver configuration for Gecode
gecode = Solver.lookup("gecode")
# Create an Instance of the n-Queens model for Gecode
params = {"n":4}
# Assign 4 to n
for k,v in params.items():
    print(k,v)
    instance = Instance(gecode, nqueens)
    instance[k] = v
    result = instance.solve()
    # Output the array q
    print(result["q"])
