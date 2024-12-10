import json
import os
from utils_sat import *
import time
from z3 import *
import numpy as np
z = [Bool("z1"), Bool("z2"), Bool("z3")]
x = [Bool("x1"), Bool("x2"), Bool("x3"), Bool("x4")]
y = [Bool("y1"), Bool("y2"), Bool("y3"), Bool("y4")]

for i in range(1,25):
    
    for j in range(1,25):
        s = Solver()
        s.add(greater(int_to_binary(i,num_bits(i)),int_to_binary(j, num_bits(j))))
        print(i,">",j, end = " ")
        if s.check() == sat:
            print("sat")
            if i<=j:
                print("Fail")
        else:
            print("unsat")
"""s = Solver()

s.add(z[0] == True)
s.add(z[1] == True)
s.add(z[2] == True)

s.add(x[0] == False)
s.add(x[1] == True)
s.add(x[2] == False)
s.add(x[3] == False)

s.add(y[0] == True)
s.add(y[1] == False)
s.add(y[2] == True)
s.add(y[3] == False)

s.add(greater(x,y))
print(s.check())


"""