from itertools import combinations
from z3 import *

import math
import re
n = 2
path_b = [[0,1,0],
          [0,1,1],
          [1,0,1],
          [1,1,1]]
list_res = []

for j in range(len(path_b[0])):

    for i in range(len(path_b)):
        list_res.append(path_b[i][j])
        print(path_b[i][j], end = " ")

    print()

"""
x = Int('x')
y = Int('y')

s = Solver()
print (s)

s.add(x > 10, y == x + 2)
print (s)
print ("Solving constraints in the solver s ...")
print (s.check())

###########
print ("1)Create a new scope...")
s.push()

s.add(y < 11)
print (s)
print ("Solving updated set of constraints...")
print (s.check())

print ("Restoring state...")
s.pop()
print (s)
print ("Solving restored set of constraints...")
print (s.check())
########

print ("2)Create a new scope...")
s.push()

s.add(y < 14)
print (s)
print ("Solving updated set of constraints...")
print (s.check())


print ("Restoring state...")
s.pop()
print (s)
print ("Solving restored set of constraints...")
print (s.check())
print(s.model().evaluate(y))
print(s.model()[y])
"""
