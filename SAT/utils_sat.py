
from itertools import combinations
from z3 import *

import math
import re

def at_least_one_np(bool_vars):
    return Or(bool_vars)

def at_most_one_np(bool_vars, name = ""):
    return And([Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)])

def exactly_one_np(bool_vars, name = ""):
    return And(at_least_one_np(bool_vars), at_most_one_np(bool_vars, name))

#################SEQUENTIAL###################
def at_least_one_seq(bool_vars):
    return at_least_one_np(bool_vars)

def at_most_one_seq(bool_vars, name):
    constraints = []
    n = len(bool_vars)
    s = [Bool(f"s_{name}_{i}") for i in range(n - 1)]
    constraints.append(Or(Not(bool_vars[0]), s[0]))
    constraints.append(Or(Not(bool_vars[n-1]), Not(s[n-2])))
    for i in range(1, n - 1):
        constraints.append(Or(Not(bool_vars[i]), s[i]))
        constraints.append(Or(Not(bool_vars[i]), Not(s[i-1])))
        constraints.append(Or(Not(s[i-1]), s[i]))
    return And(constraints)

def exactly_one_seq(bool_vars, name):
    return And(at_least_one_seq(bool_vars), at_most_one_seq(bool_vars, name))

################BITWISE##################àà
def toBinary(num, length = None):
    num_bin = bin(num).split("b")[-1]
    if length:
        return "0"*(length - len(num_bin)) + num_bin
    return num_bin
    
def at_least_one_bw(bool_vars):
    return at_least_one_np(bool_vars)

def at_most_one_bw(bool_vars, name):
    constraints = []
    n = len(bool_vars)
    m = math.ceil(math.log2(n))
    r = [Bool(f"r_{name}_{i}") for i in range(m)]
    binaries = [toBinary(i, m) for i in range(n)]
    for i in range(n):
        for j in range(m):
            phi = Not(r[j])
            if binaries[i][j] == "1":
                phi = r[j]
            constraints.append(Or(Not(bool_vars[i]), phi))        
    return And(constraints)

def exactly_one_bw(bool_vars, name):
    return And(at_least_one_bw(bool_vars), at_most_one_bw(bool_vars, name)) 


############HEURISTIC##############
def at_least_one_he(bool_vars):
    return at_least_one_np(bool_vars)

def at_most_one_he(bool_vars, name):
    if len(bool_vars) <= 4:
        return And(at_most_one_np(bool_vars))
    y = Bool(f"y_{name}")
    return And(And(at_most_one_np(bool_vars[:3] + [y])), And(at_most_one_he(bool_vars[3:] + [Not(y)], name+"_")))

def exactly_one_he(bool_vars, name):
    return And(at_most_one_he(bool_vars, name), at_least_one_he(bool_vars))

###################################
def at_least_k_np(bool_vars, k):
    return at_most_k_np([Not(var) for var in bool_vars], len(bool_vars)-k)

def at_most_k_np(bool_vars, k):
    return And([Or([Not(x) for x in X]) for X in combinations(bool_vars, k + 1)])

def exactly_k_np(bool_vars, k):
    return And(at_most_k_np(bool_vars, k), at_least_k_np(bool_vars, k))

####################################
def parse_value(value):
    return value.strip().strip(';')
    
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

def num_bits(x):
    """Compute the number of bits necessary to represent the integer number x"""
    return math.floor(math.log2(x)) + 1

def int_to_binary(x, digits):
    """Get binary representation of integer x"""
    res = []
    for _ in range(digits):
        res.append(x%2==1)
        x = x//2
    res.reverse()

    return res


def geq(x,y):
    """Encoding of x >= y"""
    #encoding_num x and y

    if len(x) != len(y):
        raise ValueError("Both lists must have the same length.")
    
    if len(x) == 1:
        return Or(x[0]==y[0], And(Not(y[0]), x[0]))
    else:
        return Or(And(Not(y[0]), x[0]),
                  And(x[0]==y[0], geq(x[1:], y[1:])))
    

def sum_b(l1, l2):   
                                
    #l1, l2 = get_bool_lists(l1, l2)
    max_len = max(len(l1), len(l2))
    #l1 = pad(l1, max_len)
    #l2 = pad(l2, max_len)
    
    result = []

    carry_in = False
    carry_out = False

    for i in range(max_len - 1, -1, -1):
        a = l1[i]
        b = l2[i]
        result.append(Xor(Xor(a, b), carry_in))

        carry_out = Or(And(Xor(a, b), carry_in), And(a, b))
        carry_in = carry_out
    result.append(carry_in)
    result = result[::-1]

    return result

def sum_one_bit(x, y, c_in, res, c_res):
  """
      Implementation of the full adder for one bit
      Constraints for a + b + c_in = res with c_res (1 bit)
      :param a:     first bit
      :param b:     second bit
      :param c_in:  carry in
      :param res:   result
      :param c_res: carry result  
  """
  # Xor(A, B) encodes the binary sum between the bit A and the bit B
  c_1 = res == Xor( Xor(x, y), c_in)  #Sum between x, y and c_in
  c_2 = c_res == Or(And( Xor(x, y) , c_in), And(x, y)) #Computation of the carry
  return And(c_1, c_2) 

def pad_bool(x, length):
    return [BoolVal(False)] * (length - len(x)) + x

def sum_bin(x, y, res, name= ""):
    """
      The constraints for full adder. x + y = res
      :param x:   binary inputs
      :param y:   binary inputs
      :param res: binary result
      :param name:
    """

    max_len = max(len(x), len(y))
    x = pad_bool(x, max_len)
    y = pad_bool(y, max_len)

    c = [Bool(f"carry_{name}_{i}") for i in range(max_len)] + [BoolVal(False)]

    constr = []

    for i in range(max_len):
        constr.append(sum_one_bit(x= x[max_len-i-1], y = y[max_len-i-1], c_in= c[max_len - i], res= res[max_len - i - 1], c_res= c[max_len - i - 1]))
    constr.append(Not(c[0]))
    return And(constr)