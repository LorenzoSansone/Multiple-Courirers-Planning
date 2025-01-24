# Python program to demonstrate
# command line arguments


import sys

def input_args():
    if len(sys.argv) > 1:
        if sys.argv[1] == "LNS":
            arg_sym = False
            arg_search = "linear"
        elif sys.argv[1] == "LNS_SYB":
            arg_sym = True
            arg_search = "linear"
        elif sys.argv[1] == "BNS":
            arg_sym = False
            arg_search = "binary"
        elif sys.argv[1] == "BNS_SYB":
            arg_sym = True
            arg_search = "binary"

        arg_inst = int(sys.argv[2])

    else:
        arg_inst = 1
        arg_search = "linear"
        arg_sym = False
    print(f"Inst:{arg_inst} search:{arg_search} sym:{arg_sym}")
    return arg_inst, arg_search, arg_sym

config_inst, config_search, config_sym = input_args()
print(f"Inst:{config_inst} search:{config_search} sym:{config_sym}")

