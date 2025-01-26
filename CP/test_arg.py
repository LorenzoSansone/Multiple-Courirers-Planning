import sys
def input_args():
    if len(sys.argv) > 1:
        model, solver = sys.argv[1].rsplit("_", 1)
        #Bs model
        if model == "bs" and solver == "gecode":
            arg_model = "CP_base.mzn"
            arg_solver = "gecode"
        elif model == "bs" and solver == "chuffed":
            arg_model = "CP_base.mzn"
            arg_solver = "chuffed"
            
        #Heu model
        elif model == "bs_heu" and solver == "gecode":
            arg_model = "CP_heu_LNS.mzn"
            arg_solver = "gecode"
        elif model == "bs_heu" and solver == "chuffed":
            arg_model = "CP_heu_chuffed.mzn"
            arg_solver = "chuffed"

        #Heu impl model
        elif model == "bs_heu_impl" and solver == "gecode":
            arg_model = "CP_heu_LNS_impl.mzn"
            arg_solver = "gecode"
        elif model == "bs_heu_impl" and solver == "chuffed":
            arg_model = "CP_heu_impl_chuffed.mzn"
            arg_solver = "chuffed"
        
        #Heu sym model
        elif model == "bs_heu_sym" and solver == "gecode":
            arg_model = "CP_heu_LNS_sym.mzn"
            arg_solver = "gecode"
        elif model == "bs_heu_sym" and solver == "chuffed":
            arg_model = "CP_heu_sym_chuffed.mzn"
            arg_solver = "chuffed"
        
        #Heu impl sym model
        elif model == "bs_heu_sym_impl" and solver == "gecode":
            arg_model = "CP_heu_LNS_sym_impl.mzn"
            arg_solver = "gecode"
        elif model == "bs_heu_sym_impl" and solver == "chuffed":
            arg_model = "CP_heu_sym_impl_chuffed.mzn"
            arg_solver = "chuffed"
        else:
            arg_model = "CP_base.mzn"
            arg_solver = "gecode"
    
        arg_inst = int(sys.argv[2])
    else:
        arg_model = "CP_base.mzn"
        arg_solver = "gecode"
        arg_inst = 1
    
    return arg_model, arg_solver, arg_inst

config_model, config_solver, config_inst = input_args()
    
print(f"model:{config_model}, solver:{config_solver}, inst:{config_inst}")