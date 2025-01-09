from prettytable import PrettyTable


configs = [["CP_base.mzn","standard","gecode"],
           ["CP_base1.mzn","standard","gecode"],
           ["CP_base2.mzn","standard","gecode"]]

all_model_path = [config[0] for config in configs]
first_instance = 1
last_instance = 21


tableRes = PrettyTable() 
tableRes.title = "MODEL"
tableRes.add_column("instances",[str(x) for x in range(first_instance, last_instance+1) if x!=14])
tableRes.add_column("instances",[str(x) for x in range(first_instance, 5+1) if x!=14])
print(tableRes)