from minizinc import Instance, Model, Solver
import utils
import os, math, json, datetime

if __name__ == "__main__":
    time_limit = 300  # Time limit of 5 minutes
    for i in range(1, 22):
        print(f"\nSolving instance: inst{i:02d}.dat")
        file_path = f'instances/inst{i:02d}.dat'
        m, n, l, s, D, locations = utils.read_input(file_path)
        
        # Load the MiniZinc model
        model = Model("gurbi.mzn")
        solver = Solver.lookup("gurobi")
        instance = Instance(solver, model)
        
        # Set parameters
        instance["m"] = m
        instance["n"] = n
        instance["l"] = l
        instance["s"] = s
        instance["D"] = D
        instance["locations"] = locations
        
        # Solve the model with timeout
        result = instance.solve(timeout=datetime.timedelta(seconds=time_limit))
        print(f"Solved instance: {file_path}")
        
        # Save the solution to file
        instance_number = utils.get_instance_number(file_path)
        output_file = utils.save_solution(result, f"inst{instance_number}.dat", m, n)
