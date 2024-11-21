from minizinc import Instance, Model, Solver
import utils as utils
import os, math, json, datetime, re
import numpy as np  # Required for the boundary calculation

ALL_MODELS = [  'base_bounded', 
                'base_bounded_penaltyterm', 
                'base_bounded_penaltyterm_symbrk'
                ]

# Function to calculate the bounds

if __name__ == "__main__":
    time_limit = 300  # Time limit of 5 minutes
    for mod in ALL_MODELS:
        print(f"MODEL: {mod}")
        for i in range(1, 22):
            file_path = f'instances/inst{i:02d}.dat'
            chosen_model = mod
            # Read instance data
            m, n, l, s, D, locations = utils.read_input(file_path)
            
            # Calculate bounds
            min_dist, max_dist, LB, UB = utils.find_boundaries_standard(m, n, l, s, D)
            print(f"Instance: inst{i:02d} | LB: {LB}, UB: {UB}")

            # Load the MiniZinc model
            model = Model(f"MIP/{chosen_model}.mzn")
            solver_name = "gecode"
            solver = Solver.lookup(solver_name)
            instance = Instance(solver, model)
            
            # Set parameters for the instance
            for param in ["m", "n", "l", "s", "D", "locations", "LB", "UB", "min_dist", "max_dist"]:
                instance[param] = locals()[param]
            
            # Solve the model with timeout
            try:
                print(f"\nSolving instance: inst{i:02d}.dat | Model: {chosen_model}.mzn | Solver: {solver_name}")
                result = instance.solve(timeout=datetime.timedelta(seconds=time_limit))
                print(f"Solved instance: {file_path} | Objective: {result.objective if result.objective else 'No solution'} | Optimal: {result.status.name == 'OPTIMAL_SOLUTION'}")
                
                # Ensure result contains valid data
                if result.solution is not None:  # Check if a solution exists
                    # utils.debug(result['x'], result['y'], m, n, s, l, D) # test/debug function
                    pass
            except Exception as e:
                print(f"Error solving instance: inst{i:02d}.dat | {str(e)}")
                continue


            # Save the solution to file
            instance_number = utils.get_instance_number(file_path)
            inst_i = f"inst{i:02d}" #or: inst_i = f"0{i}" if i<10 else i
            data_path = f"../instances_dzn/{inst_i}.dzn"
            output_file = utils.save_solution_by_solver(time_limit = time_limit, result = result, input_file = f"inst{instance_number}.dat", m=m, n=n, solver_name = solver_name)
            #output_file = utils.save_solution_by_model(input_file = f"inst{instance_number}.dat", m = m, n = n, model_name = chosen_model, time_limit = time_limit, result = result)
