from minizinc import Instance, Model, Solver
import utils as utils
import datetime
import os
ALL_MODELS = [  'gurobi_base_bounded_second', 
                'gurobi_base_bounded_penaltyterm', 
                'gurobi_base_bounded_penaltyterm_symbrk',
                'cluster-first_route-second'
                ]
import argparse
import os
import time
from minizinc import Instance, Model, Solver
import utils as utils
import datetime
import json

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MiniZinc solver on a range of instances.")
    parser.add_argument("start", type=int, help="Start of the instance range (inclusive).")
    parser.add_argument("end", type=int, help="End of the instance range (inclusive).")
    parser.add_argument("--model", type=str, choices=ALL_MODELS, help="Name of the model to use. If not specified, solves for all models.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs("res/MIP", exist_ok=True)
    
    # Use either the specified model or all models
    models_to_solve = [args.model] if args.model else ALL_MODELS
    time_limit = 300  # Time limit of 5 minutes

    print(f"Solving for models: {models_to_solve}")
    for model in models_to_solve:
        print(f"MODEL: {model}")
        for i in range(args.start, args.end + 1):
            file_path = f'instances/inst{i:02d}.dat'
            print(f"\nProcessing instance {file_path}")
            start_time = time.time()

            # Read instance data
            m, n, l, s, D, locations = utils.read_input(file_path)
            
            # Calculate bounds
            min_dist, max_dist, LB1, UB1 = utils.find_boundaries_standard(m, n, l, s, D)
            LB2, UB2 = utils.find_boundaries_hybrid(m, n, l, s, D)
            print(f"Instance: inst{i:02d} | LB: {LB1}, UB: {UB1} | Range: {UB1-LB1} (find_boundaries_standard())")
            print(f"Instance: inst{i:02d} | LB: {LB2}, UB: {UB2} | Range: {UB2-LB2} (find_boundaries_hybrid())")        
            
            # Take the biggest lower bound and the smallest upper bound
            LB = max(LB1, LB2)
            UB = min(UB1, UB2)

            # Load the MiniZinc model
            mzn_path = f"MIP/{model}.mzn"
            print(f"Loading model from: {mzn_path}")
            model_instance = Model(mzn_path)
            solver_name = 'gurobi'
            solver = Solver.lookup(solver_name)
            
            # Create a new instance for each iteration
            instance = Instance(solver, model_instance)
            
            # Set all parameters first
            instance["m"] = m
            instance["n"] = n
            instance["l"] = l
            instance["s"] = s
            instance["D"] = D
            instance["locations"] = locations
            instance["LB"] = LB
            instance["UB"] = UB
            
            # Initialize prev_x only once per instance
            if model == 'cluster-first_route-second':
                instance["prev_x"] = [[0 for _ in range(n)] for _ in range(m)]
                instance["new_solution"] = True

            # Solve the model with timeout
            try:
                print(f"\tSolving... ", end='', flush=True)
                result = instance.solve(timeout=datetime.timedelta(seconds=time_limit))
                elapsed_time = time.time() - start_time

                print(f"\rCompleted in {elapsed_time:.1f}s")
                print(f"Status: {result.status.name}")
                
                if result.solution is not None:
                    print(utils.verify_solution(result['x'], result['y'], l, s, D, n, m, locations))
            except Exception as e:
                print(f"Error solving instance: inst{i:02d}.dat | {str(e)}")
                continue

            # Save the solution to file
            output_file = utils.save_solution_by_solver(
                input_file=f"inst{i:02d}.dat",
                m=m,
                n=n,
                model_name = model,
                time_limit=time_limit,
                result=result
            )
            print(f"Solution saved to {output_file}")

if __name__ == "__main__":
    main()