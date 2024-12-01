import argparse
import z3
import json
import os
import sys
import time
import numpy as np
import math
from datetime import timedelta
from z3 import is_true
from dataclasses import dataclass
from utils import minutes_to_milliseconds, seconds_to_milliseconds, milliseconds_to_seconds, Solution, Status, Result
from Z3_SMT_Base_Solver import Z3_SMT_Base_Solver
from Z3_SMT_SymBrk_Solver import Z3_SMT_SymBrk_Solver
from Z3_SMT_SymBrk_ImplConstr_Solver import Z3_SMT_SymBrk_ImplConstr_Solver
TIMEOUT_TIME = minutes_to_milliseconds(5)

models = ['z3_smt_symbrk', 'z3_smt_base', 'z3_smt_symbrk_implconstr']

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run SMT solver on a range of instances.")
    parser.add_argument("start", type=int, help="Start of the instance range (inclusive).")
    parser.add_argument("end", type=int, help="End of the instance range (inclusive).")
    parser.add_argument("--model", type=str, choices=models, help="Name of the model to use. If not specified, solves for all models.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs("res/SMT", exist_ok=True)
    
    # Use either the specified model or all models
    models_to_solve = [args.model] if args.model else models
    print(f"Solving for models: {models_to_solve}")
    for model in models_to_solve:
        print(f"Solving for model: {model}")
        for i in range(args.start, args.end + 1):
            file_path = f'instances/inst{i:02d}.dat'
            print(f"\nProcessing instance {file_path}")
            start_time = time.time()

            # Instantiate the solver based on the model
            match model:
                case "z3_smt_symbrk": solver = Z3_SMT_SymBrk_Solver(file_path, TIMEOUT_TIME)
                case "z3_smt_base": solver = Z3_SMT_Base_Solver(file_path, TIMEOUT_TIME)
                case "z3_smt_symbrk_implconstr": solver = Z3_SMT_SymBrk_ImplConstr_Solver(file_path, TIMEOUT_TIME)
                case _: raise ValueError(f"Invalid model name: {model}")

            # Solve the problem
            print("\tSolving... ", end='', flush=True)
            result = solver.solve(timeout_ms=TIMEOUT_TIME)
            elapsed_time = time.time() - start_time

            # Solving process ended
            print(f"\rCompleted in {elapsed_time:.1f}s")
            print(f"Status: {result.status.name}")

            # Save solution
            output_file = solver.save_solution_by_model(
                input_file=file_path,
                m=solver.num_couriers,
                n=solver.num_items,
                model_name=model,
                time_limit=milliseconds_to_seconds(TIMEOUT_TIME),
                result=result
            )
            print(f"Solution saved to {output_file}")
if __name__ == "__main__":
    main()