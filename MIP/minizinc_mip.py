from minizinc import Instance, Model, Solver
import utils as utils
import datetime
ALL_MODELS = [  #'base_bounded', 
                'base_bounded_penaltyterm', 
                #'base_bounded_penaltyterm_symbrk'
                ]
ALL_SOLVERS = [
                'gurobi',
                #'gecode',
                #'chuffed',
                #'cplex'
]
if __name__ == "__main__":
    time_limit = 300  # Time limit of 5 minutes
    for mod in ALL_MODELS:
        print(f"MODEL: {mod}")
        for i in range(14, 22):
            file_path = f'instances/inst{i:02d}.dat'
            chosen_model = mod
            # Read instance data
            m, n, l, s, D, locations = utils.read_input(file_path)
            # Calculate bounds
            min_dist, max_dist, LB1, UB1 = utils.find_boundaries_standard(m, n, l, s, D)
            LB2, UB2 = utils.find_boundaries_hybrid(m, n, l, s, D)
            print(f"\nInstance: inst{i:02d} | LB: {LB1}, UB: {UB1} | Range: {UB1-LB1} (find_boundaries_standard())")
            print(f"Instance: inst{i:02d} | LB: {LB2}, UB: {UB2} | Range: {UB2-LB2} (find_boundaries_hybrid())")        
            # Take the biggest lower bound and the smallest upper bound
            LB = max(LB1, LB2)
            UB = min(UB1, UB2) 
            # Load the MiniZinc model
            model = Model(f"MIP/{chosen_model}.mzn")
            solver_name = "gurobi"
            solver = Solver.lookup(solver_name)
            instance = Instance(solver, model)
            
            # Set parameters for the instance
            for param in ["m", "n", "l", "s", "D", "locations", "LB", "UB"]:
                instance[param] = locals()[param]
            
            # Solve the model with timeout
            try:
                print(f"Solving instance: inst{i:02d}.dat | Model: {chosen_model}.mzn | Solver: {solver_name}")
                result = instance.solve(timeout=datetime.timedelta(seconds=time_limit))
                print(f"Solved instance: {file_path} | Objective: {result.objective if result.objective else 'No solution'} | Optimal: {result.status.name == 'OPTIMAL_SOLUTION'}")
                if result.solution is not None:  # Check if a solution exists
                    # utils.debug(result['x'], result['y'], m, n, s, l, D) # test/debug function
                    print(utils.verify_solution(result['x'], result['y'], l, s, D, n, m, locations))
            except Exception as e:
                print(f"Error solving instance: inst{i:02d}.dat | {str(e)}")
                continue
            # Save the solution to file
            instance_number = utils.get_instance_number(file_path)
            inst_i = f"inst{i:02d}" 
            output_file = utils.save_solution_by_solver(time_limit = time_limit, result = result, input_file = f"inst{instance_number}.dat", m=m, n=n, solver_name = solver_name)
            #output_file = utils.save_solution_by_model(input_file = f"inst{instance_number}.dat", m = m, n = n, model_name = chosen_model, time_limit = time_limit, result = result)
