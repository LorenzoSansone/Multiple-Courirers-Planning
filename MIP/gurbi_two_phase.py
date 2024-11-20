from minizinc import Instance, Model, Solver
import utils
import os, math, json, datetime

if __name__ == "__main__":
    time_limit = 300  # Time limit of 5 minutes
    
    for i in range(1, 22):
        print(f"\nSolving instance: inst{i:02d}.dat")
        file_path = f'instances/inst{i:02d}.dat'
        m, n, l, s, D, locations = utils.read_input(file_path)
        
        # --- Phase 1: Assignment Problem ---
        # Load the MiniZinc model for Phase 1 (Assignment)
        assignment_model = Model("assignment_model.mzn")
        solver = Solver.lookup("gurobi")
        assignment_instance = Instance(solver, assignment_model)
        
        # Set parameters for Phase 1
        assignment_instance["m"] = m
        assignment_instance["n"] = n
        assignment_instance["l"] = l
        assignment_instance["s"] = s
        
        # Solve Phase 1 with timeout
        result_assignment = assignment_instance.solve(timeout=datetime.timedelta(seconds=time_limit))
        print(f"Solved Phase 1 (Assignment) for instance: {file_path}")
        
        # Extract the assignment solution (x[i, j] for each courier i and item j)
        assignment_result = result_assignment["x"]
        
        # --- Phase 2: Routing Problem ---
        # Prepare the assignment result for Phase 2
        routing_model = Model("routing_model.mzn")
        routing_instance = Instance(solver, routing_model)
        
        # Set parameters for Phase 2
        
        routing_instance["m"] = m
        routing_instance["n"] = n
        routing_instance["l"] = l
        routing_instance["s"] = s
        routing_instance["D"] = D
        routing_instance["locations"] = locations
        routing_instance["x"] = assignment_result  # Pass assignment solution as input
        
        # Solve Phase 2 (Routing) with timeout
        result_routing = routing_instance.solve(timeout=datetime.timedelta(seconds=time_limit))
        print(f"Solved Phase 2 (Routing) for instance: {file_path}")
        
        # Save the solution to file
        instance_number = utils.get_instance_number(file_path)
        output_file = utils.save_solution(result_routing, f"inst{instance_number}_solution.dat", m, n)
