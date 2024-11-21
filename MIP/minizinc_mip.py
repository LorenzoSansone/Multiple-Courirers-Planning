from minizinc import Instance, Model, Solver
import utils as utils
import os, math, json, datetime
import numpy as np  # Required for the boundary calculation

# Function to calculate the bounds
def find_boundaries_standard(m, n, l, s, D):
    distances = np.array(D)  # Convert distance matrix to numpy array for easier handling
    
    # Lower bound (LB): Maximum of minimum distances for a round trip to each item
    min_dist_dep_list = []
    for i in range(n):
        min_dist_dep_list.append(distances[n, i] + distances[i, n])  # To and from the origin
    LB = max(min_dist_dep_list)  # The largest of these minimum round trips

    # Upper bound (UB): Greedy estimate for total tour distance
    UB = distances[n, 0]  # Start from origin
    for i in range(n - 1):
        UB += distances[i, i + 1]  # Add distances between consecutive locations
    UB += distances[n - 1, n]  # Return to the origin
    UB = int(UB)

    return 0, UB, LB, UB  # Return values as in the example

if __name__ == "__main__":
    time_limit = 300  # Time limit of 5 minutes
    for i in range(1, 22):
        print(f"\nSolving instance: inst{i:02d}.dat")
        file_path = f'instances/inst{i:02d}.dat'
        
        # Read instance data
        m, n, l, s, D, locations = utils.read_input(file_path)
        
        # Calculate bounds
        min_dist, max_dist, LB, UB = find_boundaries_standard(m, n, l, s, D)
        print(f"Instance: inst{i:02d} | LB: {LB}, UB: {UB}, MinDist: {min_dist}, MaxDist: {max_dist}")

        # Load the MiniZinc model
        model = Model("MIP/base.mzn")
        solver = Solver.lookup("gurobi")
        instance = Instance(solver, model)
        
        # Set parameters for the instance
        for param in ["m", "n", "l", "s", "D", "locations", "LB", "UB", "min_dist", "max_dist"]:
            instance[param] = locals()[param]

        # Solve the model with timeout
        try:
            result = instance.solve(timeout=datetime.timedelta(seconds=time_limit))
            print(f"Solved instance: {file_path} | Objective: {result.objective if result.objective else 'No solution'} | Optimal: {result.status.name=='OPTIMAL_SOLUTION'}")
        except Exception as e:
            print(f"Error solving instance: inst{i:02d}.dat | {str(e)}")
            continue

        # Save the solution to file
        instance_number = utils.get_instance_number(file_path)
        output_file = utils.save_solution(result, f"inst{instance_number}.dat", m, n)
