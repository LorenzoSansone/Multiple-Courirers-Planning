import gurobipy as gp
from gurobipy import GRB, quicksum
import json
import os
import math
import utils as utils
import argparse

def create_mcp_model(env, m, n, l, s, D, locations):
    """Create the MCP model with the given parameters."""
    try:
        print("\nCreating MCP model...")
        # Create a new model with the shared environment
        model = gp.Model("MCP", env=env)
        
        print("Creating variables...")
        # Variables
        x = model.addVars(m, n, vtype=GRB.BINARY, name='x')  # x[i, j] = 1 if courier i delivers item j
        y = model.addVars(m, locations, locations, vtype=GRB.BINARY, name='y')  # y[i, j1, j2] = 1 if courier i travels from j1 to j2
        Distance = model.addVars(m, vtype=GRB.INTEGER, name="Distance")  # Total distance for each courier
        max_distance = model.addVar(vtype=GRB.INTEGER, name="max_distance")  # Maximum distance among couriers
        # Add variables for subtour elimination
        u = model.addVars(m, n, vtype=GRB.CONTINUOUS, name="u")

        print("Adding constraints...")
        # Load constraints
        for i in range(m):
            model.addConstr(quicksum(x[i, j] * s[j] for j in range(n)) <= l[i],
                            name=f"Load constraint: courier {i}")

        # Each item must be delivered by exactly one courier
        for j in range(n):
            model.addConstr(quicksum(x[i, j] for i in range(m)) == 1,
                            name=f"Delivery by one courier constraint: item {j}")

        # Each courier starts at the origin (location n) and ends at the origin
        for i in range(m):
            model.addConstr(quicksum(y[i, n, j] for j in range(n)) == 1, 
                          name=f"Start at origin, courier {i}")
            model.addConstr(quicksum(y[i, j, n] for j in range(n)) == 1, 
                          name=f"End at origin, courier {i}")

        # Prevent couriers from traveling from a point to itself
        for i in range(m):
            for j in range(locations):
                model.addConstr(y[i, j, j] == 0,
                                name=f"Travel from point to itself, courier {i}, location {j}")

        # if a courier picks up an item, they must visit the item's location and leave it
        for i in range(m):
            for j in range(n):
                model.addConstr(quicksum(y[i,j,k] for k in range(locations)) == x[i,j],
                                name=f"Visit item location, courier {i}, location {j}")
                model.addConstr(quicksum(y[i,k,j] for k in range(locations)) == x[i,j],
                                name=f"Leave item location, courier {i}, location {j}")

        # Flow conservation constraints
        for i in range(m):
            for j in range(1, locations):
                model.addConstr(quicksum(y[i, k, j] for k in range(locations)) ==
                                quicksum(y[i, j, k] for k in range(locations)),
                                name=f"Flow conservation constraint, courier {i}, location {j}")

        # Subtour elimination constraints
        for i in range(m):
            for j in range(n):
                for k in range(n):
                    if j != k:
                        model.addConstr(u[i, j] - u[i, k] + (n - 1) * y[i, j, k] <= n - 2,
                                    name=f"Subtour elimination constraint, courier {i}, locations {j} and {k}")

        # Set bounds for u variables
        for i in range(m):
            for j in range(1, n):
                model.addConstr(u[i, j] >= 1)
                model.addConstr(u[i, j] <= n - 1,
                                name=f"Set bounds for u variable: courier {i}, location {j}")

        # Distance calculation constraint
        for i in range(m):
            model.addConstr(
                Distance[i] == quicksum(y[i, j1, j2] * D[j1][j2]
                                    for j1 in range(locations)
                                    for j2 in range(locations)),
                name=f"Distance calculation for courier {i}")

        # Max distance objective
        for i in range(m):
            model.addConstr(max_distance >= Distance[i],
                            name=f"Max distance constraint courier {i}")

        print("Setting objective...")
        model.setObjective(max_distance, GRB.MINIMIZE)
        
        # Store model attributes for solution extraction
        model._vars = {
            'x': x,
            'y': y,
            'Distance': Distance,
            'max_distance': max_distance,
            'm': m,
            'n': n,
            'locations': locations
        }
        
        print("Model created successfully")
        return model

    except Exception as e:
        print(f"Error creating model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def extract_solution(model, time_limit=300):
    """Extract solution from the optimized model."""
    try:
        print("\nExtracting solution...")
        model.Params.MIPGap = 0.05    # 5% gap on time limit
        model.Params.TimeLimit = time_limit
        model.optimize()
        
        print(f"Model status: {model.Status} (OPTIMAL={GRB.OPTIMAL}, TIME_LIMIT={GRB.TIME_LIMIT})")
        
        if model.Status == GRB.OPTIMAL:
            print("Optimal solution found.")
        elif model.Status == GRB.TIME_LIMIT:
            print("Time limit reached. Best feasible solution obtained.")
        elif model.Status == GRB.INFEASIBLE:
            print("Model is infeasible; no solution found.")
            diagnose_infeasibility(model)
            return None
        else:
            print(f"Optimization was stopped with status {model.Status}")
            return None

        if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
            vars = model._vars
            m = vars['m']
            n = vars['n']
            locations = vars['locations']
            x = vars['x']
            y = vars['y']
            Distance = vars['Distance']
            max_distance = vars['max_distance']
            
            print(f"Solution count: {model.SolCount}")
            print(f"Best objective value: {model.ObjVal}")
            
            # Get the best solution found
            model.setParam(GRB.Param.SolutionNumber, 0)
            
            # Extract the best solution
            x_sol = model.getAttr('Xn', x)
            x_matrix = [[int(x_sol[i, j]) for j in range(n)] for i in range(m)]
            y_sol = model.getAttr('Xn', y)
            y_matrix = [[[int(y_sol[i, j1, j2]) for j2 in range(locations)] 
                        for j1 in range(locations)] for i in range(m)]
            distance_values = [Distance[i].Xn for i in range(m)]
            max_dist_val = max(distance_values)
            
            solution = {
                "objective": model.ObjVal,
                "x": x_matrix,
                "y": y_matrix,
                "tour_distance": distance_values,
                "max_dist": max_dist_val
            }
            print("Solution extracted successfully")
            return solution
            
        return None
        
    except Exception as e:
        print(f"Error extracting solution: {e}")
        import traceback
        traceback.print_exc()
        return None

def diagnose_infeasibility(model):
    """Diagnose why the model is infeasible."""
    try:
        model.computeIIS()
        print("\nThe following constraints are causing the model to be infeasible:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"Infeasible constraint: {c.constrName}")
    except Exception as e:
        print(f"Error diagnosing infeasibility: {e}")

def save_solution(solution, input_file, m, n, model, locations, solver_name='gurobipy', time_limit=300):
    """Save the solution to a JSON file."""
    try:
        print("\nSaving solution...")
        instance_number = input_file.split('/')[-1].split('.')[0].replace('inst', '')
        output_file = f"res/MIP/{instance_number}.json"
        print(f"Output file will be: {output_file}")
        
        # Create solution dictionary
        if solution is None or solution['objective'] is None:
            solver_solution_dict = {
                "time": time_limit,
                "optimal": False,
                "obj": None,
                "sol": []
            }
        else:
            solver_solution_dict = {
                "time": time_limit if model.Status != GRB.OPTIMAL else math.floor(model.Runtime),
                "optimal": model.Status == GRB.OPTIMAL,
                "obj": int(solution['max_dist']),
                "sol": []
            }
            
            # Extract routes
            for courier in range(m):
                route = []
                current_location = locations - 1  # Start at depot
                while True:
                    next_location = None
                    for j in range(locations):
                        if solution['y'][courier][current_location][j] == 1:
                            if j != locations - 1:  # If not returning to depot
                                route.append(j + 1)  # Add location (1-indexed)
                            next_location = j
                            break
                    
                    if next_location is None or next_location == locations - 1:
                        break
                    current_location = next_location
                
                solver_solution_dict["sol"].append(route)
        
        # Read existing solutions or create new dictionary
        try:
            with open(output_file, 'r') as f:
                all_solutions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_solutions = {}
        
        # Add this solution
        all_solutions[solver_name] = solver_solution_dict
        
        # Save all solutions
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_solutions, f, indent=4)
        
        print(f"Solution saved to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error saving solution: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir('.')}")
        import traceback
        traceback.print_exc()
        raise

def debug(x, y, m, n, s, l, D):
    """Debug function to check solution validity."""
    utils.picked_up_objects(x, m, n)
    # Check load sizes
    print(utils.check_load_sizes(x, s, m, n, l))
    # Check start and end
    print(f"All couriers start at the origin: {utils.check_if_every_courier_starts_at_origin(y, n, m)}")
    print(f"All couriers end at the origin: {utils.check_if_every_courier_ends_at_origin(y, n, m)}")
    # Check if all items are being taken
    print(f"All items are being taken by a courier: {utils.check_if_items_are_taken_by_couriers(x, m, n)}")
    # Print distances
    print("Distances calculation for each courier:")
    utils.distances_check(D, y)

def main():
    parser = argparse.ArgumentParser(description="Run Gurobi solver on a range of instances.")
    parser.add_argument("start", type=int, help="Start of the instance range (inclusive).")
    parser.add_argument("end", type=int, help="End of the instance range (inclusive).")
    parser.add_argument("--model", type=str, default="gurobi_base_bounded_second", 
                       help="Model name for the solution file")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs("res/MIP", exist_ok=True)
    print(f"Created/verified output directory: res/MIP")

    time_limit = 300  # Time limit of 5 minutes
    
    try:
        # Create a single Gurobi environment to be shared across all models
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 1)
        env.start()
        
        for i in range(args.start, args.end + 1):
            file_path = f'instances/inst{i:02d}.dat'
            print(f"\nProcessing instance {file_path}")
            
            try:
                # Read instance data
                m, n, l, s, D, locations = utils.read_input(file_path)
                print(f"Read instance data: m={m}, n={n}, locations={locations}")
                
                # Create and solve the model
                model = create_mcp_model(env, m, n, l, s, D, locations)
                if model is None:
                    print(f"Error: Could not create model for instance {i}")
                    continue
                
                # Extract solution
                solution = extract_solution(model, time_limit)
                
                # Save solution
                output_file = save_solution(solution, file_path, m, n, model, locations, args.model, time_limit)
                
                if solution:
                    debug(solution['x'], solution['y'], m, n, s, l, D)
                
                # Explicitly dispose of the model
                model.dispose()
                
            except Exception as e:
                print(f"Error solving instance {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                
    finally:
        # Always clean up the environment
        if 'env' in locals():
            env.dispose()
            print("\nGurobi environment cleaned up")

if __name__ == "__main__":
    main()