import gurobipy as gp
from gurobipy import GRB, quicksum
import json
import os
import math
import utils as utils
import argparse

def create_mcp_model(m, n, l, s, D, locations):
    """Create the MCP model with the given parameters."""
    try:
        # Create a new model
        model = gp.Model("MCP")
        
        # Create variables
        x = model.addVars(m, n, vtype=GRB.BINARY, name="x")
        y = model.addVars(m, locations, locations, vtype=GRB.BINARY, name="y")
        Distance = model.addVars(m, vtype=GRB.INTEGER, name="Distance")
        max_distance = model.addVar(vtype=GRB.INTEGER, name="max_distance")
        
        # Each item must be delivered by exactly one courier
        for j in range(n):
            model.addConstr(quicksum(x[i,j] for i in range(m)) == 1)
        
        # Load capacity constraint for each courier
        for i in range(m):
            model.addConstr(quicksum(s[j] * x[i,j] for j in range(n)) <= l[i])
        
        # Flow conservation constraints
        for i in range(m):
            for j in range(1, locations):  # Skip depot
                model.addConstr(
                    quicksum(y[i,k,j] for k in range(locations) if k != j) ==
                    quicksum(y[i,j,k] for k in range(locations) if k != j)
                )
                
        # Each location must be visited if items are assigned to the courier
        for i in range(m):
            for j in range(1, locations):  # Skip depot
                model.addConstr(
                    quicksum(y[i,k,j] for k in range(locations) if k != j) ==
                    x[i,j-1]  # -1 because locations are 1-indexed
                )
        
        # Each courier must start and end at the depot
        for i in range(m):
            model.addConstr(quicksum(y[i,0,j] for j in range(1, locations)) == 
                          quicksum(x[i,j] for j in range(n)))
            model.addConstr(quicksum(y[i,j,0] for j in range(1, locations)) == 
                          quicksum(x[i,j] for j in range(n)))
        
        # Calculate total distance for each courier
        for i in range(m):
            model.addConstr(Distance[i] == quicksum(D[j][k] * y[i,j,k] 
                                                  for j in range(locations) 
                                                  for k in range(locations) if j != k))
        
        # Set max_distance
        for i in range(m):
            model.addConstr(Distance[i] <= max_distance)
        
        # Set objective
        model.setObjective(max_distance, GRB.MINIMIZE)
        
        # Store variables for solution extraction
        model._vars = {
            'x': x,
            'y': y,
            'Distance': Distance,
            'max_distance': max_distance,
            'm': m,
            'n': n,
            'locations': locations
        }
        
        return model
        
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        return None

def extract_solution(model, time_limit=300):
    """Extract solution from the optimized model."""
    try:
        if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
            vars = model._vars
            m = vars['m']
            n = vars['n']
            locations = vars['locations']
            x = vars['x']
            y = vars['y']
            Distance = vars['Distance']
            max_distance = vars['max_distance']
            
            solution = {
                "objective": model.ObjVal if model.SolCount > 0 else None,
                "x": [[int(x[i,j].X) if model.SolCount > 0 else 0 
                       for j in range(n)] for i in range(m)],
                "y": [[[int(y[i,j,k].X) if model.SolCount > 0 else 0 
                        for k in range(locations)] 
                        for j in range(locations)] 
                        for i in range(m)],
                "tour_distance": [int(Distance[i].X) if model.SolCount > 0 else 0 
                                for i in range(m)],
                "max_dist": int(max_distance.X) if model.SolCount > 0 else None
            }
            return solution
        else:
            print(f"Model status: {model.Status}")
            return None
    except Exception as e:
        print(f"Error extracting solution: {e}")
        return None

def save_solution(solution, input_file, m, n, solver_name='gurobipy', time_limit=300):
    """Save the solution to a JSON file."""
    output_file = f"res/MIP/{os.path.splitext(os.path.basename(input_file))[0]}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "solver": solver_name,
            "time_limit": time_limit,
            "solution": solution
        }, f, indent=2)
    
    print(f"Solution saved to {output_file}")
    return output_file

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Gurobi solver on a range of instances.")
    parser.add_argument("start", type=int, help="Start of the instance range (inclusive).")
    parser.add_argument("end", type=int, help="End of the instance range (inclusive).")
    parser.add_argument("--model", type=str, help="Model name (ignored for Gurobi implementation)")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs("res/MIP", exist_ok=True)

    time_limit = 300  # Time limit of 5 minutes

    for i in range(args.start, args.end + 1):
        file_path = f'instances/inst{i:02d}.dat'
        print(f"\nProcessing instance {file_path}")
        
        try:
            # Read instance data
            m, n, l, s, D, locations = utils.read_input(file_path)
            
            # Create and solve the model
            model = create_mcp_model(m, n, l, s, D, locations)
            if model is None:
                print(f"Error: Could not create model for instance {i}")
                continue
                
            # Set time limit
            model.setParam('TimeLimit', time_limit)
            
            # Optimize
            model.optimize()
            
            # Extract and save solution
            solution = extract_solution(model, time_limit)
            if solution:
                output_file = save_solution(solution, f"inst{i:02d}.dat", m, n)
                print(f"Solution saved to {output_file}")
            
        except Exception as e:
            print(f"Error solving instance {i}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
