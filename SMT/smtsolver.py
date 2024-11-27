import z3
import json
import os
import sys
import time
import numpy as np
import traceback
import math
from datetime import timedelta
from z3 import is_true

class SMTMultipleCouriersSolver:
    def __init__(self, input_file):
        # Parse input file
        with open(input_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        # Parsing first line: number of couriers
        if len(lines) < 1:
            raise ValueError("The input file is empty.")
        self.m_couriers = int(lines[0])
        
        # Parsing second line: number of items
        if len(lines) < 2:
            raise ValueError("Missing line specifying the number of items.")
        self.n_items = int(lines[1])
        
        # Parsing third line: courier load limits
        if len(lines) < 3:
            raise ValueError("Missing line specifying courier load limits.")
        self.courier_load_limits = list(map(int, lines[2].split()))
        if len(self.courier_load_limits) != self.m_couriers:
            raise ValueError(f"Expected {self.m_couriers} load limits, found {len(self.courier_load_limits)}.")
        
        # Parsing fourth line: item sizes
        if len(lines) < 4:
            raise ValueError("Missing line specifying item sizes.")
        self.item_sizes = list(map(int, lines[3].split()))
        if len(self.item_sizes) != self.n_items:
            raise ValueError(f"Expected {self.n_items} item sizes, found {len(self.item_sizes)}.")
        
        # Parsing distance matrix
        if len(lines) < 4 + self.n_items + 1:
            raise ValueError(f"Expected at least {4 + self.n_items + 1} lines for the distance matrix.")
        
        self.distance_matrix = [
            list(map(int, line.split())) 
            for line in lines[4:4 + self.n_items + 1]
        ]
        if len(self.distance_matrix) != self.n_items + 1:
            raise ValueError(f"Expected {self.n_items + 1} rows in the distance matrix, found {len(self.distance_matrix)}.")
        for row in self.distance_matrix:
            if len(row) != self.n_items + 1:
                raise ValueError(f"Malformed distance matrix row: expected {self.n_items + 1} elements, found {len(row)}.")

        # Initialize the Z3 optimizer
        self.solver = z3.Optimize()
        self.solver.set("timeout", 1)
        
        # Calculate bounds right after initialization
        self.LB, self.UB = self.find_boundaries_hybrid()

    def find_boundaries_hybrid(self):
        """
        Calculate bounds for MCP returning single LB and UB
        Returns:
            Tuple of (LB, UB)
        """
        distances = np.array(self.distance_matrix)
        n = self.n_items
        
        # Lower Bound: Maximum round-trip
        min_dist_dep_list = []
        for i in range(n):
            min_dist_dep_list.append(distances[n, i] + distances[i, n])
        LB = max(min_dist_dep_list)
        
        # Upper Bound: Enhanced nearest neighbor
        current = n  # Start at origin
        unvisited = set(range(n))
        UB = 0
        
        while unvisited:
            # Find nearest unvisited point
            if len(unvisited) == 1:
                next_point = unvisited.pop()
            else:
                next_point = min(unvisited, 
                               key=lambda x: distances[current, x])
                unvisited.remove(next_point)
            
            # Add distance to next point
            UB += distances[current, next_point]
            current = next_point
        
        # Return to origin
        UB += distances[current, n]
        UB = int(UB)
        
        return LB, UB

    def create_smt_model(self):
        # Boolean assignment variables
        x = [[z3.Bool(f'assign_{i}_{j}') for j in range(self.n_items)] 
             for i in range(self.m_couriers)]
        
        # Boolean routing variables (1 if courier i travels from point j to k)
        y = [[[z3.Bool(f'route_{i}_{j}_{k}') 
               for k in range(self.n_items + 1)]  # +1 for depot
               for j in range(self.n_items + 1)]
               for i in range(self.m_couriers)]
        
        # Distance variables
        courier_distances = [z3.Int(f'distance_{i}') for i in range(self.m_couriers)]
        max_distance = z3.Int('max_distance')
        
        # 1. Load constraints - respect maximum load size for each courier
        for i in range(self.m_couriers):
            self.solver.add(
                z3.Sum([z3.If(x[i][j], self.item_sizes[j], 0) for j in range(self.n_items)]) 
                <= self.courier_load_limits[i]
            )
        
        # 2. Each item must be delivered by exactly one courier
        for j in range(self.n_items):
            self.solver.add(z3.Sum([z3.If(x[i][j], 1, 0) for i in range(self.m_couriers)]) == 1)
        
        depot = self.n_items  # depot index
        
        # 3. Link assignment and routing variables
        for i in range(self.m_couriers):
            for j in range(self.n_items):
                # If item j is assigned to courier i, it must be visited in the route
                self.solver.add(z3.Implies(x[i][j],
                    z3.Sum([z3.If(y[i][k][j], 1, 0) for k in range(self.n_items + 1)]) == 1))
                self.solver.add(z3.Implies(x[i][j],
                    z3.Sum([z3.If(y[i][j][k], 1, 0) for k in range(self.n_items + 1)]) == 1))
                # If item j is not assigned to courier i, it must not be visited
                self.solver.add(z3.Implies(z3.Not(x[i][j]),
                    z3.Sum([z3.If(y[i][k][j], 1, 0) for k in range(self.n_items + 1)]) == 0))
                self.solver.add(z3.Implies(z3.Not(x[i][j]),
                    z3.Sum([z3.If(y[i][j][k], 1, 0) for k in range(self.n_items + 1)]) == 0))
        
        # 4. Start and end at depot constraints
        for i in range(self.m_couriers):
            # Must leave depot exactly once if courier is used
            courier_used = z3.Bool(f'courier_used_{i}')
            self.solver.add(z3.Sum([z3.If(y[i][depot][j], 1, 0) for j in range(self.n_items)]) == 
                           z3.If(courier_used, 1, 0))
            # Must return to depot exactly once if courier is used
            self.solver.add(z3.Sum([z3.If(y[i][j][depot], 1, 0) for j in range(self.n_items)]) == 
                           z3.If(courier_used, 1, 0))
            # Courier is used if it has any assignments
            self.solver.add(courier_used == 
                           z3.Or([x[i][j] for j in range(self.n_items)]))
        
        # 5. Flow conservation
        for i in range(self.m_couriers):
            for j in range(self.n_items + 1):
                self.solver.add(
                    z3.Sum([z3.If(y[i][k][j], 1, 0) for k in range(self.n_items + 1)]) ==
                    z3.Sum([z3.If(y[i][j][k], 1, 0) for k in range(self.n_items + 1)])
                )
        
        # 6. Subtour elimination using MTZ formulation
        u = [[z3.Int(f'u_{i}_{j}') for j in range(self.n_items + 1)] 
             for i in range(self.m_couriers)]
        
        for i in range(self.m_couriers):
            # Initialize depot position
            self.solver.add(u[i][depot] == 0)
            
            for j in range(self.n_items):
                # Position constraints
                self.solver.add(u[i][j] >= 1)
                self.solver.add(u[i][j] <= self.n_items)
                
                # MTZ constraints
                for k in range(self.n_items):
                    if j != k:
                        self.solver.add(z3.Implies(y[i][j][k],
                            u[i][k] >= u[i][j] + 1))
        
        # 7. Distance calculation and objective
        for i in range(self.m_couriers):
            dist_terms = []
            for j in range(self.n_items + 1):
                for k in range(self.n_items + 1):
                    dist_terms.append(
                        z3.If(y[i][j][k], self.distance_matrix[j][k], 0)
                    )
            self.solver.add(courier_distances[i] == z3.Sum(dist_terms))
            self.solver.add(max_distance >= courier_distances[i])
        
        # 8. No self-loops
        for i in range(self.m_couriers):
            for j in range(self.n_items + 1):
                self.solver.add(z3.Not(y[i][j][j]))
                
        # Add symmetry breaking constraints
        for i in range(self.m_couriers - 1):
            # Force couriers to be used in order (if courier i+1 is used, courier i must be used)
            self.solver.add(z3.Implies(
                z3.Or([x[i+1][j] for j in range(self.n_items)]),
                z3.Or([x[i][j] for j in range(self.n_items)])
            ))
        # Set objective
        self.solver.minimize(max_distance)
        
        return x, y, max_distance

    def save_solution_by_model(self, input_file, m, n, model_name, time_limit=300, result=None):
        # Determine the output file path
        instance_number = input_file.split('/')[-1].split('.')[0].replace('inst', '')
        output_dir = "res/SMT"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"{instance_number}.json")
        
        # Prepare the solution dictionary
        if result is None or result.solution is None:
            # Default solution if no result or solution found
            solution = {
                "objective": None,
                "x": [[0 for _ in range(n)] for _ in range(m)],
                "y": [[[0 for _ in range(n+1)] for _ in range(n+1)] for _ in range(m)],
                "tour_distance": [0 for _ in range(m)],
                "max_dist": None
            }
            optimal = False
            objective = None
        else:
            solution = result.solution
            optimal = (result.status.name == 'OPTIMAL_SOLUTION')
            objective = result.objective if result.objective is not None else None

        # Create a dictionary to store solution details
        solution_data = {
            "y": [[[0 for _ in range(n+1)] for _ in range(n+1)] for _ in range(m)]
        }

        # Try to extract 'y' from solution if it exists
        if hasattr(solution, "y"):
            # Convert Z3 Boolean expressions to Python booleans
            solution_data["y"] = [[[1 if is_true(y_val) else 0 
                                   for y_val in row] 
                                   for row in courier] 
                                   for courier in solution.y]

        # Prepare the solver-specific solution dictionary
        solver_solution_dict = {
            "time": time_limit if result.status.name != 'OPTIMAL_SOLUTION' else math.floor(result.statistics['solveTime'].total_seconds()),
            "optimal": optimal,
            "obj": objective,
            "sol": []
        }

        # Populate the solution routes
        if hasattr(solution, "y"):
            for courier in range(m):
                route = []
                current_location = n  # Start at the origin (assuming last location is the origin)
                while True:
                    next_location = None
                    for j2 in range(n+1):
                        if solution_data["y"][courier][current_location][j2] == 1:
                            next_location = j2
                            break

                    if next_location is None or next_location == n:
                        break  # No further movement or return to origin

                    route.append(next_location + 1)
                    current_location = next_location
                
                solver_solution_dict["sol"].append(route)
        else:
            solver_solution_dict["sol"] = []

        # Read existing solutions or create new dictionary
        try:
            with open(output_file, 'r') as infile:
                existing_solutions = json.load(infile)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_solutions = {}

        # Add or update the current solver's solution
        existing_solutions[model_name] = solver_solution_dict

        # Write updated solutions back to the file
        with open(output_file, 'w') as outfile:
            json.dump(existing_solutions, outfile, indent=4)
        return output_file

    def solve(self, timeout_ms=None):
        """
        Solve the MCP problem
        Args:
            timeout_ms: timeout in milliseconds
        Returns:
            Result object with solution details
        """
        start_time = time.time()
        
        if timeout_ms is not None:
            self.solver.set("timeout", timeout_ms)
        
        x, y, max_distance = self.create_smt_model()
        result = self.solver.check()
        solve_time = math.floor(time.time() - start_time)
        
        # Create a Result object to match the expected format
        class Result:
            def __init__(self):
                self.solution = None
                self.status = None
                self.objective = None
                self.statistics = {}
        
        result_obj = Result()
        
        if result == z3.sat:
            model = self.solver.model()
            
            # Create Solution object
            class Solution:
                def __init__(self):
                    self.x = None
                    self.y = None
            
            solution = Solution()
            
            # Extract x values
            solution.x = [[model.evaluate(x[i][j]) for j in range(self.n_items)]
                         for i in range(self.m_couriers)]
            
            # Extract y values
            solution.y = [[[model.evaluate(y[i][j][k]) for k in range(self.n_items + 1)]
                          for j in range(self.n_items + 1)]
                         for i in range(self.m_couriers)]
            
            result_obj.solution = solution
            result_obj.objective = model[max_distance].as_long()
            
            # Set status
            class Status:
                def __init__(self, name):
                    self.name = name
            
            result_obj.status = Status('OPTIMAL_SOLUTION')
            result_obj.statistics = {'solveTime': timedelta(seconds=solve_time)}
            
        else:
            result_obj.status = Status('INFEASIBLE')
            
        return result_obj

def main():
    os.makedirs("res/SMT", exist_ok=True)
    
    for i in range(1, 22):
        file_path = f'instances/inst{i:02d}.dat'
        print(f"\nProcessing instance {file_path}")
        
        try:
            start_time = time.time()
            solver = SMTMultipleCouriersSolver(file_path)
            timeout = 300000  # 5 minutes (300000 milliseconds)
            
            # Print initial status
            print("Solving... ", end='', flush=True)
            
            # Start progress tracking in a separate thread
            def print_progress():
                while True:
                    elapsed = time.time() - start_time
                    if elapsed >= 300:  # Stop at timeout
                        break
                    print(f"\rSolving... {elapsed:.1f}s", end='', flush=True)
                    time.sleep(1)  # Update every second
            
            import threading
            progress_thread = threading.Thread(target=print_progress)
            progress_thread.daemon = True  # Thread will be killed when main program exits
            progress_thread.start()
            
            # Solve the instance
            result = solver.solve(timeout_ms=timeout)
            
            # Print final status
            elapsed_time = time.time() - start_time
            print(f"\rCompleted in {elapsed_time:.1f}s")
            
            # Print meaningful solution information
            if result.status.name == 'OPTIMAL_SOLUTION':
                print(f"Status: Optimal solution found")
                print(f"Objective value: {result.objective}")
            else:
                print(f"Status: No optimal solution found within time limit")
            
            # Save solution
            output_file = solver.save_solution_by_model(
                input_file=file_path,
                m=solver.m_couriers,
                n=solver.n_items,
                model_name="z3_smt_base",
                time_limit=300,
                result=result
            )
            print(f"Solution saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing instance {file_path}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()