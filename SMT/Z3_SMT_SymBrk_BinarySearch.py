import z3
import numpy as np
import time
from utils import minutes_to_milliseconds, seconds_to_milliseconds, milliseconds_to_seconds, Solution, Status, Result
from datetime import timedelta
import math, os, json
class Z3_SMT_SymBrk_BinarySearch: # name: z3_smt_symbrk_binarysearch
    def __init__(self, input_file, timeout_time):
        self.parse_input(input_file)
        self.solver = z3.Solver()
        self.timeout_time = timeout_time
        self.solver.set("timeout", timeout_time)
        self.LB, self.UB = self.find_boundaries_hybrid()

    def parse_input(self, input_file):
        with open(input_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        if len(lines) < 4:
            raise ValueError("The input file is incomplete.")
        self.num_couriers = int(lines[0])
        self.num_items = int(lines[1])
        self.courier_load_limits = list(map(int, lines[2].split()))
        self.item_sizes = list(map(int, lines[3].split()))
        self.distance_matrix = [
            list(map(int, line.split())) 
            for line in lines[4:4 + self.num_items + 1]
        ]

    def find_boundaries_hybrid(self):
        distances = np.array(self.distance_matrix)
        n = self.num_items
        min_dist_dep_list = [distances[n, i] + distances[i, n] for i in range(n)]
        LB = max(min_dist_dep_list)
        current = n  # Start at origin
        unvisited = set(range(n))
        UB = 0
        while unvisited:
            if len(unvisited) == 1:
                next_point = unvisited.pop()
            else:
                next_point = min(unvisited, key=lambda x: distances[current, x])
                unvisited.remove(next_point)
            UB += distances[current, next_point]
            current = next_point
        UB += distances[current, n]
        return int(LB), int(UB)

    def create_smt_model(self):
        print("Starting create_smt_model")
        
        # Create all variables first
        x = self.create_assignment_variables()
        y = self.create_routing_variables()
        courier_distances = self.create_distance_variables()
        max_distance = z3.Int('max_distance')
        
        # Collect all constraints
        all_constraints = []
        
        # Add constraints in batches
        all_constraints.extend(self.add_load_constraints(x))
        all_constraints.extend(self.add_item_assignment_constraints(x))
        self.link_assignment_and_routing(x, y)
        all_constraints.extend(self.add_depot_constraints(x, y))
        all_constraints.extend(self.add_flow_conservation_constraints(y))
        all_constraints.extend(self.add_subtour_elimination_constraints(y))
        self.calculate_distance_and_objective(courier_distances, y, max_distance)
        all_constraints.extend(self.add_symmetry_breaking_constraints(x))
        all_constraints.extend(self.add_bound_constraints(max_distance))
        
        # Add all constraints at once
        self.solver.add(z3.And(all_constraints))
        print("Constraints added")
        return x, y, max_distance

    def create_assignment_variables(self):
        return [[z3.Bool(f'assign_{i}_{j}') for j in range(self.num_items)] 
                for i in range(self.num_couriers)]

    def create_routing_variables(self):
        # Instead of triple nested list comprehension
        y = {}  # Use dictionary for sparse representation
        for i in range(self.num_couriers):
            for j in range(self.num_items + 1):
                for k in range(self.num_items + 1):
                    if j != k:  # Only create necessary variables
                        y[i,j,k] = z3.Bool(f'route_{i}_{j}_{k}')
        return y

    def create_distance_variables(self):
        return [z3.Int(f'distance_{i}') for i in range(self.num_couriers)]

    def add_load_constraints(self, x):
        constraints = []
        for i in range(self.num_couriers):
            load_sum = z3.Sum([z3.If(x[i][j], self.item_sizes[j], 0) 
                          for j in range(self.num_items)])
            constraints.append(load_sum <= self.courier_load_limits[i])
        self.solver.add(z3.And(constraints))  # Add all constraints at once
        return constraints

    def add_item_assignment_constraints(self, x):
        constraints = []
        for j in range(self.num_items):
            constraints.append(z3.Sum([z3.If(x[i][j], 1, 0) for i in range(self.num_couriers)]) == 1)
        return constraints  # Return the constraints instead of adding them directly

    def link_assignment_and_routing(self, x, y):
        for i in range(self.num_couriers):
            for j in range(self.num_items):
                # Create lists of valid y variables for each constraint
                incoming = [y[i,k,j] for k in range(self.num_items + 1) if k != j]
                outgoing = [y[i,j,k] for k in range(self.num_items + 1) if k != j]
                
                self.solver.add(z3.Implies(x[i][j],
                    z3.Sum([z3.If(y_val, 1, 0) for y_val in incoming]) == 1))
                self.solver.add(z3.Implies(x[i][j],
                    z3.Sum([z3.If(y_val, 1, 0) for y_val in outgoing]) == 1))
                self.solver.add(z3.Implies(z3.Not(x[i][j]),
                    z3.Sum([z3.If(y_val, 1, 0) for y_val in incoming]) == 0))
                self.solver.add(z3.Implies(z3.Not(x[i][j]),
                    z3.Sum([z3.If(y_val, 1, 0) for y_val in outgoing]) == 0))

    def add_depot_constraints(self, x, y):
        constraints = []
        depot = self.num_items  # depot index
        for i in range(self.num_couriers):
            courier_used = z3.Or([x[i][j] for j in range(self.num_items)])
            constraints.append(
                z3.Sum([z3.If(y[i,depot,j], 1, 0) for j in range(self.num_items)]) == 
                z3.If(courier_used, 1, 0))
            constraints.append(
                z3.Sum([z3.If(y[i,j,depot], 1, 0) for j in range(self.num_items)]) == 
                z3.If(courier_used, 1, 0))
        return constraints

    def add_flow_conservation_constraints(self, y):
        constraints = []
        for i in range(self.num_couriers):
            for j in range(self.num_items + 1):
                # Access y using tuple keys (i,k,j) instead of nested indexing
                incoming = z3.Sum([z3.If(y[i,k,j], 1, 0) 
                                 for k in range(self.num_items + 1) if k != j])
                outgoing = z3.Sum([z3.If(y[i,j,k], 1, 0) 
                                 for k in range(self.num_items + 1) if k != j])
                constraints.append(incoming == outgoing)
        return constraints

    def add_subtour_elimination_constraints(self, y):
        # This creates O(n³) constraints for n items
        u = [[z3.Int(f'u_{i}_{j}') for j in range(self.num_items + 1)] 
             for i in range(self.num_couriers)]
        depot = self.num_items
        constraints = []
        
        for i in range(self.num_couriers):
            constraints.append(u[i][depot] == 0)
            for j in range(self.num_items):
                constraints.append(u[i][j] >= 1)
                constraints.append(u[i][j] <= self.num_items)
                for k in range(self.num_items):
                    if j != k:
                        # Use dictionary access with tuple key instead of nested indexing
                        constraints.append(z3.Implies(y[i,j,k], u[i][k] >= u[i][j] + 1))
        
        self.solver.add(z3.And(constraints))
        return constraints

    def calculate_distance_and_objective(self, courier_distances, y, max_distance):
        for i in range(self.num_couriers):
            # Use dictionary access with tuple keys
            dist_terms = [z3.If(y[i,j,k], self.distance_matrix[j][k], 0)
                          for j in range(self.num_items + 1)
                          for k in range(self.num_items + 1)
                          if j != k]  # Only consider valid routes
            self.solver.add(courier_distances[i] == z3.Sum(dist_terms))
            self.solver.add(courier_distances[i] <= max_distance)

        # Force max_distance to be equal to the maximum courier distance
        self.solver.add(z3.Or([max_distance == courier_distances[i] 
                              for i in range(self.num_couriers)]))

    def add_no_self_loops_constraint(self, y):
        for i in range(self.num_couriers):
            for j in range(self.num_items + 1):
                self.solver.add(z3.Not(y[i][j][j]))

    def add_symmetry_breaking_constraints(self, x):
        constraints = []
        # Basic courier ordering
        for i in range(self.num_couriers - 1):
            constraints.append(
                z3.Implies(
                    z3.Or([x[i+1][j] for j in range(self.num_items)]),
                    z3.Or([x[i][j] for j in range(self.num_items)])
                )
            )
        
        # Load-based ordering (simplified)
        courier_loads = [
            z3.Sum([z3.If(x[i][j], self.item_sizes[j], 0) 
                    for j in range(self.num_items)])
            for i in range(self.num_couriers)
        ]
        
        for i in range(self.num_couriers - 1):
            constraints.append(courier_loads[i] >= courier_loads[i+1])
        
        self.solver.add(z3.And(constraints))
        return constraints

    def add_bound_constraints(self, max_distance):
        constraints = []
        constraints.append(max_distance <= self.UB)
        constraints.append(max_distance >= self.LB)
        self.solver.add(z3.And(constraints))
        return constraints  # Return the constraints

    def save_solution_by_model(self, input_file, m, n, model_name, time_limit, result=None):
        instance_number = input_file.split('/')[-1].split('.')[0].replace('inst', '')
        output_dir = "res/SMT"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{instance_number}.json")
        if result is None or result.solution is None:
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
        solution_data = {"y": [[[0 for _ in range(n+1)] for _ in range(n+1)] for _ in range(m)]}
        if hasattr(solution, "y"):
            solution_data["y"] = [[[1 if str(model_val) == "True" else 0 for model_val in row] for row in courier] for courier in solution.y]
        solver_solution_dict = {
            "time": time_limit if not optimal else math.floor(result.statistics['solveTime'].total_seconds()),
            "optimal": optimal,
            "obj": objective,
            "sol": []
        }
        if hasattr(solution, "y"):
            for courier in range(m):
                route = []
                current_location = n
                while True:
                    next_location = None
                    for j2 in range(n+1):
                        if solution_data["y"][courier][current_location][j2] == 1:
                            next_location = j2
                            break
                    if next_location is None or next_location == n:
                        break
                    route.append(next_location + 1)
                    current_location = next_location
                solver_solution_dict["sol"].append(route)
        else:
            solver_solution_dict["sol"] = []
        try:
            with open(output_file, 'r') as infile:
                existing_solutions = json.load(infile)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_solutions = {}
        existing_solutions[model_name] = solver_solution_dict
        with open(output_file, 'w') as outfile:
            json.dump(existing_solutions, outfile, indent=4)
        return output_file

    def set_timeout(self, timeout_ms):
        if timeout_ms is not None:
            self.solver.set("timeout", timeout_ms)

    def solve(self, timeout_ms):
        print("Starting solve function")
        start_time = time.time()
        print(f"Start time set: {start_time}")
        
        self.set_timeout(timeout_ms)
        print(f"Timeout set to: {timeout_ms}ms")
        
        x, y, max_distance = self.create_smt_model()
        print("SMT model created")
        
        best_solution, best_objective, low, high = self.find_best_solution(start_time, x, y, max_distance)
        print(f"Best solution found with objective: {best_objective}")
        
        result_obj = self.create_result_object(best_solution, best_objective, start_time, low, high)
        print(f"Result object created with status: {result_obj.status}")
        
        return result_obj

    def find_best_solution(self, start_time, x, y, max_distance):
        print("\n=== Starting Binary Search Optimization ===")
        print(f"Initial LB: {self.LB}, UB: {self.UB}")
        
        best_solution = None
        print("Best solution initialized to None")
        
        best_objective = self.UB
        print(f"Best objective initialized to UB: {self.UB}")
        
        low = self.LB
        high = self.UB
        print(f"Search range initialized: [{low}, {high}]")
        
        while low <= high:
            print(f"\nCurrent search range: [{low}, {high}]")
            
            elapsed_time = time.time() - start_time
            print(f"Elapsed time: {elapsed_time:.2f}s")
            
            if elapsed_time >= milliseconds_to_seconds(self.timeout_time):
                print("❌ Time limit reached - stopping search")
                break
                
            mid = (low + high) // 2
            print(f"\n--- Trying objective value: {mid} ---")
            
            self.solver.push()
            print("Solver state saved")
            
            self.solver.add(max_distance <= mid)
            print(f"Added constraint: max_distance <= {mid}")
            print("Solver check called")
            result = self.solver.check()
            print(f"Solver result: {result}")
            
            if result == z3.sat:
                print("Solution found")
                model = self.solver.model()
                print("Model extracted")
                
                solution = self.extract_solution(model, x, y)
                print("Solution extracted from model")
                
                best_solution = solution
                best_objective = mid
                print(f"✅ Updated best solution with objective: {mid}")
                
                high = mid - 1
                print(f"Updated high bound to: {high}")
            else:
                print("❌ No solution found at this value")
                low = mid + 1
                print(f"Updated low bound to: {low}")
            
            self.solver.pop()
            print("Solver state restored")
        
        print("\n=== Binary Search complete ===")
        print(f"Final best objective: {best_objective}")
        print(f"Solution found: {'Yes' if best_solution is not None else 'No'}")
        print(f"Search range at end: [{low}, {high}]")
        
        return best_solution, best_objective, low, high

    def extract_solution(self, model, x, y):
        # x remains the same as it's a nested list
        x_sol = [[model.evaluate(x[i][j]) for j in range(self.num_items)] 
                 for i in range(self.num_couriers)]
        
        # Create a nested list for y from the dictionary representation
        y_sol = [[[model.evaluate(y.get((i,j,k), z3.BoolVal(False))) 
                   for k in range(self.num_items + 1)]
                  for j in range(self.num_items + 1)]
                 for i in range(self.num_couriers)]
        
        solution = Solution(x=x_sol, y=y_sol)
        return solution

    def create_result_object(self, best_solution, best_objective, start_time, low, high):
        result_obj = Result()
        solve_time = time.time() - start_time
        
        if best_solution is None:
            result_obj.status = Status.INFEASIBLE
            print("No solution found")
        else:
            result_obj.solution = best_solution
            result_obj.objective = best_objective
            result_obj.statistics = {'solveTime': timedelta(seconds=math.floor(solve_time))}
            
            # Only mark as optimal if binary search converged (low > high) and we didn't hit timeout
            if solve_time < milliseconds_to_seconds(self.timeout_time) and low > high:
                result_obj.status = Status.OPTIMAL_SOLUTION
                print("Found optimal solution (binary search converged)")
            else:
                result_obj.status = Status.FEASIBLE_SOLUTION
                print("Found feasible solution (binary search did not converge or time limit reached)")
        
        return result_obj

    def add_constraints_in_batches(self, constraint_generator, batch_size=1000):
        constraints = []
        for constraint in constraint_generator:
            constraints.append(constraint)
            if len(constraints) >= batch_size:
                self.solver.add(z3.And(constraints))
                constraints = []
        if constraints:
            self.solver.add(z3.And(constraints))

