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
from dataclasses import dataclass
from enum import Enum

def seconds_to_milliseconds(seconds: int):
    return seconds * 1000

def minutes_to_milliseconds(minutes: int):
    return minutes * 60 * 1000

TIMEOUT_TIME = minutes_to_milliseconds(5)  # 5 minutes in milliseconds

class Status(Enum):
    OPTIMAL_SOLUTION = "OPTIMAL_SOLUTION"
    FEASIBLE_SOLUTION = "FEASIBLE_SOLUTION"
    INFEASIBLE = "INFEASIBLE"

@dataclass
class Solution:
    x: list
    y: list

@dataclass
class Result:
    solution: Solution = None
    status: Status = None
    objective: int = None
    statistics: dict = None

class SMTMultipleCouriersSolver:
    def __init__(self, input_file):
        self.parse_input(input_file)
        self.solver = z3.Solver()
        self.solver.set("timeout", TIMEOUT_TIME)
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
        x = self.create_assignment_variables()
        y = self.create_routing_variables()
        courier_distances = self.create_distance_variables()
        max_distance = z3.Int('max_distance')
        self.add_load_constraints(x)
        self.add_item_assignment_constraints(x)
        self.link_assignment_and_routing(x, y)
        self.add_depot_constraints(x, y)
        self.add_flow_conservation_constraints(y)
        self.add_subtour_elimination_constraints(y)
        self.calculate_distance_and_objective(courier_distances, y, max_distance)
        self.add_no_self_loops_constraint(y)
        self.add_bound_constraints(max_distance)
        return x, y, max_distance

    def create_assignment_variables(self):
        return [[z3.Bool(f'assign_{i}_{j}') for j in range(self.num_items)] 
                for i in range(self.num_couriers)]

    def create_routing_variables(self):
        return [[[z3.Bool(f'route_{i}_{j}_{k}') 
                  for k in range(self.num_items + 1)]  
                  for j in range(self.num_items + 1)]
                  for i in range(self.num_couriers)]

    def create_distance_variables(self):
        return [z3.Int(f'distance_{i}') for i in range(self.num_couriers)]

    def add_load_constraints(self, x):
        for i in range(self.num_couriers):
            load_sum = z3.Sum([z3.If(x[i][j], self.item_sizes[j], 0) for j in range(self.num_items)])
            self.solver.add(load_sum <= self.courier_load_limits[i])

    def add_item_assignment_constraints(self, x):
        for j in range(self.num_items):
            self.solver.add(z3.Sum([z3.If(x[i][j], 1, 0) for i in range(self.num_couriers)]) == 1)

    def link_assignment_and_routing(self, x, y):
        for i in range(self.num_couriers):
            for j in range(self.num_items):
                self.solver.add(z3.Implies(x[i][j],
                    z3.Sum([z3.If(y[i][k][j], 1, 0) for k in range(self.num_items + 1)]) == 1))
                self.solver.add(z3.Implies(x[i][j],
                    z3.Sum([z3.If(y[i][j][k], 1, 0) for k in range(self.num_items + 1)]) == 1))
                self.solver.add(z3.Implies(z3.Not(x[i][j]),
                    z3.Sum([z3.If(y[i][k][j], 1, 0) for k in range(self.num_items + 1)]) == 0))
                self.solver.add(z3.Implies(z3.Not(x[i][j]),
                    z3.Sum([z3.If(y[i][j][k], 1, 0) for k in range(self.num_items + 1)]) == 0))

    def add_depot_constraints(self, x, y):
        depot = self.num_items  # depot index
        for i in range(self.num_couriers):
            courier_used = z3.Or([x[i][j] for j in range(self.num_items)])
            self.solver.add(z3.Sum([z3.If(y[i][depot][j], 1, 0) for j in range(self.num_items)]) == 
                           z3.If(courier_used, 1, 0))
            self.solver.add(z3.Sum([z3.If(y[i][j][depot], 1, 0) for j in range(self.num_items)]) == 
                           z3.If(courier_used, 1, 0))

    def add_flow_conservation_constraints(self, y):
        for i in range(self.num_couriers):
            for j in range(self.num_items + 1):
                incoming = z3.Sum([z3.If(y[i][k][j], 1, 0) for k in range(self.num_items + 1)])
                outgoing = z3.Sum([z3.If(y[i][j][k], 1, 0) for k in range(self.num_items + 1)])
                self.solver.add(incoming == outgoing)

    def add_subtour_elimination_constraints(self, y):
        u = [[z3.Int(f'u_{i}_{j}') for j in range(self.num_items + 1)] 
             for i in range(self.num_couriers)]
        depot = self.num_items
        for i in range(self.num_couriers):
            self.solver.add(u[i][depot] == 0)
            for j in range(self.num_items):
                self.solver.add(u[i][j] >= 1)
                self.solver.add(u[i][j] <= self.num_items)
                for k in range(self.num_items):
                    if j != k:
                        self.solver.add(z3.Implies(y[i][j][k], u[i][k] >= u[i][j] + 1))

    def calculate_distance_and_objective(self, courier_distances, y, max_distance):
        for i in range(self.num_couriers):
            dist_terms = [z3.If(y[i][j][k], self.distance_matrix[j][k], 0)
                          for j in range(self.num_items + 1)
                          for k in range(self.num_items + 1)]
            self.solver.add(courier_distances[i] == z3.Sum(dist_terms))
            self.solver.add(courier_distances[i] <= max_distance)

        # Force max_distance to be equal to the maximum courier distance
        self.solver.add(z3.Or([max_distance == courier_distances[i] 
                              for i in range(self.num_couriers)]))

    def add_no_self_loops_constraint(self, y):
        for i in range(self.num_couriers):
            for j in range(self.num_items + 1):
                self.solver.add(z3.Not(y[i][j][j]))

    def add_bound_constraints(self, max_distance):
        self.solver.add(max_distance <= self.UB)
        self.solver.add(max_distance >= self.LB)

    def save_solution_by_model(self, input_file, m, n, model_name, time_limit=300, result=None):
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
            solution_data["y"] = [[[1 if is_true(y_val) else 0 for y_val in row] for row in courier] for courier in solution.y]
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

    def solve(self, timeout_ms):
        start_time = time.time()
        self.set_timeout(timeout_ms)
        x, y, max_distance = self.create_smt_model()
        best_solution, best_objective = self.find_best_solution(start_time, x, y, max_distance)
        result_obj = self.create_result_object(best_solution, best_objective, start_time)
        return result_obj

    def set_timeout(self, timeout_ms):
        if timeout_ms is not None:
            self.solver.set("timeout", timeout_ms)

    def find_best_solution(self, start_time, x, y, max_distance):
        print(f"\n***Now entering find_best_solution***\n")
        best_solution = None
        best_objective = float('inf')
        time_limit = 300  # 5 minutes in seconds

        while True:
            # Check if time limit has been reached
            current_time = time.time()
            elapsed_time = current_time - start_time
            remaining_time = time_limit - elapsed_time
            
            print(f"\rElapsed: {elapsed_time:.1f}s, Remaining: {remaining_time:.1f}s", end='', flush=True)
            
            if elapsed_time >= time_limit:
                print("\nTime limit reached!")
                break

            # Set remaining time for solver
            solver_timeout = int(remaining_time * 1000)  # Convert to milliseconds
            self.solver.set("timeout", solver_timeout)
            print(f"\nSetting solver timeout to: {solver_timeout/1000}manus")
            
            result = self.solver.check()

            if result == z3.sat:
                model = self.solver.model()
                current_objective = model.evaluate(max_distance).as_long()
                
                print(f"\nFound solution with objective: {current_objective}")
                
                # Save this solution
                solution = self.extract_solution(model, x, y)
                best_solution = solution
                best_objective = current_objective
                
                # Add constraint for next iteration to find better solution
                self.solver.add(max_distance < current_objective)
                
                # Check if we've reached the lower bound (optimal solution)
                if current_objective <= self.LB:
                    print("\nReached lower bound - solution is optimal!")
                    break
            else:  # z3.unsat or z3.unknown
                print(f"\nSolver result: {result}")
                break

        return best_solution, best_objective

    def extract_solution(self, model, x, y):
        solution = Solution(
            x=[[model.evaluate(x[i][j]) for j in range(self.num_items)] for i in range(self.num_couriers)],
            y=[[[model.evaluate(y[i][j][k]) for k in range(self.num_items + 1)] for j in range(self.num_items + 1)] for i in range(self.num_couriers)]
        )
        return solution

    def create_result_object(self, best_solution, best_objective, start_time):
        result_obj = Result()
        solve_time = time.time() - start_time
        
        if best_solution is None:
            result_obj.status = Status.INFEASIBLE
            print("No solution found")
        else:
            result_obj.solution = best_solution
            result_obj.objective = best_objective
            result_obj.statistics = {'solveTime': timedelta(seconds=math.floor(solve_time))}
            
            # Check if we hit the time limit
            if solve_time >= minutes_to_milliseconds(5) / 1000:
                result_obj.status = Status.FEASIBLE_SOLUTION
                print("Found feasible solution (time limit reached)")
            else:
                result_obj.status = Status.OPTIMAL_SOLUTION
                print("Found optimal solution")
        
        return result_obj

def main():
    os.makedirs("res/SMT", exist_ok=True)
    for i in range(1, 22):
        file_path = f'instances/inst{i:02d}.dat'
        print(f"\nProcessing instance {file_path}")
        try:
            start_time = time.time()
            solver = SMTMultipleCouriersSolver(file_path)
            timeout = TIMEOUT_TIME  # 5 minutes (300000 milliseconds)
            print("\tSolving... ", end='', flush=False)
            def print_progress():
                while True:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        break
                    print(f"\rSolving... {elapsed:.1f}s", end='', flush=False)
                    time.sleep(1)
            import threading
            progress_thread = threading.Thread(target=print_progress)
            progress_thread.daemon = True
            progress_thread.start()
            result = solver.solve(timeout_ms=timeout)
            elapsed_time = time.time() - start_time
            print(f"\rCompleted in {elapsed_time:.1f}s")
            if result.status.name == 'OPTIMAL_SOLUTION':
                print(f"Status: Optimal solution found")
                print(f"Objective value: {result.objective}")
            elif result.status.name == 'FEASIBLE':
                print(f"Status: Sub-optimal solution found")
                print(f"Objective value: {result.objective}")
            else:
                print(f"Status: No solution found within time limit")
            output_file = solver.save_solution_by_model(
                input_file=file_path,
                m=solver.num_couriers,
                n=solver.num_items,
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