import z3
import json
import os
import sys
import time
import numpy as np
import traceback

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
        
        # 3. Start and end at depot constraints
        for i in range(self.m_couriers):
            # Must leave depot at least once
            self.solver.add(z3.Sum([z3.If(y[i][depot][j], 1, 0) for j in range(self.n_items)]) >= 1)
            # Must return to depot at least once
            self.solver.add(z3.Sum([z3.If(y[i][j][depot], 1, 0) for j in range(self.n_items)]) >= 1)
        
        # 4. Prevent traveling from point to itself
        for i in range(self.m_couriers):
            for j in range(self.n_items + 1):
                self.solver.add(z3.Not(y[i][j][j]))
        
        # 5. If courier picks up an item, must visit its location
        for i in range(self.m_couriers):
            for j in range(self.n_items):
                # If item j is assigned to courier i
                self.solver.add(z3.Implies(x[i][j],
                    # Must enter and leave the item's location
                    z3.And(
                        z3.Sum([z3.If(y[i][k][j], 1, 0) for k in range(self.n_items + 1)]) == 1,
                        z3.Sum([z3.If(y[i][j][k], 1, 0) for k in range(self.n_items + 1)]) == 1
                    )
                ))
        
        # 6. Flow conservation constraint
        for i in range(self.m_couriers):
            for j in range(self.n_items + 1):
                # Number of incoming edges equals number of outgoing edges
                self.solver.add(
                    z3.Sum([z3.If(y[i][k][j], 1, 0) for k in range(self.n_items + 1)]) ==
                    z3.Sum([z3.If(y[i][j][k], 1, 0) for k in range(self.n_items + 1)])
                )
        
        # 7. Subtour elimination
        # Using MTZ formulation with additional variables
        u = [[z3.Int(f'u_{i}_{j}') for j in range(self.n_items + 1)] 
             for i in range(self.m_couriers)]
        
        for i in range(self.m_couriers):
            # Set bounds for position variables
            for j in range(self.n_items + 1):
                self.solver.add(u[i][j] >= 0)
                self.solver.add(u[i][j] <= self.n_items + 1)
            
            # MTZ constraints
            for j in range(self.n_items):
                for k in range(self.n_items):
                    if j != k:
                        self.solver.add(z3.Implies(y[i][j][k],
                            u[i][j] + 1 <= u[i][k] + (self.n_items + 1) * (1 - z3.If(y[i][j][k], 1, 0))))
        
        # Distance calculation
        for i in range(self.m_couriers):
            dist_terms = []
            for j in range(self.n_items + 1):
                for k in range(self.n_items + 1):
                    dist_terms.append(
                        z3.If(y[i][j][k], self.distance_matrix[j][k], 0)
                    )
            self.solver.add(courier_distances[i] == z3.Sum(dist_terms))
        
        # Objective: minimize maximum distance
        for i in range(self.m_couriers):
            self.solver.add(max_distance >= courier_distances[i])
        
        self.solver.minimize(max_distance)
        
        return x, y, max_distance
    
    def solve(self, timeout_ms=3):
        
        start_time = time.time()
        
        print(f"Bounds for objective function:")
        print(f"Lower Bound (LB) = {self.LB}")
        print(f"Upper Bound (UB) = {self.UB}")
        print(f"Gap = {self.UB - self.LB}")
        
        x, next_node, max_distance = self.create_smt_model()
        
        self.solver.set("timeout", timeout_ms)
        
        result = self.solver.check()
        if result == z3.sat:
            model = self.solver.model()
            
            solution = [[] for _ in range(self.m_couriers)]
            for i in range(self.m_couriers):
                for j in range(self.n_items):
                    if z3.is_true(model.evaluate(x[i][j])):
                        solution[i].append(j + 1)
            
            return {
                "time": int(time.time() - start_time),
                "optimal": True,
                "obj": int(model.evaluate(max_distance).as_long()),
                "sol": solution
            }
        else:
            print(f"Solver returned: {result}")
            print(f"Solver statistics: {self.solver.statistics()}")
            return {
                "time": int(time.time() - start_time),
                "optimal": False,
                "obj": None,
                "sol": None
            }

def main():
    os.makedirs("res/SMT", exist_ok=True)
    
    for i in range(1, 22):
        file_path = f'instances/inst{i:02d}.dat'
        print(f"\nProcessing instance {file_path}")
        
        try:
            solver = SMTMultipleCouriersSolver(file_path)
            # Increase timeout for larger instances
            timeout = 300000  # 10 minutes for larger instances
            result = solver.solve(timeout_ms=timeout)
            print(f"Solver result: {result}")
            
            output_file = os.path.join("res/SMT", f"{i:02d}.json")
            with open(output_file, 'w') as f:
                json.dump({"z3_smt": result}, f, indent=2)
            
            print(f"Solution saved to {output_file}")
        
        except Exception as e:
            print(f"Error processing instance {file_path}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()