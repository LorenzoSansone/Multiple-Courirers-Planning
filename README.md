# Multiple Couriers Planning

This project implements different approaches to solve the Multiple Couriers Planning problem using various solving techniques:
- SMT (Satisfiability Modulo Theories)
- SAT (Boolean Satisfiability)
- CP (Constraint Programming)
- MIP (Mixed Integer Programming)
![Graph for BS_Gec model from instances 01 to 09](graphs/cp/graph_cp_bs_gec_inst01_to_09.png)

## Docker Setup

To build solver images for SAT, SMT and MIP:
```bash
docker-compose build
```

To build solver images for CP
```bash
docker build . -t image_cp -f Dockerfile_cp
```

## Running the Solvers

### CP (Constraint Programming)
```bash
docker run -v ./res:/app/res image_cp <model_name> <start_instance> <end_instance>
```

Available models:
- `bs_gecode`: base CP model using Gecode solver
- `bs_chuffed`: base CP model using Chuffed solver
- `bs_heu_gecode`: CP with heuristics using Gecode solver
- `bs_heu_chuffed`:  CP with heuristics using Chuffed solver
- `bs_heu_impl_gecode`: CP with heuristics and implied constraints using Gecode solver
- `bs_heu_impl_chuffed`: CP with heuristics and implied constraints using Chuffed solver
- `bs_heu_sym_gecode`: CP with heuristics and symmetry breaking constraints using Gecode solver
- `bs_heu_sym_chuffed`: CP with heuristics and symmetry breaking constraints using Chuffed solver
- `bs_heu_sym_impl_gecode`: CP with heuristics, implied constraints and symmetry breaking constraints using Gecode solver
- `bs_heu_sym_impl_chuffed`: CP with heuristics, implied constraints and symmetry breaking constraints using Chuffed solver

Example:
```bash
docker run -v ./res:/app/res image_cp bs_gecode 1 2
```

### SAT (Boolean Satisfiability)
```bash
docker run -v ./res:/app/res mcp-sat <model_name> <start_instance> <end_instance>
```

Available models:
- `LNS`: Linear Search
- `LNS_SYB`: Linear Search with symmetry breaking constraints
- `BNS`: Binary Search
- `BNS_SYB`: Binary Search with symmetry breaking constraints

Example:
```bash
docker run -v ./res:/app/res mcp-sat LNS 1 2
```

### SMT (Satisfiability Modulo Theories)
```bash
docker run -v ./res:/app/res mcp-smt --model <model_name> <start_instance> <end_instance>
```

Available models:
- `z3_smt_base`
- `z3_smt_symbrk`
- `z3_smt_symbrk_implconstr`
- `z3_smt_symbrk_binarysearch`

Example:
```bash
docker run -v ./res:/app/res mcp-smt --model z3_smt_base 1 3
```

### MIP (Mixed Integer Programming)

First, build the base image:
```bash
docker build -t mcp-base -f Dockerfile.base .
```

Then build the MIP solver:
```bash
docker-compose build mip
```

To run the solver:
```bash
docker run -v ./res:/app/res -v ./instances:/app/instances mcp-mip <start_instance> <end_instance>
```

For better container cleanup, use:
```bash
docker run --rm -v ./res:/app/res -v ./instances:/app/instances mcp-mip <start_instance> <end_instance> && docker container prune -f
```

Example:
```bash
docker run --rm -v ./res:/app/res -v ./instances:/app/instances mcp-mip 1 3 && docker container prune -f
```

## Results

Solutions are saved in the `res` directory, organized by solver type:
- `res/SMT/`: SMT solutions
- `res/SAT/`: SAT solutions
- `res/CP/`: CP solutions
- `res/MIP/`: MIP solutions

Each solution file is in JSON format and contains:
- Solving time
- Optimality status
- Objective value
- Solution (assignment of items to couriers)

### Solution Checker
To verify the correctness of the solutions, you can use the solution checker:
```bash
python3 solution_checker.py <instances_directory> <results_directory>
```

Example:
```bash
python3 solution_checker.py instances res/
```

## Graph Creation

The project includes a visualization tool that creates graphs showing the routes taken by each courier. The tool uses matplotlib to generate visual representations of the solutions.

### Usage
```bash
python graph_creator.py <implementation> <start_instance> <end_instance> <model>
```
Parameters:
- `implementation`: The solver implementation (cp/sat/smt/mip)
- `start_instance`: First instance to visualize (1-21)
- `end_instance`: Last instance to visualize (1-21)
- `model`: Model name as it appears in the solution files
Examples:
```bash
# Generate graphs for CP implementation with bs_gecode model, instances 1-10
python graph_creator.py cp 1 9 BS_Gec
# Generate graphs for SAT implementation with LNS_SYB model, instances 1-5
python graph_creator.py sat 1 9 LNS_SYB
# Generate graphs for SMT implementation with z3_smt_symbrk model, instances 1-5
python graph_creator.py smt 1 9 z3_smt_symbrk
# Generate graphs for MIP implementation with gurobipy model, instances 1-3
python graph_creator.py mip 1 9 gurobipy
#
```
The generated graphs will be saved in the `graphs/<IMPLEMENTATION>` directory, showing:
- Depot location (marked as Origin)
- Item locations (numbered points)
- Routes for each courier (colored arrows)
- Maximum distance achieved for each instance

## Authors
- Leonardo Mannini
- Lorenzo Sansone
