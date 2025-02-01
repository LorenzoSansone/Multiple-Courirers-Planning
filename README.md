# Multiple Couriers Planning

This project implements different approaches to solve the Multiple Couriers Planning problem using various solving techniques:
- SMT (Satisfiability Modulo Theories)
- SAT (Boolean Satisfiability)
- CP (Constraint Programming)
- MIP (Mixed Integer Programming)

## Docker Setup

To build all solver images:
```bash
docker-compose build
```

## Running the Solvers

### CP (Constraint Programming)
```bash
docker run -v ./res:/app/res mcp-cp --model <model_name> <start_instance> <end_instance>
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
docker run -v ./res:/app/res mcp-cp --model bs_gecode 1 3
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

## Project Structure
```
.
├── CP/             # Constraint Programming implementation
├── SAT/            # Boolean Satisfiability implementation
├── SMT/            # SMT implementation
├── MIP/            # Mixed Integer Programming implementation
├── res/            # Results directory
│   ├── CP/        # CP solutions
│   ├── SAT/       # SAT solutions
│   ├── SMT/       # SMT solutions
│   └── MIP/       # MIP solutions
└── instances/      # Problem instances
```

## Authors
- Leonardo Mannini
- Lorenzo Sansone
