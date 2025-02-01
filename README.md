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

To build a specific solver (e.g., SMT):
```bash
docker-compose build smt
```

## Running the Solvers

### CP (Constraint Programming)
```bash
docker run -v ./res:/app/res mcp-cp --model <model_name> <start_instance> <end_instance>
```

Available models:
- `CP_base`: Base CP model
- `CP_heu_chuffed`: CP with heuristics using Chuffed solver
- `CP_heu_sym_chuffed`: CP with heuristics and symmetry breaking using Chuffed
- `CP_heu_impl_chuffed`: CP with heuristics and implied constraints using Chuffed
- `CP_heu_sym_impl_chuffed`: CP with heuristics, symmetry breaking, and implied constraints using Chuffed
- `CP_heu_LNS`: CP with Large Neighborhood Search
- `CP_heu_LNS_sym`: CP with LNS and symmetry breaking
- `CP_heu_LNS_impl`: CP with LNS and implied constraints
- `CP_heu_LNS_sym_impl`: CP with LNS, symmetry breaking, and implied constraints

Example:
```bash
docker run -v ./res:/app/res mcp-cp --model CP_base 1 3
```

### SAT (Boolean Satisfiability)
```bash
docker run -v ./res:/app/res mcp-sat <model_name> <start_instance> <end_instance>
```

Available models:
- `LNS`: Linear Neighborhood Search
- `LNS_SYB`: Linear Neighborhood Search with Symmetry Breaking
- `BNS`: Binary Neighborhood Search
- `BNS_SYB`: Binary Neighborhood Search with Symmetry Breaking

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
```bash
docker run -v ./res:/app/res mcp-mip --model <model_name> <start_instance> <end_instance>
```

Available models:
- `mip_base_bounded`: Base MIP model with bounded variables
- `mip_base_bounded_penaltyterm`: MIP with bounded variables and penalty terms
- `mip_base_bounded_penaltyterm_symbrk`: MIP with bounded variables, penalty terms, and symmetry breaking
- `cluster-first_route-second`: Two-phase approach: clustering first, then routing

Example:
```bash
docker run -v ./res:/app/res mcp-mip --model mip_base_bounded 1 3
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