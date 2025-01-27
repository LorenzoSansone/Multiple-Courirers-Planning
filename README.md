# Multiple Couriers Planning

This project implements different approaches to solve the Multiple Couriers Planning problem using various solving techniques:
- SMT (Satisfiability Modulo Theories)
- SAT (Boolean Satisfiability)
- CP (Constraint Programming)
- MIP (Mixed Integer Programming)

## Docker Setup

The project includes Docker support for all implementations. Each solver has its own Docker image, and they can be built and run independently.

### Building the Images

To build all solver images:
```bash
docker-compose build
```

To build a specific solver (e.g., SMT):
```bash
docker-compose build smt
```

### Running the Solvers

Each solver can be run using Docker with the following format:

#### SMT Solver
```bash
docker run -v ./res:/app/res mcp-smt --model <model_name> <start_instance> <end_instance>
```
Available models:
- z3_smt_base
- z3_smt_symbrk
- z3_smt_symbrk_implconstr
- z3_smt_symbrk_binarysearch

Example:
```bash
docker run -v ./res:/app/res mcp-smt --model z3_smt_base 1 3
```

#### SAT Solver
```bash
docker run -v ./res:/app/res mcp-sat --model <model_name> <start_instance> <end_instance>
```

#### CP Solver
```bash
docker run -v ./res:/app/res mcp-cp --model <model_name> <start_instance> <end_instance>
```

#### MIP Solver
```bash
docker run -v ./res:/app/res mcp-mip --model <model_name> <start_instance> <end_instance>
```

### Results

Solutions are saved in the `res` directory, organized by solver type:
- SMT solutions: `res/SMT/`
- SAT solutions: `res/SAT/`
- CP solutions: `res/CP/`
- MIP solutions: `res/MIP/`

Each solution file is in JSON format and contains:
- Solving time
- Optimality status
- Objective value
- Solution (assignment of items to couriers)

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
