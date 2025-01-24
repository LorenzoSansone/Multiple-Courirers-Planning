#Multiple Couriers Planning
This repository contains the solution of the Multiple Couriers Planning (MCP) problem, developed as a group project using several technique (SAT, MIP, SMT, CP) 
for the Combinatorial Decision Making and Optimization 2023/2024 exam at Alma Mater Studiorum (Unibo) by Leonardo Mannini and Lorenzo Sansone

##Docker 
In order to reproduce and launch the experiment you have to run the following commands:
```bash
docker build --no-cache -t mcp-solver .
```
In order to run a specific model follow the instructions:
```bash
docker run mcp-solver <model_type> <model> <instances>
```

##Authors
Leonardo Mannini
Lorenzo Sansone
