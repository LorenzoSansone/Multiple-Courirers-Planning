#!/bin/bash

# Build all images
echo "Building all solver images..."
docker build -t mcp-base -f Dockerfile.base .
docker-compose build
docker build . -t image_cp -f Dockerfile_cp

# cP 
echo "Running CP models..."
docker run -v ./res:/app/res image_cp bs_gecode 1 21 &&
docker run -v ./res:/app/res image_cp bs_chuffed 1 21 &&
docker run -v ./res:/app/res image_cp bs_heu_gecode 1 21 &&
docker run -v ./res:/app/res image_cp bs_heu_chuffed 1 21 &&
docker run -v ./res:/app/res image_cp bs_heu_impl_gecode 1 21 &&
docker run -v ./res:/app/res image_cp bs_heu_impl_chuffed 1 21 &&
docker run -v ./res:/app/res image_cp bs_heu_sym_gecode 1 21 &&
docker run -v ./res:/app/res image_cp bs_heu_sym_chuffed 1 21 &&
docker run -v ./res:/app/res image_cp bs_heu_sym_impl_gecode 1 21 &&
docker run -v ./res:/app/res image_cp bs_heu_sym_impl_chuffed 1 21

# SAT
echo "Running SAT models..."
docker run -v ./res:/app/res mcp-sat LNS 1 21 &&
docker run -v ./res:/app/res mcp-sat LNS_SYB 1 21 &&
docker run -v ./res:/app/res mcp-sat BNS 1 21 &&
docker run -v ./res:/app/res mcp-sat BNS_SYB 1 21

# SMT 
echo "Running SMT models..."
docker run -v ./res:/app/res mcp-smt --model z3_smt_base 1 21 &&
docker run -v ./res:/app/res mcp-smt --model z3_smt_symbrk 1 21 &&
docker run -v ./res:/app/res mcp-smt --model z3_smt_symbrk_implconstr 1 21 &&
docker run -v ./res:/app/res mcp-smt --model z3_smt_symbrk_binarysearch 1 21

#  MIP model
echo "Running MIP model..."
docker run --rm -v ./res:/app/res -v ./instances:/app/instances mcp-mip 1 21

# Clean up containers
echo "Cleaning up..."
docker container prune -f

echo "All models have been executed on all instances." 