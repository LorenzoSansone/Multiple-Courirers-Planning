FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    minizinc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install packages - match exact versions from local environment
RUN pip install --upgrade pip && \
    pip install --no-cache-dir z3==0.2.0 && \
    pip install --no-cache-dir z3-solver==4.13.3.0 && \
    pip install --no-cache-dir minizinc==0.9.0 && \
    pip install --no-cache-dir -r requirements.txt

# Verify Z3 installation
RUN python -c "import z3; z3.Solver()"

# Copy the project files
COPY . .

# Create symbolic links to instances directory in each solver directory
RUN ln -s /app/instances /app/MIP/instances && \
    ln -s /app/instances /app/SMT/instances && \
    ln -s /app/instances /app/CP/instances && \
    ln -s /app/instances /app/SAT/instances

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
\n\
usage() {\n\
    echo "Usage:"\n\
    echo "  Single instance:   docker run <image> <model_type> <implementation> <instance_number>"\n\
    echo "  Instance range:    docker run <image> <model_type> <implementation> <start_instance> <end_instance>"\n\
    echo "\n"\n\
    echo "Model types: sat, smt, cp, mip"\n\
    echo "Implementation: name of the implementation (e.g., gurobi_base_bounded_second)"\n\
    echo "Instance numbers: 01-21"\n\
    exit 1\n\
}\n\
\n\
if [ "$#" -lt 3 ]; then\n\
    usage\n\
fi\n\
\n\
MODEL_TYPE=$1\n\
IMPLEMENTATION=$2\n\
START_INSTANCE=$3\n\
END_INSTANCE=${4:-$START_INSTANCE}\n\
\n\
# Function to pad number with leading zero if needed\n\
pad_number() {\n\
    printf "%02d" $1\n\
}\n\
\n\
# Function to run solver for a single instance\n\
run_solver() {\n\
    instance=$(pad_number $1)\n\
    case $MODEL_TYPE in\n\
        "mip")\n\
            # Remove .mzn extension if present\n\
            IMPL_NAME=${IMPLEMENTATION%.mzn}\n\
            cd MIP && PYTHONPATH=/app python minizinc_mip.py --model $IMPL_NAME --mzn-dir /app/MIP $START_INSTANCE $END_INSTANCE\n\
            ;;\n\
        "smt")\n\
            cd SMT && PYTHONPATH=/app python smtsolver.py --model $IMPLEMENTATION $START_INSTANCE $END_INSTANCE\n\
            ;;\n\
        "cp")\n\
            cd CP && python cp_solver_all_model.py $IMPLEMENTATION $instance\n\
            ;;\n\
        "sat")\n\
            cd SAT && python model_sat.py $IMPLEMENTATION $instance\n\
            ;;\n\
        *)\n\
            echo "Invalid model type. Use: sat, smt, cp, or mip"\n\
            exit 1\n\
            ;;\n\
    esac\n\
}\n\
\n\
# For MIP and SMT, we handle the range differently as their scripts expect both start and end\n\
if [ "$MODEL_TYPE" = "mip" ] || [ "$MODEL_TYPE" = "smt" ]; then\n\
    run_solver $START_INSTANCE\n\
else\n\
    # For other solvers, we loop through the range\n\
    for (( i=$START_INSTANCE; i<=$END_INSTANCE; i++ ))\n\
    do\n\
        run_solver $i\n\
    done\n\
fi\n\
' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
