FROM mcp-base

RUN pip install gurobipy==11.0.3

WORKDIR /app

COPY MIP/gurobi.lic /root/gurobi.lic

# Create directory for Gurobi license
RUN mkdir -p /root/.gurobi/

# Set the default command to use minizinc_mip.py
ENTRYPOINT ["python3", "MIP/minizinc_mip.py"]