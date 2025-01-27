FROM mcp-base

# Install Gurobi
RUN pip install gurobipy==11.0.3

WORKDIR /app

# Copy the license file to the root directory
COPY MIP/gurobi.lic /root/gurobi.lic

ENTRYPOINT ["python", "MIP/mip_gurobi.py"] 