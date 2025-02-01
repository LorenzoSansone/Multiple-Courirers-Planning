FROM mcp-base

# Install Gurobi
RUN pip install gurobipy==11.0.3

WORKDIR /app

# Create directory for Gurobi license
RUN mkdir -p /root/.gurobi/

# Copy the WLS license file
COPY MIP/gurobi.lic /root/.gurobi/gurobi.lic

# Set Gurobi environment variables for WLS
ENV GRB_WLSACCESSID=eda3381c-425f-40ec-b4d9-3a04d6b868e5
ENV GRB_WLSSECRET=c908f0de-3d90-4fdd-b186-fbbb45b1572e
ENV GRB_LICENSEID=2616230

# Set Gurobi environment variables
ENV GRB_LICENSE_FILE=/root/.gurobi/gurobi.lic
ENV GUROBI_HOME=/usr/local/lib/python3.11/site-packages/gurobipy
ENV PATH=$PATH:$GUROBI_HOME/bin
ENV LD_LIBRARY_PATH=$GUROBI_HOME/lib

# Copy the Python script
COPY MIP/mip_gurobi.py /app/MIP/

# Set the default command to use Python-based Gurobi solver
ENTRYPOINT ["python3", "MIP/mip_gurobi.py"]