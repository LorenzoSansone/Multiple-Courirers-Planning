#!/bin/bash

# Build the graph creator image
docker build -t mcp-graph -f Dockerfile.graph .

# Run the graph creator with the provided arguments
docker run --rm \
    -v "$(pwd)/instances:/app/instances:ro" \
    -v "$(pwd)/res:/app/res:ro" \
    -v "$(pwd)/graphs:/app/graphs" \
    mcp-graph "$@" 