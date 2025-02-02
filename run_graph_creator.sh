
docker build -t mcp-graph -f Dockerfile.graph .

docker run --rm \
    -v "$(pwd)/instances:/app/instances:ro" \
    -v "$(pwd)/res:/app/res:ro" \
    -v "$(pwd)/graphs:/app/graphs" \
    mcp-graph "$@" 