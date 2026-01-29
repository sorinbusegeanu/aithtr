FROM python:3.11-slim

WORKDIR /app

COPY mcp_servers/memory/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

ENV MEMORY_DB_PATH=/data/sqlite/memory.db \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=7102 \
    MCP_PATH=/mcp

CMD ["python", "-m", "mcp_servers.memory.mcp_server"]
