#! /bin/sh

set -e

# Stop the ollama server
docker compose -f secure-ollama-server/docker-compose.yaml -f secure-ollama-server/docker-compose.gpu.yaml --env-file secure-ollama-server/.env down

# wait a bit
sleep 1

# Start the server again
docker compose -f secure-ollama-server/docker-compose.yaml -f secure-ollama-server/docker-compose.gpu.yaml --env-file secure-ollama-server/.env up -d

# finally wait a bit
sleep 1