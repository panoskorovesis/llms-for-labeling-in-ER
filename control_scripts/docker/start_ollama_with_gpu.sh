#!/bin/sh

set -e

docker compose -f secure-ollama-server/docker-compose.yaml -f secure-ollama-server/docker-compose.gpu.yaml --env-file secure-ollama-server/.env up -d
