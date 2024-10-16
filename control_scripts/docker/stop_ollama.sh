#!/bin/sh

set -e

docker compose -f secure-ollama-server/docker-compose.yaml --env-file secure-ollama-server/.env down
