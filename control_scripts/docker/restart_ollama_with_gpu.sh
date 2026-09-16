#! /bin/sh

set -e

# Stop the ollama server
docker compose -f secure-ollama-server/docker-compose.yaml -f secure-ollama-server/docker-compose.gpu.yaml --env-file secure-ollama-server/.env down

# wait a bit
sleep 1

# Start the server again
docker compose -f secure-ollama-server/docker-compose.yaml -f secure-ollama-server/docker-compose.gpu.yaml --env-file secure-ollama-server/.env up -d

# Wait until Caddy can reach the Ollama API.
. secure-ollama-server/.env
attempt=0
until curl --fail --silent --show-error --connect-timeout 1 --max-time 2 \
	-u "$CADDY_USERNAME:$CADDY_PASSWORD" \
	http://127.0.0.1:8200/api/tags >/dev/null; do
	attempt=$((attempt + 1))
	if [ "$attempt" -ge 60 ]; then
		echo "Ollama did not become ready within 60 seconds." >&2
		exit 1
	fi
	sleep 1
done

echo "Ollama is ready."