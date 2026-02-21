#!/bin/sh

# If LLM_MODEL is not explicitly set, try to scrape it from the docker-compose.yml file if mounted
if [ -z "$LLM_MODEL" ] && [ -f /app/docker-compose.yml ]; then
    # Parse the command: -m /models/qwen2.5-1.5b-instruct-q4_k_m.gguf ...
    # Extract just the filename without the path and the .gguf extension
    EXTRACTED_MODEL=$(grep -A 10 "llm-service:" /app/docker-compose.yml | grep "\-m /models/" | sed -E 's|.*-m /models/([^ ]+)\.gguf.*|\1|')
    if [ -n "$EXTRACTED_MODEL" ]; then
        LLM_MODEL="$EXTRACTED_MODEL"
    fi
fi

# Use envsubst to inject the LLM proxy endpoint dynamically (moved up so we can curl it)
export LLM_ENDPOINT="${LLM_ENDPOINT:-http://host.docker.internal:11434}"

# If scraping AlmaLinux compose file failed, try probing the local Mac/Windows Ollama API
if [ -z "$LLM_MODEL" ]; then
    # Try fetching the first model from Ollama's /api/tags JSON output using basic tools (no jq required)
    # Output looks like: {"models":[{"name":"qwen2.5:1.5b", ...
    OLLAMA_MODEL=$(wget -qO- "$LLM_ENDPOINT/api/tags" | grep -o '"name":"[^"]*"' | head -1 | sed 's/"name":"\([^"]*\)"/\1/')
    if [ -n "$OLLAMA_MODEL" ]; then
        LLM_MODEL="$OLLAMA_MODEL"
    fi
fi

# Absolute final fallback if both scraping and Ollama probing fail
LLM_MODEL="${LLM_MODEL:-qwen2.5:1.5b}"

cat <<EOF > /usr/share/nginx/html/config.js
window.APP_CONFIG = {
  LLM_MODEL: "$LLM_MODEL"
};
EOF

envsubst '\$LLM_ENDPOINT' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf

exec nginx -g "daemon off;"
