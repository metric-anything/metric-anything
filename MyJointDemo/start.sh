#!/bin/sh
cat <<EOF > /usr/share/nginx/html/config.js
window.APP_CONFIG = {
  LLM_MODEL: "${LLM_MODEL:-qwen3:0.6b}"
};
EOF

# Use envsubst to inject the LLM proxy endpoint dynamically
export LLM_ENDPOINT="${LLM_ENDPOINT:-http://host.docker.internal:11434}"
envsubst '\$LLM_ENDPOINT' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf

exec nginx -g "daemon off;"
