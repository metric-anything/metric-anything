#!/bin/sh
cat <<EOF > /usr/share/nginx/html/config.js
window.APP_CONFIG = {
  LLM_MODEL: "${LLM_MODEL:-qwen2.5:0.5b}"
};
EOF

exec nginx -g "daemon off;"
