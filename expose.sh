#!/usr/bin/env bash
# Optional public expose via nginx on port 80.
#
# Use this when the host (Vast.ai, a colo, a local GPU box) actually routes inbound
# traffic to this machine. Do NOT use it on RunPod-style hosts where port 80 never
# reaches the container — use ./tunnel.sh or SSH -L instead.
#
#   ./expose.sh           install nginx, proxy :80 -> :8000 (WebSocket-safe)
#   ./expose.sh --remove  stop and disable the site
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
LISTEN="${LISTEN:-80}"
CONF_SRC="$(pwd)/nginx/kupe-agent.conf"

if [[ "${1:-}" == "--remove" ]]; then
  rm -f /etc/nginx/sites-enabled/kupe-agent /etc/nginx/conf.d/kupe-agent.conf
  nginx -s reload 2>/dev/null || true
  echo "==> nginx site removed. Agent still listens on :${PORT}"
  exit 0
fi

if ! command -v nginx >/dev/null 2>&1; then
  echo "==> installing nginx"
  apt-get update -qq && apt-get install -y -qq nginx
fi

# Drop the distro default so :80 is ours.
rm -f /etc/nginx/sites-enabled/default

if [[ -d /etc/nginx/sites-available ]]; then
  cp "$CONF_SRC" /etc/nginx/sites-available/kupe-agent
  ln -sfn /etc/nginx/sites-available/kupe-agent /etc/nginx/sites-enabled/kupe-agent
  TARGET=/etc/nginx/sites-available/kupe-agent
else
  mkdir -p /etc/nginx/conf.d
  cp "$CONF_SRC" /etc/nginx/conf.d/kupe-agent.conf
  TARGET=/etc/nginx/conf.d/kupe-agent.conf
fi

sed -i "s/listen 80 /listen ${LISTEN} /; s/127.0.0.1:8000/127.0.0.1:${PORT}/" "$TARGET"

nginx -t
nginx -s reload 2>/dev/null || nginx

ip=$(curl -fsS --max-time 2 https://api.ipify.org 2>/dev/null || echo "<public-ip>")
echo
echo "==> nginx :${LISTEN} -> 127.0.0.1:${PORT}  (WebSockets on)"
echo "    UI:      ws://${ip}/ws"
echo "    health:  http://${ip}/health"
echo "    open port ${LISTEN} in the provider panel if curl from your laptop fails"
echo
echo "    Vast / direct 8000 already published? skip nginx, paste ws://${ip}:${PORT}/ws"
