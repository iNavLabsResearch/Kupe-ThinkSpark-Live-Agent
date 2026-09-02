#!/usr/bin/env bash
# Expose the agent at a real domain (default: thinkspark.kupe.in) via Cloudflare Tunnel.
#
# Why a tunnel and not nginx: on RunPod (and most rented GPU hosts) inbound port 8000
# is never routed to your container, so no web server or DNS A record can help — the
# packets do not reach you. A tunnel dials OUT to Cloudflare, so nothing needs opening.
#
#   ./tunnel.sh quick     one-off trycloudflare.com URL, zero setup, no account
#   ./tunnel.sh setup     bind thinkspark.kupe.in permanently (needs a CF account)
#   ./tunnel.sh run       run an already-configured named tunnel
set -euo pipefail
cd "$(dirname "$0")"

DOMAIN="${DOMAIN:-thinkspark.kupe.in}"
PORT="${PORT:-8000}"
TUNNEL_NAME="${TUNNEL_NAME:-kupe-thinkspark}"

install_cloudflared() {
  command -v cloudflared >/dev/null 2>&1 && return
  echo "==> installing cloudflared"
  arch=$(uname -m)
  case "$arch" in
    x86_64|amd64) a=amd64 ;;
    aarch64|arm64) a=arm64 ;;
    *) echo "unsupported arch: $arch"; exit 1 ;;
  esac
  curl -fsSL -o /tmp/cloudflared \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${a}"
  chmod +x /tmp/cloudflared
  mv /tmp/cloudflared /usr/local/bin/cloudflared
}

case "${1:-quick}" in
  quick)
    install_cloudflared
    echo "==> starting a throwaway tunnel to localhost:${PORT}"
    echo "    look for the https://<random>.trycloudflare.com URL below,"
    echo "    then connect the UI to  wss://<that-host>/ws"
    echo
    cloudflared tunnel --url "http://localhost:${PORT}"
    ;;

  setup)
    install_cloudflared
    echo "==> logging in — opens a browser link, pick the kupe.in zone"
    cloudflared tunnel login
    cloudflared tunnel create "$TUNNEL_NAME" 2>/dev/null || \
      echo "    tunnel '$TUNNEL_NAME' already exists, reusing"
    echo "==> creating the DNS record for ${DOMAIN}"
    cloudflared tunnel route dns "$TUNNEL_NAME" "$DOMAIN"

    mkdir -p /etc/cloudflared
    id=$(cloudflared tunnel list --output json | python3 -c \
      "import sys,json;print(next(t['id'] for t in json.load(sys.stdin) if t['name']=='${TUNNEL_NAME}'))")
    cat > /etc/cloudflared/config.yml <<YML
tunnel: ${id}
credentials-file: /root/.cloudflared/${id}.json

ingress:
  - hostname: ${DOMAIN}
    service: http://localhost:${PORT}
    originRequest:
      # websockets stay open for the whole call; do not let the edge time them out
      noTLSVerify: true
      connectTimeout: 30s
  - service: http_status:404
YML
    echo "==> config written to /etc/cloudflared/config.yml"
    echo
    echo "    DNS is handled automatically — ${DOMAIN} now CNAMEs to ${id}.cfargotunnel.com"
    echo "    start it with:  ./tunnel.sh run"
    ;;

  run)
    install_cloudflared
    echo "==> ${DOMAIN} -> localhost:${PORT}"
    echo "    connect the UI to  wss://${DOMAIN}/ws"
    cloudflared tunnel --config /etc/cloudflared/config.yml run "$TUNNEL_NAME"
    ;;

  *)
    echo "usage: ./tunnel.sh [quick|setup|run]"; exit 1 ;;
esac
