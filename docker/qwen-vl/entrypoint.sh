#!/usr/bin/env bash
# Bring the tunnel up, wait for it to actually carry traffic, then exec the
# command. Waiting on "Initialization Sequence Completed" rather than sleeping
# means a slow handshake delays the run instead of failing it.
set -euo pipefail

CONFIG=${VPN_CONFIG:-/vpn/client.ovpn}
PROBE=${VPN_PROBE:-}
TIMEOUT=${VPN_TIMEOUT:-60}

[ -r "$CONFIG" ] || { echo "no readable VPN config at $CONFIG" >&2; exit 1; }

openvpn --config "$CONFIG" --daemon --log /tmp/openvpn.log --verb 3
for _ in $(seq "$TIMEOUT"); do
    grep -q "Initialization Sequence Completed" /tmp/openvpn.log && break
    sleep 1
done
if ! grep -q "Initialization Sequence Completed" /tmp/openvpn.log; then
    echo "--- VPN did not come up in ${TIMEOUT}s ---" >&2
    tail -25 /tmp/openvpn.log >&2
    exit 1
fi
echo "VPN up: $(ip -4 addr show dev tun0 2>/dev/null | awk '/inet /{print $2}')" >&2

if [ -n "$PROBE" ]; then
    echo "probing $PROBE" >&2
    curl -fsS -m 15 "$PROBE" >&2 || { echo "probe FAILED" >&2; exit 1; }
    echo >&2
fi

exec "$@"
