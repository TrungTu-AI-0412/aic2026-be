# OpenVPN client + an OpenAI-compatible client, so the tunnel lives inside the
# container and the host's routing table is never touched.
#
# The .ovpn is NEVER baked in — it embeds a client private key. Mount it
# read-only at runtime instead.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        openvpn iproute2 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir openai==1.109.1 pyarrow==25.0.1 pillow==11.3.0

WORKDIR /work
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
