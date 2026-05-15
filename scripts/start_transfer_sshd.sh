#!/usr/bin/env bash

set -euo pipefail

user="${VBOGS_TRANSFER_USER:-vbogs}"
uid="${VBOGS_TRANSFER_UID:-1000}"
gid="${VBOGS_TRANSFER_GID:-1000}"
port="${VBOGS_TRANSFER_PORT:-2222}"
keys="${VBOGS_TRANSFER_AUTHORIZED_KEYS:-}"
keys_file="${VBOGS_TRANSFER_AUTHORIZED_KEYS_FILE:-}"

if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
  echo "error: VBOGS_TRANSFER_PORT must be an integer from 1 to 65535" >&2
  exit 2
fi

if [[ -z "$keys" && -z "$keys_file" ]]; then
  echo "error: set VBOGS_TRANSFER_AUTHORIZED_KEYS or VBOGS_TRANSFER_AUTHORIZED_KEYS_FILE" >&2
  exit 2
fi

if [[ -n "$keys_file" && ! -f "$keys_file" ]]; then
  echo "error: authorized keys file not found: $keys_file" >&2
  exit 2
fi

if [[ "$user" != "root" ]] && ! id "$user" >/dev/null 2>&1; then
  if ! getent group "$user" >/dev/null 2>&1; then
    groupadd --gid "$gid" "$user" 2>/dev/null || groupadd "$user"
  fi
  useradd --uid "$uid" --gid "$user" --create-home --shell /bin/bash "$user" 2>/dev/null \
    || useradd --gid "$user" --create-home --shell /bin/bash "$user"
fi

home_dir="$(getent passwd "$user" | cut -d: -f6)"
if [[ -z "$home_dir" ]]; then
  echo "error: unable to resolve home directory for transfer user: $user" >&2
  exit 2
fi

install -d -m 700 -o "$user" -g "$user" "$home_dir/.ssh"
authorized_keys="$home_dir/.ssh/authorized_keys"
: > "$authorized_keys"

if [[ -n "$keys_file" ]]; then
  cat "$keys_file" >> "$authorized_keys"
  printf '\n' >> "$authorized_keys"
fi

if [[ -n "$keys" ]]; then
  printf '%s\n' "$keys" >> "$authorized_keys"
fi

if ! grep -Eq '^[[:space:]]*(ssh-(rsa|ed25519)|ecdsa-sha2-nistp(256|384|521)) ' "$authorized_keys"; then
  echo "error: authorized keys do not contain a supported public key" >&2
  exit 2
fi

chown "$user:$user" "$authorized_keys"
chmod 600 "$authorized_keys"

mkdir -p /run/sshd /etc/ssh/sshd_config.d
ssh-keygen -A >/dev/null

cat > /etc/ssh/sshd_config.d/vbogs-transfer.conf <<EOF
Port $port
ListenAddress 0.0.0.0
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
PermitEmptyPasswords no
X11Forwarding no
AllowTcpForwarding no
PermitTunnel no
AllowAgentForwarding no
PermitRootLogin prohibit-password
AllowUsers $user
EOF

echo "Starting VBOGS SSH/SFTP transfer service on port $port for user $user" >&2
echo "Mounted outputs are available under /workspace/VBOGS/outputs" >&2
exec /usr/sbin/sshd -D -e
