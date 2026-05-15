#!/usr/bin/env bash

set -euo pipefail

user="vbogs"
uid="1000"
gid="1000"
port="2222"
keys="${VBOGS_TRANSFER_AUTHORIZED_KEYS:-}"

if [[ -z "$keys" ]]; then
  echo "error: set VBOGS_TRANSFER_AUTHORIZED_KEYS to one or more SSH public keys" >&2
  exit 2
fi

if ! id "$user" >/dev/null 2>&1; then
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

printf '%s\n' "$keys" >> "$authorized_keys"

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
