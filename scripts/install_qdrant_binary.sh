#!/usr/bin/env bash
set -euo pipefail

# 安装参数可通过环境变量覆盖
QDRANT_VERSION="${QDRANT_VERSION:-v1.15.4}"
QDRANT_ARCH="${QDRANT_ARCH:-x86_64-unknown-linux-gnu}"
QDRANT_DATA_DIR="${QDRANT_DATA_DIR:-/var/lib/qdrant}"
QDRANT_INSTALL_DIR="${QDRANT_INSTALL_DIR:-/opt/qdrant}"
QDRANT_HTTP_PORT="${QDRANT_HTTP_PORT:-6333}"
QDRANT_GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"

ARCHIVE="qdrant-${QDRANT_ARCH}.tar.gz"
BASE_URL="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}"
TMP_ARCHIVE="/tmp/${ARCHIVE}"
TMP_DIR="/tmp/qdrant-release"

if [[ $EUID -ne 0 ]]; then
  echo "[!] 请使用 root 权限执行，以便写入 ${QDRANT_INSTALL_DIR} 与 ${QDRANT_DATA_DIR}" >&2
  exit 1
fi

mkdir -p "${QDRANT_INSTALL_DIR}" "${QDRANT_DATA_DIR}"
rm -rf "${TMP_DIR}"

if ! command -v curl >/dev/null 2>&1; then
  echo "[*] 安装 curl..."
  apt-get update -y
  apt-get install -y curl
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "[*] 安装 tar..."
  apt-get update -y
  apt-get install -y tar
fi

echo "[*] 下载 Qdrant ${QDRANT_VERSION} (arch=${QDRANT_ARCH})"
curl -fL --retry 5 --retry-delay 5 --retry-all-errors \
  "${BASE_URL}/${ARCHIVE}" -o "${TMP_ARCHIVE}"

mkdir -p "${TMP_DIR}"
tar -xzf "${TMP_ARCHIVE}" -C "${TMP_DIR}" --strip-components=0

if [[ -f "${TMP_DIR}/qdrant" ]]; then
  install -m 0755 "${TMP_DIR}/qdrant" "${QDRANT_INSTALL_DIR}/qdrant"
elif [[ -f "${TMP_DIR}/bin/qdrant" ]]; then
  install -m 0755 "${TMP_DIR}/bin/qdrant" "${QDRANT_INSTALL_DIR}/qdrant"
else
  echo "[!] 未在解压目录找到 qdrant 可执行文件" >&2
  exit 1
fi

cat >"${QDRANT_INSTALL_DIR}/config.default.yaml" <<'EOF'
# 默认配置示例，可根据需要修改后覆盖 config.yaml
storage:
  path: /var/lib/qdrant/storage
  on_disk: true
  wal_path: /var/lib/qdrant/wal
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
cluster:
  enabled: false
telemetry:
  enabled: false
EOF

cat >"${QDRANT_INSTALL_DIR}/config.yaml" <<EOF
# 自动生成的 Qdrant 配置，可按需修改
storage:
  path: ${QDRANT_DATA_DIR}/storage
  on_disk: true
  wal_path: ${QDRANT_DATA_DIR}/wal
service:
  host: 0.0.0.0
  http_port: ${QDRANT_HTTP_PORT}
  grpc_port: ${QDRANT_GRPC_PORT}
cluster:
  enabled: false
telemetry:
  enabled: false
EOF

chmod 600 "${QDRANT_INSTALL_DIR}/config.yaml"

cat >"${QDRANT_INSTALL_DIR}/run_qdrant.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config.yaml"
BIN_PATH="${SCRIPT_DIR}/qdrant"

echo "[*] 使用配置 ${CONFIG_PATH} 启动 Qdrant..."
exec "${BIN_PATH}" --config-path "${CONFIG_PATH}"
EOF

chmod 750 "${QDRANT_INSTALL_DIR}/run_qdrant.sh"

cat <<EOF
[+] Qdrant 已安装到 ${QDRANT_INSTALL_DIR}
    数据目录：${QDRANT_DATA_DIR}
    HTTP 端口：${QDRANT_HTTP_PORT}
    gRPC 端口：${QDRANT_GRPC_PORT}

使用方法：
  ${QDRANT_INSTALL_DIR}/run_qdrant.sh

如需修改端口或目录，可编辑 ${QDRANT_INSTALL_DIR}/config.yaml 或重跑脚本并传入新环境变量。
EOF
