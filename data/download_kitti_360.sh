#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${KITTI_360_DATA_ROOT:-"${SCRIPT_DIR}/KITTI-360"}"
DOWNLOADS_DIR="${KITTI_360_DOWNLOADS_DIR:-"${DATA_ROOT}/_downloads"}"

usage() {
  cat <<'USAGE'
Download and extract KITTI-360 archives into data/KITTI-360.

Required environment variables:
  KITTI_CALIBRATION_LINK   URL for KITTI-360 calibration archive
  KITTI_POSES_LINK         URL for KITTI-360 poses archive
  KITTI_IMAGES             URL(s) for KITTI-360 image archive(s)

KITTI_IMAGES may contain multiple URLs separated by spaces, commas, or newlines.

Optional:
  KITTI_360_DATA_ROOT       Extraction root. Default: ./KITTI-360 beside this script
  KITTI_360_DOWNLOADS_DIR   Archive cache. Default: <KITTI_360_DATA_ROOT>/_downloads
  FORCE_DOWNLOAD=1          Re-download archives even when cached
  KEEP_ARCHIVES=0           Delete cached archives after extraction

Example:
  export KITTI_CALIBRATION_LINK='https://.../calibration.zip'
  export KITTI_POSES_LINK='https://.../data_poses.zip'
  export KITTI_IMAGES='https://.../data_2d_raw_2013_05_28_drive_0008_sync.zip'
  ./download_kitti_360.sh
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

get_env_first() {
  local name value
  for name in "$@"; do
    value="$(printenv "${name}" 2>/dev/null || true)"
    if [[ -n "${value}" ]]; then
      printf '%s' "${value}"
      return 0
    fi
  done
  return 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

archive_name_from_url() {
  local url="$1"
  local no_query base
  no_query="${url%%\?*}"
  no_query="${no_query%%#*}"
  base="${no_query##*/}"
  [[ -n "${base}" ]] || die "could not infer archive name from URL: ${url}"
  printf '%s' "${base}"
}

download_archive() {
  local url="$1"
  local destination="$2"

  if [[ -f "${destination}" && "${FORCE_DOWNLOAD:-0}" != "1" ]]; then
    echo "Using cached archive: ${destination}"
    return 0
  fi

  mkdir -p "$(dirname "${destination}")"
  echo "Downloading: ${url}"

  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --show-error --progress-bar --output "${destination}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget --output-document="${destination}" "${url}"
  else
    die "either curl or wget is required to download archives"
  fi
}

extract_archive() {
  local archive="$1"
  local destination="$2"

  mkdir -p "${destination}"
  echo "Extracting: ${archive}"

  case "${archive}" in
    *.zip)
      require_command unzip
      unzip -oq "${archive}" -d "${destination}"
      ;;
    *.tar)
      tar -xf "${archive}" -C "${destination}"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "${archive}" -C "${destination}"
      ;;
    *.tar.bz2|*.tbz|*.tbz2)
      tar -xjf "${archive}" -C "${destination}"
      ;;
    *.tar.xz|*.txz)
      tar -xJf "${archive}" -C "${destination}"
      ;;
    *)
      die "unsupported archive format: ${archive}"
      ;;
  esac
}

split_urls() {
  tr ',\n' '  ' | xargs -n 1 printf '%s\n'
}

process_url() {
  local label="$1"
  local url="$2"
  local archive_name archive_path

  [[ -n "${url}" ]] || die "missing URL for ${label}"
  archive_name="$(archive_name_from_url "${url}")"
  archive_path="${DOWNLOADS_DIR}/${archive_name}"

  echo
  echo "==> ${label}"
  download_archive "${url}" "${archive_path}"
  extract_archive "${archive_path}" "${DATA_ROOT}"

  if [[ "${KEEP_ARCHIVES:-1}" == "0" ]]; then
    rm -f "${archive_path}"
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

calibration_url="$(get_env_first KITTI_CALIBRATION_LINK || true)"
poses_url="$(get_env_first KITTI_POSES_LINK || true)"
images_urls="$(get_env_first KITTI_IMAGES || true)"

missing=()
[[ -n "${calibration_url}" ]] || missing+=("KITTI_CALIBRATION_LINK")
[[ -n "${poses_url}" ]] || missing+=("KITTI_POSES_LINK")
[[ -n "${images_urls}" ]] || missing+=("KITTI_IMAGES")

if (( ${#missing[@]} > 0 )); then
  usage >&2
  echo >&2
  die "missing required environment variable(s): ${missing[*]}"
fi

mkdir -p "${DATA_ROOT}" "${DOWNLOADS_DIR}"

echo "KITTI-360 data root : ${DATA_ROOT}"
echo "Archive cache       : ${DOWNLOADS_DIR}"

process_url "calibration" "${calibration_url}"
process_url "poses" "${poses_url}"

while IFS= read -r image_url; do
  [[ -n "${image_url}" ]] || continue
  process_url "images" "${image_url}"
done < <(printf '%s\n' "${images_urls}" | split_urls)

echo
echo "KITTI-360 downloads extracted into ${DATA_ROOT}"
